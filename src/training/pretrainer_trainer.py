"""
Continued pretraining (causal LM + LoRA). Experiment: tehr_stage1.
"""
import math
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import torch
import time
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from src.training.training_utils import apply_hf_step_limit, extract_and_pack, init_wandb
from src.data.dataset_factory import get_dataset
from src.training.eval_schedule import append_max_runtime_callback

import logging
logger = logging.getLogger(__name__)


class Rank0EvalSFTTrainer(SFTTrainer):
    """
    SFTTrainer that runs mid-training eval on rank 0 only.

    Multi-GPU Unsloth eval at long ``max_length`` can desync NCCL (train skips
    dense attention masks; eval builds them). Rank 0 evaluates the unwrapped
    model with a plain local loop; other ranks skip the forward pass and
    receive the metrics.

    Two constraints drive this implementation:

    - ``Trainer.evaluation_loop`` cannot be reused on rank 0. Accelerate's
      gather helpers read ``PartialState``, which is a different shared-state
      singleton from ``accelerator.state``, so overriding ``num_processes``
      there does not stop the ``all_gather`` in ``gather_for_metrics``. Rank 0
      then blocks in an all-gather while the other ranks block in the metrics
      broadcast, and the NCCL watchdog aborts the job.
    - Metrics are shared over a separate gloo group. Rank 0's eval takes far
      longer than the 600 s NCCL watchdog timeout, and a NCCL collective left
      pending that long is killed even when the ranks agree on it.
    """

    def __init__(
        self,
        *args,
        eval_max_batches: Optional[int] = None,
        eval_share_timeout_hours: float = 12.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_max_batches = (
            int(eval_max_batches) if eval_max_batches else None
        )
        self._eval_share_timeout = timedelta(hours=float(eval_share_timeout_hours))
        self._eval_share_group = self._build_share_group()

    def _is_distributed(self) -> bool:
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and int(getattr(self.args, "world_size", 1) or 1) > 1
        )

    def _build_share_group(self):
        """Gloo group for metric hand-off, outside the NCCL watchdog."""
        if not self._is_distributed():
            return None
        try:
            group = torch.distributed.new_group(
                backend="gloo", timeout=self._eval_share_timeout
            )
            logger.info(
                "eval_on_main_process_only: created gloo group for metric "
                f"broadcast (timeout {self._eval_share_timeout})"
            )
            return group
        except Exception as exc:
            logger.warning(
                f"Could not create gloo group for metric broadcast ({exc}); "
                "falling back to the default process group. Rank 0 eval must "
                "then finish inside the NCCL watchdog timeout."
            )
            return None

    @torch.no_grad()
    def _local_eval(self, eval_dataset, metric_key_prefix: str) -> Dict[str, Any]:
        """Single-process eval loop: no accelerate, no DDP, no collectives."""
        from torch.utils.data import DataLoader

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=int(self.args.per_device_eval_batch_size),
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

        n_batches = len(loader)
        if self.eval_max_batches:
            n_batches = min(n_batches, self.eval_max_batches)
        logger.info(
            f"eval_on_main_process_only: rank 0 evaluating {n_batches} batches "
            f"of {len(loader)} (eval_max_batches={self.eval_max_batches})"
        )

        device = self.args.device
        total_loss = 0.0
        seen = 0
        start = time.time()
        for step, batch in enumerate(loader):
            if step >= n_batches:
                break
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            total_loss += float(loss.detach().item())
            seen += 1

        if was_training:
            model.train()

        runtime = time.time() - start
        mean_loss = total_loss / max(1, seen)
        metrics = {
            f"{metric_key_prefix}_loss": mean_loss,
            f"{metric_key_prefix}_perplexity": float(math.exp(min(mean_loss, 20.0))),
            f"{metric_key_prefix}_batches": seen,
            f"{metric_key_prefix}_runtime": runtime,
        }
        logger.info(
            f"eval_on_main_process_only: rank 0 eval done in {runtime:.1f}s "
            f"(loss {mean_loss:.4f})"
        )
        return metrics

    def _share_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        payload = [metrics]
        torch.distributed.broadcast_object_list(
            payload, src=0, group=self._eval_share_group
        )
        torch.distributed.barrier(group=self._eval_share_group)
        return payload[0]

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if not self._is_distributed():
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        metrics: Dict[str, Any] = {}
        if self.args.process_index == 0:
            metrics = self._local_eval(eval_dataset, metric_key_prefix)
        else:
            logger.info(
                f"eval_on_main_process_only: rank {self.args.process_index} "
                "skipping eval forward; waiting for rank 0"
            )

        metrics = self._share_metrics(metrics)

        # super().evaluate() would normally do these; keep logging and callbacks
        # (wandb, best-metric tracking) behaving the same on every rank.
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics


class TehrStage1Trainer:
    """
    Manages the entire workflow for continued pretraining of LLMs on EHR data.
    
    This class encapsulates:
    - Model loading with Unsloth
    - LoRA adapter configuration
    - Dataset preparation
    - Training setup
    - WandB integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pretrainer with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML.
        """
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.training_config = config['training']
        self.lora_config = config['lora']
        self.wandb_config = config.get('wandb', {})
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    def setup_wandb(self) -> tuple[str, str]:
        default_run_name = (
            f"_r{self.lora_config['r']}"
            f"_alpha{self.lora_config['lora_alpha']}"
            f"_dropout{self.lora_config['lora_dropout']}"
            f"_lr{self.training_config['learning_rate']}"
            f"_bs{self.training_config['batch_size']}"
            f"_wd{self.training_config['weight_decay']}"
            f"_ga{self.training_config['gradient_accumulation_steps']}"
            f"_length{self.model_config['max_length']}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_name = self.wandb_config.get("run_name") or default_run_name
        return init_wandb(
            self.config,
            local_rank=self.local_rank,
            run_name=run_name,
            default_project="ehr-llm-pretraining",
        )
    
    def load_model(self):
        """
        Load model and tokenizer.
        Uses Unsloth by default; set ``model.use_unsloth: false`` for
        HuggingFace + BitsAndBytes.
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"Loading model: {self.model_config['model_name']}")
        logger.info("=" * 80)

        if torch.cuda.is_available():
            logger.info(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  - CUDA devices: {torch.cuda.device_count()}")
        else:
            logger.info(f"  - CUDA not available, using CPU")

        use_unsloth = self.model_config.get('use_unsloth', True)

        if use_unsloth:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_config['model_name'],
                max_seq_length=self.model_config['max_length'],
                dtype=None,
                load_in_4bit=self.training_config.get('load_in_4bit', True),
                device_map={"": self.local_rank},
            )
        else:
            from src.training.model_loader import load_hf_model
            self.model, self.tokenizer = load_hf_model(
                model_name=self.model_config['model_name'],
                max_seq_length=self.model_config['max_length'],
                load_in_4bit=self.training_config.get('load_in_4bit', True),
                local_rank=self.local_rank,
            )
    
    
    def apply_lora(self):
        """
        Apply LoRA adapters to the model.
        Uses Unsloth's ``get_peft_model`` when ``use_unsloth=True``, otherwise
        falls back to standard PEFT ``get_peft_model``.
        """
        use_unsloth = self.model_config.get('use_unsloth', True)

        if use_unsloth:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_config.get('r', 16),
                target_modules=self.lora_config.get('target_modules', [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head",
                ]),
                lora_alpha=self.lora_config.get('lora_alpha', 16),
                lora_dropout=self.lora_config.get('lora_dropout', 0.05),
                bias=self.lora_config.get('bias', "none"),
                use_gradient_checkpointing=self.training_config.get('gradient_checkpointing', 'unsloth'),
                random_state=42,
                use_rslora=self.lora_config.get('use_rslora', True),
                loftq_config=self.lora_config.get('loftq_config', None),
            )
        else:
            from src.training.model_loader import apply_lora_peft
            self.model = apply_lora_peft(self.model, self.lora_config)

        logger.info("  - Applied LoRA adapters to the model.")
    
    def prepare_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        """
        Prepare training and validation datasets using a Rank-0-only processing pattern.
        """
        # Length must be in the cache path. A leftover 32k pack under the old
        # unversioned ``train_packed`` dir is truncated by Unsloth to
        # max_length while labels stay 32k, which crashes fused CE:
        # ``Expected input batch_size (64000) to match target batch_size (128000)``.
        max_length = int(self.model_config["max_length"])
        cache_root = self.training_config.get("data_cache_dir") or self.training_config["output_dir"]
        cache_dir = os.path.join(cache_root, "data_cache")
        train_cache_path = os.path.join(cache_dir, f"train_packed_len{max_length}")
        val_cache_path = os.path.join(cache_dir, f"val_packed_len{max_length}")
        train_info = os.path.join(train_cache_path, "dataset_info.json")

        if not torch.distributed.is_initialized():
            # Fallback for some environments: ensure we know who we are
            if "WORLD_SIZE" in os.environ:
                torch.distributed.init_process_group(backend="nccl")

        # 2. Only Rank 0 performs the heavy lifting
        if self.local_rank == 0:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                if not os.path.exists(train_info):
                    logger.info("\n--- Data Preparation (Rank 0 Only) ---")
                    logger.info(f"  - Packed cache → {train_cache_path}")

                    # Get datasets that returns dict of text,label and subject id
                    train_base = get_dataset(self.data_config['dataset'], self.data_config, "train")
                    val_base = get_dataset(self.data_config['dataset'], self.data_config, "tuning")
                    max_p = int(self.training_config.get("smoke_max_patients") or 0)
                    if max_p > 0:
                        from torch.utils.data import Subset
                        n_train = min(max_p, len(train_base))
                        n_val = min(max_p, len(val_base))
                        train_base = Subset(train_base, range(n_train))
                        val_base = Subset(val_base, range(n_val))
                        logger.info(
                            f"  - smoke_max_patients: train={n_train} val={n_val}"
                        )

                    # Process Training Data
                    extract_and_pack(train_base, self.tokenizer, max_length, train_cache_path)

                    # Process Validation Data
                    extract_and_pack(val_base, self.tokenizer, max_length, val_cache_path)
                    
                else:
                    logger.info("\n--- Found existing cached dataset, skipping processing ---")
                    logger.info(f"  - Packed cache → {train_cache_path}")
            
            except Exception as e:
                logger.error(f"Rank {self.local_rank} error: {e}")
                raise

        
        max_idle = 600  # max seconds Rank 0 can be idle before giving up
        heartbeat_file = os.path.join(train_cache_path, "heartbeat_rank0.txt")
        dataset_file = train_info

        while not os.path.exists(dataset_file):
            if os.path.exists(heartbeat_file):
                # read last heartbeat
                last_time = float(open(heartbeat_file).read())
                idle_time = time.time() - last_time
                if idle_time > max_idle:
                    raise TimeoutError(f"Rank {self.local_rank} timed out: Rank 0 idle for {idle_time:.1f}s")
            else:
                # no heartbeat yet, could be just starting
                idle_time = 0

            logger.info(f"Rank {self.local_rank} waiting... last heartbeat {idle_time:.1f}s ago")
            time.sleep(5)

        # 4. All ranks load from the disk cache
        # .load_from_disk() uses memory mapping, so it's nearly instant and RAM-efficient
        train_dataset = Dataset.load_from_disk(train_cache_path)
        
        val_dataset = None
        if os.path.exists(val_cache_path):
            val_dataset = Dataset.load_from_disk(val_cache_path)

        if len(train_dataset) > 0:
            observed = len(train_dataset[0]["input_ids"])
            if observed != max_length:
                raise ValueError(
                    f"Packed cache sequence length {observed} != model max_length "
                    f"{max_length} at {train_cache_path}. Delete that directory and rerun."
                )

        return train_dataset, val_dataset
    
    def create_trainer(
        self, 
        train_dataset: Dataset, 
        val_dataset: Optional[Dataset],
        run_name: str,
        report_to: str
    ):
        """
        Create the SFTTrainer using the manually packed datasets.
        """
        logger.info("\n" + "=" * 80)
        logger.info("Setting up training...")
        logger.info("=" * 80)
        
        # Align AMP flags with the loaded model dtype.
        # Unsloth patches SFTTrainer globally and raises if the model is float16
        # while bf16=True (common for HF + BitsAndBytes loads).
        use_fp16 = bool(self.training_config.get("fp16", False))
        use_bf16 = bool(self.training_config.get("bf16", True))
        try:
            param_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            param_dtype = None
        if param_dtype == torch.float16 and use_bf16 and not use_fp16:
            logger.warning(
                "Model parameters are float16 but training.bf16=True; "
                "switching to fp16=True, bf16=False for Unsloth/SFTTrainer compatibility."
            )
            use_fp16, use_bf16 = True, False
        elif param_dtype == torch.bfloat16 and use_fp16 and not use_bf16:
            logger.warning(
                "Model parameters are bfloat16 but training.fp16=True; "
                "switching to bf16=True, fp16=False."
            )
            use_fp16, use_bf16 = False, True

        eval_steps = int(self.training_config.get("eval_steps", 500) or 0)
        save_steps = int(self.training_config.get("save_steps", 1000) or 0)
        eval_on_main = bool(
            self.training_config.get("eval_on_main_process_only", False)
        )
        # Unsloth's inner loop desyncs DDP at eval_steps (NCCL timeout). Skip
        # mid-train eval when requested; keep save_steps checkpoints.
        do_eval = bool(val_dataset) and not eval_on_main
        load_best = do_eval
        if load_best and eval_steps and save_steps % eval_steps != 0:
            load_best = False
            msg = (
                "load_best_model_at_end disabled: save_steps="
                f"{save_steps} is not a multiple of eval_steps={eval_steps}"
            )
            logger.warning(msg)
            print(f"  {msg}", flush=True)
        if eval_on_main and val_dataset is not None:
            msg = (
                "eval_on_main_process_only: eval_strategy=no to avoid "
                f"Unsloth/NCCL desync; checkpoints every {save_steps} steps"
            )
            logger.info(msg)
            print(f"  {msg}", flush=True)

        # Define the SFT configuration with all your detailed parameters
        sft_kwargs = dict(
            output_dir=self.training_config['output_dir'],
            overwrite_output_dir=self.training_config.get('overwrite_output_dir', True),
            packing=False, # Have manually packed the data ourselves
            max_seq_length=self.model_config['max_length'],
            dataset_num_proc=4,
            num_train_epochs=self.training_config['epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config.get('eval_batch_size', self.training_config['batch_size']),
            learning_rate=float(self.training_config['learning_rate']),
            weight_decay=float(self.training_config.get('weight_decay', 0.01)),
            warmup_steps=self.training_config.get('warmup_steps', 500),
            logging_steps=self.training_config.get('logging_steps', 100),
            eval_strategy="steps" if do_eval else "no",
            eval_steps=eval_steps or 500,
            save_strategy="steps",
            save_steps=save_steps or 1000,
            save_total_limit=self.training_config.get('save_total_limit', 2),
            load_best_model_at_end=load_best,
            metric_for_best_model="loss" if load_best else None,
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 1),
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', False),
            average_tokens_across_devices=False,
            report_to=report_to,
            run_name=run_name,
            ddp_find_unused_parameters=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            dataloader_num_workers=self.training_config.get('dataloader_num_workers', 8),
            dataloader_drop_last=True,
        )
        apply_hf_step_limit(sft_kwargs, self.training_config)
        training_args = SFTConfig(**sft_kwargs)

        logger.info("\nInitializing SFTTrainer...")
        print("\nInitializing SFTTrainer...", flush=True)

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "train_dataset": train_dataset, # This is already packed!
            "eval_dataset": val_dataset if do_eval else None,
            "args": training_args,
            "data_collator": collator,
        }

        callbacks: list = []
        append_max_runtime_callback(callbacks, self.training_config)
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        self.trainer = SFTTrainer(**trainer_kwargs)
        
        logger.info(f"  ✓ SFTTrainer initialized")
        print("  ✓ SFTTrainer initialized", flush=True)
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                world_size = torch.distributed.get_world_size()
            except Exception:
                world_size = 1

        per_device_bs = self.training_config['batch_size']
        grad_acc = self.training_config.get('gradient_accumulation_steps', 1)
        effective_global_bs = per_device_bs * grad_acc * world_size

    
    def train(self, resume_from_checkpoint: str | None = None):
        """Run the training loop."""
        logger.info("\n" + "=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80 + "\n")

        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            self.trainer.train()

    def save_model(self):
        """Save the final trained model."""
        logger.info("\n" + "=" * 80)
        logger.info("Saving final model...")
        logger.info("=" * 80)
        
        final_subdir = self.training_config.get('final_subdir', "final_model")
        final_model_path = os.path.join(self.training_config['output_dir'], final_subdir)
        os.makedirs(final_model_path, exist_ok=True)
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"  - Model + tokenizer saved to: {final_model_path}")
    
    def run(self, resume_from_checkpoint: str | None = None):
        self.run_full_pipeline(resume_from_checkpoint=resume_from_checkpoint)
        return None, None

    def run_full_pipeline(self, resume_from_checkpoint: str | None = None):
        """
        Run the complete pretraining pipeline.
        
        """
        # Setup WandB
        run_name, report_to = self.setup_wandb()
        
        # Load model
        self.load_model()
        
        # Apply LoRA
        self.apply_lora()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Create trainer
        self.create_trainer(train_dataset, val_dataset, run_name, report_to)
        
        # Train
        self.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save
        self.save_model()
