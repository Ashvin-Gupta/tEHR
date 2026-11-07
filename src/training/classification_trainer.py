"""
Linear-head LLM classifier (LoRA).

Used by: tehr_stage2, ablation_supervised_only, ablation_cpt_frozen,
ablation_hash, ablation_scrambled, ablation_no_time, ablation_no_value,
ablation_max_events.
"""
import os
import time
import torch
from typing import Dict, Any
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from src.training.eval_schedule import (
    EvalAtStepsCallback,
    append_max_runtime_callback,
    eval_optimizer_steps_from_fractions,
    get_eval_epoch_fractions,
)
from torch.utils.data import ConcatDataset

from src.data.classification_collator import ClassificationCollator
from src.training.training_utils import (
    apply_hf_step_limit,
    compute_and_sort_by_length,
    init_wandb,
    skip_final_eval,
)
from src.models.llm_classifier import LLMClassifier
from src.training.model_loader import (
    load_pretrained_lora_model,
    load_pretrained_lora_model_hf,
    load_hf_model,
    apply_lora_peft,
)
from src.evaluations.evaluations import (
    compute_classification_metrics,
    save_results,
)

from src.data.dataset_factory import get_dataset
import logging
logger = logging.getLogger(__name__)

class TehrClassifierTrainer:
    """
    Manages the complete workflow for EHR classification fine-tuning.
    
    This class encapsulates:
    - Loading pretrained model with LoRA
    - Creating classification wrapper
    - Dataset preparation
    - Training setup
    - Evaluation and visualization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the classification trainer.
        
        Args:
            config: Configuration dictionary loaded from YAML.
        """
        self.config = config
        self.model_config = config['model']
        self.data_config = config['data']
        self.training_config = config['training']
        self.wandb_config = config.get('wandb', {})
        
        self.base_model = None
        self.tokenizer = None
        self.classifier_model = None
        self.trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    def setup_wandb(self) -> str:
        run_name, _ = init_wandb(
            self.config,
            local_rank=self.local_rank,
            run_name=self.wandb_config.get("run_name")
            or f"classifier_{self.config.get('name', 'default')}",
            default_project="llm-classification",
        )
        return run_name
    
    def load_pretrained_model(self):
        """
        Load model for classification fine-tuning.

        Supports four modes controlled by config fields:
          1. ``pretrained_checkpoint`` set, ``use_unsloth: true``  (default):
             Load Unsloth checkpoint with LoRA — original behaviour.
          2. ``pretrained_checkpoint`` set, ``use_unsloth: false``:
             Load HF/PEFT checkpoint.
          3. ``pretrained_checkpoint: null``, ``use_unsloth: true``:
             Load fresh Unsloth base model; apply LoRA if ``train_lora: true``.
          4. ``pretrained_checkpoint: null``, ``use_unsloth: false``:
             Load fresh HF base model.
        """
        print("\n" + "=" * 80)
        print("Loading model...")
        print("=" * 80)

        pretrained_checkpoint = self.model_config.get('pretrained_checkpoint')
        use_unsloth = self.model_config.get('use_unsloth', True)
        train_lora = self.model_config.get('train_lora', False)

        if pretrained_checkpoint:
            # ── Mode 1 / 2: load from saved pretraining checkpoint ──────────
            if use_unsloth:
                self.base_model, self.tokenizer = load_pretrained_lora_model(
                    pretrained_checkpoint=pretrained_checkpoint,
                    max_seq_length=self.model_config.get('max_length', 8192),
                    load_in_4bit=self.training_config.get('load_in_4bit', True),
                    local_rank=self.local_rank,
                )
            else:
                self.base_model, self.tokenizer = load_pretrained_lora_model_hf(
                    pretrained_checkpoint=pretrained_checkpoint,
                    max_seq_length=self.model_config.get('max_length', 8192),
                    load_in_4bit=self.training_config.get('load_in_4bit', True),
                    local_rank=self.local_rank,
                )
        else:
            # ── Mode 3 / 4: no pretrained checkpoint — fresh base model ─────
            model_name = self.model_config.get('model_name')
            if not model_name:
                raise ValueError(
                    "Either 'model.pretrained_checkpoint' or 'model.model_name' "
                    "must be set in config."
                )
            print(f"  - No pretrained checkpoint; loading fresh base model: {model_name}")

            if use_unsloth:
                from unsloth import FastLanguageModel
                self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=self.model_config.get('max_length', 8192),
                    dtype=None,
                    load_in_4bit=self.training_config.get('load_in_4bit', True),
                    device_map={"": self.local_rank},
                )
                if train_lora:
                    lora_cfg = self.config.get('lora', {})
                    self.base_model = FastLanguageModel.get_peft_model(
                        self.base_model,
                        r=lora_cfg.get('r', 16),
                        target_modules=lora_cfg.get('target_modules', [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ]),
                        lora_alpha=lora_cfg.get('lora_alpha', 32),
                        lora_dropout=lora_cfg.get('lora_dropout', 0.05),
                        bias=lora_cfg.get('bias', 'none'),
                        use_gradient_checkpointing='unsloth',
                        random_state=42,
                        use_rslora=lora_cfg.get('use_rslora', True),
                    )
                    print("  - Applied fresh LoRA adapters (Unsloth)")
            else:
                self.base_model, self.tokenizer = load_hf_model(
                    model_name=model_name,
                    max_seq_length=self.model_config.get('max_length', 8192),
                    load_in_4bit=self.training_config.get('load_in_4bit', True),
                    local_rank=self.local_rank,
                )
                if train_lora:
                    lora_cfg = self.config.get('lora', {})
                    self.base_model = apply_lora_peft(self.base_model, lora_cfg)
                    print("  - Applied fresh LoRA adapters (PEFT)")
    
    def create_classifier(self):
        """Wrap the pretrained model with a classification head."""
        print("\n" + "=" * 80)
        print("Creating LLM Classifier wrapper...")
        print("=" * 80)
        
        # Determine training mode
        train_lora = self.model_config.get('train_lora', False)
        freeze_llm = self.model_config.get('freeze_llm', True)
        
        # If training LoRA, unfreeze LoRA parameters
        trainable_keywords = ["lora_"] if train_lora else None
        
        if train_lora:
            print("  - Training mode: LoRA adapters + Classification head")
        else:
            print("  - Training mode: Classification head only (LoRA frozen)")
        
        self.classifier_model = LLMClassifier(
            base_model=self.base_model,
            hidden_size=self.model_config['hidden_size'],
            num_labels=self.model_config.get('num_labels', 2),
            freeze_base=freeze_llm,
            trainable_param_keywords=trainable_keywords,
            pooling=self.model_config.get('pooling', 'last_token'),
        )

    def _sort_with_cache(self, dataset, split: str, shuffle_buckets: bool, num_buckets: int = 20):
        """
        Sort a dataset by sequence length, caching the sorted indices.

        Cache layout (shared across experiments for the same task+model)::

            {data_cache_dir}/MIMIC_{split}_sorted.pt

        ``data_cache_dir`` is set by ``build_config`` to
        ``{labels_cache_root}/{dataset}/{task}``.

        The cache is **tokenizer-specific** across model families because token
        lengths (and thus sort order) differ. Filenames no longer embed
        checkpoint basenames.
        """
        print("Sorting dataset with cache...")
        if not self.data_config.get('sort_by_length', False):
            return dataset

        dataset_name = self.data_config.get("dataset", "CPRD")
        cache_root = self.training_config.get("data_cache_dir")
        if not cache_root:
            # Fallback for ad-hoc runs without the shared labels_cache path.
            cache_root = os.path.join(
                self.training_config["output_dir"], "data_cache", "length_sorted"
            )
        os.makedirs(cache_root, exist_ok=True)

        cache_filename = f"{dataset_name}_{split}_sorted.pt"
        cache_path = os.path.join(cache_root, cache_filename)
        lock_path = cache_path + ".lock"

        # One-time migration: accept legacy
        #   MIMIC_{split}_sorted_checkpoint-*.pt  (or any *_sorted_*.pt)
        # sitting in the same directory and rename to the stable filename.
        if not os.path.exists(cache_path):
            prefix = f"{dataset_name}_{split}_sorted_"
            legacy = [
                f
                for f in os.listdir(cache_root)
                if f.startswith(prefix) and f.endswith(".pt")
            ]
            if legacy:
                legacy.sort(
                    key=lambda f: os.path.getmtime(os.path.join(cache_root, f)),
                    reverse=True,
                )
                legacy_path = os.path.join(cache_root, legacy[0])
                if self.local_rank == 0:
                    print(
                        f"  - Migrating legacy cache {legacy_path} → {cache_path}"
                    )
                    try:
                        os.rename(legacy_path, cache_path)
                    except OSError:
                        import shutil
                        shutil.copy2(legacy_path, cache_path)

        def _load_indices(path):
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(path, map_location="cpu")

        def _try_load(path):
            try:
                return _load_indices(path)
            except (EOFError, RuntimeError, OSError, ValueError):
                return None

        def _remove_stale_lock():
            # A killed writer can leave a 0-byte lock; waiters then sit silent
            # for 4h (see ehrshot scrambled readmission/pancreatic). Held-out
            # sort itself is ~7 min, so a lock older than this is abandoned.
            if not os.path.exists(lock_path):
                return False
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                return False
            if age <= 20 * 60:
                return False
            print(
                f"  - Removing stale length-sort lock ({age / 60:.1f} min old) "
                f"at {lock_path}",
                flush=True,
            )
            try:
                os.remove(lock_path)
            except OSError:
                return False
            return True

        def _load_finished_cache():
            if not os.path.exists(cache_path) or os.path.exists(lock_path):
                return None
            indices = _try_load(cache_path)
            if indices is None:
                print(f"  - Ignoring corrupt length-sort cache at {cache_path}")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
            return indices

        # Fast path: finished cache, no writer in progress
        indices = _load_finished_cache()
        if indices is not None:
            print(
                f"  - Found cached length-sorted indices for split '{split}' "
                f"at {cache_path}"
            )
            return torch.utils.data.Subset(dataset, indices)

        _remove_stale_lock()
        indices = _load_finished_cache()
        if indices is not None:
            print(
                f"  - Found cached length-sorted indices for split '{split}' "
                f"at {cache_path}"
            )
            return torch.utils.data.Subset(dataset, indices)

        # Try to acquire lock so only one process computes (multi-GPU / multi-seed)
        acquired = False
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            acquired = True
        except FileExistsError:
            pass

        if acquired:
            # We hold the lock: compute and write atomically, then remove lock
            print(
                f"  - No cached length-sorted indices for split '{split}', "
                f"computing and caching to {cache_path}...",
                flush=True,
            )
            try:
                sorted_dataset = compute_and_sort_by_length(
                    dataset,
                    self.tokenizer,
                    shuffle_buckets=shuffle_buckets,
                    num_buckets=num_buckets,
                )
                if isinstance(sorted_dataset, torch.utils.data.Subset):
                    tmp_path = cache_path + ".tmp"
                    torch.save(sorted_dataset.indices, tmp_path)
                    os.replace(tmp_path, cache_path)
                    print(
                        f"  ✓ Cached sorted indices for split '{split}' to: {cache_path}"
                    )
            finally:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                try:
                    os.remove(cache_path + ".tmp")
                except OSError:
                    pass
            return sorted_dataset

        # Another process is computing; wait until the lock is gone and the
        # cache file is complete (exists() during torch.save is a partial file).
        for waited in range(14400):
            time.sleep(1)
            if waited > 0 and waited % 60 == 0:
                print(
                    f"  - Waiting for length-sort lock ({waited}s) at {lock_path}",
                    flush=True,
                )
            if _remove_stale_lock():
                return self._sort_with_cache(
                    dataset,
                    split=split,
                    shuffle_buckets=shuffle_buckets,
                    num_buckets=num_buckets,
                )
            if os.path.exists(lock_path):
                continue
            if os.path.exists(cache_path):
                indices = _try_load(cache_path)
                if indices is None:
                    continue
                print(
                    f"  - Found cached length-sorted indices for split '{split}' "
                    f"(written by another process), loading..."
                )
                return torch.utils.data.Subset(dataset, indices)
        raise TimeoutError(
            f"Timeout waiting for length-sorted cache for split '{split}' at {cache_path}"
        )
    
    def prepare_datasets(self):
        """Prepare train, validation, and test datasets."""
        print("\n" + "=" * 80)
        print("Loading datasets...")
        print("=" * 80)

        train_dataset = get_dataset(self.data_config['dataset'], self.data_config, "train")
        val_dataset = get_dataset(self.data_config['dataset'], self.data_config, "tuning")
        test_dataset = get_dataset(self.data_config['dataset'], self.data_config, "held_out")

        # Optional: Sort by length for efficient batching, with caching
        if self.data_config.get('sort_by_length', False):
            print("\n  - Sorting datasets by sequence length (with caching)...")
            train_dataset = self._sort_with_cache(
                train_dataset, split="train", shuffle_buckets=True, num_buckets=20
            )
            val_dataset = self._sort_with_cache(
                val_dataset, split="tuning", shuffle_buckets=False, num_buckets=20
            )
            test_dataset = self._sort_with_cache(
                test_dataset, split="held_out", shuffle_buckets=False, num_buckets=20
            )
            print("  ✓ Datasets sorted by length (cached)")
        
        print(f"  - Train dataset: {len(train_dataset)} patients")
        print(f"  - Validation dataset: {len(val_dataset)} patients")
        print(f"  - Test dataset: {len(test_dataset)} patients")

        for split_name, ds in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
            self._log_class_distribution(ds, split_name)

        return train_dataset, val_dataset, test_dataset

    def _log_class_distribution(self, dataset, split_name: str):
        """Log positive/negative counts and prevalence for a dataset split."""
        try:
            labels = []
            for i in range(len(dataset)):
                item = dataset[i]
                lbl = item.get('label') if isinstance(item, dict) else None
                if lbl is None:
                    continue
                value = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
                # Everything that's not 0 is positive
                labels.append(1 if value != 0 else 0)
            if not labels:
                return
            import collections
            counts = collections.Counter(labels)
            total = len(labels)
            pos = counts.get(1, 0)
            neg = counts.get(0, 0)
            print(
                f"  - [{split_name}] class distribution: "
                f"{pos} positive ({100*pos/total:.1f}%), "
                f"{neg} negative ({100*neg/total:.1f}%), "
                f"total={total}"
            )
        except Exception as e:
            print(f"  - Could not compute class distribution for {split_name}: {e}")

    def create_collator(self):
        """Create data collator for batching."""
        print("\n" + "=" * 80)
        print("Creating data collator...")
        print("=" * 80)
        
        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_config.get('max_length'),
            binary_classification=True,
            truncation=False,
        )
        
        return collator
    
    def create_trainer(self, train_dataset, val_dataset, collator, run_name: str):
        """Create HuggingFace Trainer."""
        print("\n" + "=" * 80)
        print("Setting up trainer...")
        print("=" * 80)

        # Control whether checkpoints are written with safetensors.
        save_safetensors = bool(self.training_config.get("save_safetensors", False))

        per_device_batch = int(self.training_config.get('batch_size', 8))
        eval_batch = int(self.training_config.get('eval_batch_size', 8))
        env_eval_bs = os.environ.get("TEHR_EVAL_BATCH_SIZE", "").strip()
        if env_eval_bs:
            eval_batch = int(env_eval_bs)
            print(f"  - TEHR_EVAL_BATCH_SIZE={eval_batch} (overrides config eval_batch_size)")
        grad_accum = int(self.training_config.get('gradient_accumulation_steps', 1))
        raw_fractions = self.training_config.get("eval_epoch_fractions", "default")
        use_fraction_schedule = raw_fractions not in (None, False, "steps", "legacy")
        fractions = get_eval_epoch_fractions(self.training_config) if use_fraction_schedule else []

        # Matched paper schedule: eval only at epoch fractions (not fixed step gaps).
        # Disable the built-in interval by using a huge eval/save interval; the
        # EvalAtStepsCallback fires should_evaluate / should_save at fraction steps.
        if use_fraction_schedule:
            eval_at_steps = eval_optimizer_steps_from_fractions(
                len(train_dataset),
                fractions,
                per_device_batch=per_device_batch,
                grad_accum=grad_accum,
            )
            eval_steps = 10**9
            save_steps = 10**9
            print(f"  - Train size: {len(train_dataset)}")
            print(f"  - Eval epoch fractions: {fractions}")
            print(f"  - Eval optimizer steps: {eval_at_steps}")
        else:
            eval_at_steps = None
            eval_steps = int(self.training_config.get('eval_steps', 100))
            save_steps = int(self.training_config.get('save_steps', 500))

        training_args_kwargs = dict(
            output_dir=self.training_config['output_dir'],
            overwrite_output_dir=bool(self.training_config.get("overwrite_output_dir", False)),
            run_name=run_name,
            report_to="wandb" if self.wandb_config.get('enabled', False) else "none",
            
            # Training hyperparameters
            num_train_epochs=int(self.training_config.get('epochs', 10)),
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=eval_batch,
            learning_rate=float(self.training_config.get('learning_rate', 1e-3)),
            weight_decay=float(self.training_config.get('weight_decay', 0.01)),
            warmup_steps=int(self.training_config.get('warmup_steps', 100)),
            
            # Gradient settings
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,
            
            # Multi-GPU settings
            ddp_find_unused_parameters=False,
            dataloader_num_workers=self.training_config.get('dataloader_num_workers', 8),
            
            # Precision
            fp16=bool(self.training_config.get('fp16', False)),
            bf16=bool(self.training_config.get('bf16', True)),
            
            # Logging and evaluation
            logging_steps=int(self.training_config.get('logging_steps', 10)),
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=int(self.training_config.get('save_total_limit', 2)),
            
            # Best model tracking (early-stop selection among fraction checkpoints)
            load_best_model_at_end=True,
            metric_for_best_model="auroc",
            greater_is_better=True,
            
            # Other
            remove_unused_columns=False,
        )
        training_args_kwargs["save_safetensors"] = save_safetensors
        apply_hf_step_limit(training_args_kwargs, self.training_config)

        training_args = TrainingArguments(**training_args_kwargs)
        
        # Create callbacks
        callbacks = []
        if eval_at_steps is not None:
            callbacks.append(EvalAtStepsCallback(eval_at_steps, fractions=fractions))
        early_stopping_patience = self.training_config.get('early_stopping_patience')
        if early_stopping_patience:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=self.training_config.get('early_stopping_threshold', 0.0)
            ))
            print(f"  - Early stopping enabled with patience={early_stopping_patience}")
        append_max_runtime_callback(callbacks, self.training_config)
        
        
        self.trainer = Trainer(
            model=self.classifier_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_classification_metrics,
            callbacks=callbacks,
        )
    
    def train(self, resume_from_checkpoint: str | None = None):
        """Run the training loop."""
        print("\n" + "=" * 80)
        print("Starting classification training...")
        print("=" * 80)

        if resume_from_checkpoint:
            if not os.path.exists(resume_from_checkpoint):
                raise ValueError(f"Checkpoint not found: {resume_from_checkpoint}")
            print(f"  - Resuming from checkpoint: {resume_from_checkpoint}")

            safetensors_path = os.path.join(resume_from_checkpoint, "model.safetensors")
            bin_path = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
            if os.path.exists(safetensors_path) and not os.path.exists(bin_path):
                print("  - Checkpoint is in safetensors format; enabling safetensors loading.")
                self.trainer.args.save_safetensors = True

            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            self.trainer.train()

    def save_model(self):
        """Save the final trained model."""
        print("\n" + "=" * 80)
        print("Saving final model...")
        print("=" * 80)

        if not self.training_config.get('save_model', True):
            print("  - Skipping model save because 'save_model' is set to False in training config.")
            return
        
        final_model_path = os.path.join(self.training_config['output_dir'], "final_model")
        self.trainer.save_model(final_model_path)
        print(f"  - Model saved to: {final_model_path}")
    
    def _score_and_save_split(self, dataset, split_name: str, metric_key_prefix: str) -> Dict[str, Any]:
        """One forward pass: metrics + labels/probs JSON. No ROC/PR plots."""
        print("\n" + "=" * 80)
        print(f"Running evaluation on {split_name} set...")
        print("=" * 80)
        pred_output = self.trainer.predict(dataset, metric_key_prefix=metric_key_prefix)
        metrics = pred_output.metrics or {}
        print(f"\nFinal {split_name} Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        self._save_split_predictions(
            dataset,
            pred_output,
            split_name=split_name,
            metrics=metrics,
        )
        return metrics

    def evaluate_and_visualize(self, val_dataset, test_dataset):
        """Score val/test, write metrics and prediction JSON. No plots."""
        combine_eval_splits = self.data_config.get("combine_eval_splits", False)

        if combine_eval_splits:
            combined_dataset = ConcatDataset([val_dataset, test_dataset])
            print(
                f"  - Combined eval dataset size: {len(combined_dataset)} "
                f"(val={len(val_dataset)} + test={len(test_dataset)})"
            )
            combined_results = self._score_and_save_split(
                combined_dataset, "combined", "combined"
            )
            return combined_results, combined_results

        eval_results = self._score_and_save_split(val_dataset, "validation", "eval")
        test_results = self._score_and_save_split(test_dataset, "test", "test")
        return eval_results, test_results
    
    def restore_weights_from_checkpoint(
        self,
        checkpoint_path: str,
        require_lora: bool = True,
    ):
        """
        Warm-start LoRA adapter weights AND the classifier head from a
        previously saved classifier checkpoint, WITHOUT resuming the training
        step counter, optimizer, or scheduler.

        Use this when the task is the same (binary prediction) but the dataset
        has changed: you want the same trained weights as a starting point,
        but a clean training run from step 0.

        The quantised base-model weights are intentionally skipped — they are
        already loaded correctly from `pretrained_checkpoint` and bitsandbytes
        NF4 state cannot be trivially restored from a safetensors file.

        Args:
            checkpoint_path: Directory of a Trainer checkpoint containing
                             `model.safetensors` (full LLMClassifier state).
            require_lora: Fail if the checkpoint has no LoRA tensors. Set false
                          for head-only runs (``train_lora: false``, e.g.
                          ablation_frozen_pretrained), where the checkpoint
                          legitimately contains only ``classifier.*`` weights.
        """
        print("\n" + "=" * 80)
        print("Warm-starting weights from checkpoint (LoRA adapters + classifier head)...")
        print("=" * 80)
        print(f"  - Stage 1 already loaded: {self.model_config.get('pretrained_checkpoint')}")
        print(f"  - Overlaying stage 2 from: {checkpoint_path}")

        weights_file = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"No model weights found in checkpoint directory: {checkpoint_path}"
            )

        try:
            from safetensors.torch import load_file as _st_load
            ckpt_state = _st_load(weights_file)
        except Exception:
            ckpt_state = torch.load(weights_file, map_location="cpu", weights_only=True)

        # ── Diagnostic: show what's actually in the checkpoint ───────────────
        all_ckpt_keys = sorted(ckpt_state.keys())
        print(f"  - Checkpoint contains {len(all_ckpt_keys)} total keys.")
        print("  - First 10 checkpoint keys (to verify naming convention):")
        for k in all_ckpt_keys[:10]:
            print(f"      {k}")

        model_state = self.classifier_model.state_dict()
        all_model_keys = sorted(model_state.keys())
        print(f"  - Current model has {len(all_model_keys)} total parameter keys.")
        print("  - First 10 model keys (to compare naming with checkpoint):")
        for k in all_model_keys[:10]:
            print(f"      {k}")
        # ─────────────────────────────────────────────────────────────────────

        # Load LoRA adapter matrices and the classifier head.
        # Skip quantised base-model weights (bitsandbytes NF4 state tensors) —
        # they are already in memory from the original pretrained_checkpoint.
        lora_keywords = (
            "lora_A", "lora_B", "lora_magnitude_vector",
            "lora_embedding_A", "lora_embedding_B",
        )
        transfer_state = {
            k: v for k, v in ckpt_state.items()
            if any(kw in k for kw in lora_keywords)
            or k.startswith("classifier.")
            or k.startswith("input_proj.")
            or k.startswith("input_norm.")
        }

        lora_keys = [k for k in transfer_state if any(kw in k for kw in lora_keywords)]
        head_keys = [k for k in transfer_state if k.startswith("classifier.")]
        proj_keys = [
            k for k in transfer_state
            if k.startswith("input_proj.") or k.startswith("input_norm.")
        ]

        if not lora_keys and require_lora:
            raise RuntimeError(
                f"No LoRA adapter keys found in {weights_file}. "
                f"Make sure the checkpoint was saved from an LLMClassifier with "
                f"LoRA adapters (keys containing lora_A / lora_B)."
            )
        if not lora_keys:
            print("  - No LoRA keys in checkpoint (expected for a head-only run).")
        if not head_keys:
            print(
                "  - No 'classifier.*' keys found in checkpoint; "
                "classifier head will be randomly initialised."
            )

        # Snapshot live (stage-1 CPT) LoRA before overlay. Clone so in-place
        # writes cannot alias the "before" tensors.
        live_params = dict(self.classifier_model.named_parameters())
        before_lora = {
            k: live_params[k].detach().float().cpu().clone()
            for k in lora_keys
            if k in live_params
        }

        copied, skipped = 0, []
        for key, value in transfer_state.items():
            param = live_params.get(key)
            if param is None:
                skipped.append(key)
                continue
            src = value.detach().to(device=param.device, dtype=param.dtype)
            if src.shape != param.shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}: checkpoint {tuple(src.shape)} "
                    f"vs model {tuple(param.shape)}"
                )
            param.data.copy_(src)
            copied += 1

        print(
            f"  - Copied {copied} tensors in-place "
            f"({len(lora_keys)} LoRA, {len(head_keys)} classifier, "
            f"{len(proj_keys)} projection)."
        )
        if skipped:
            print(
                f"  - *** WARNING: {len(skipped)} keys from the checkpoint could NOT "
                f"be matched to named_parameters and were DROPPED. ***"
            )
            for k in skipped[:10]:
                print(f"      {k}")
        else:
            print("  - All transfer keys matched the current model successfully.")

        # Prove overlay: live LoRA must match stage-2 checkpoint, and (when
        # stage 1 adapters are on disk) must differ from CPT LoRA.
        after_state = dict(self.classifier_model.named_parameters())
        lora_vs_ckpt = []
        lora_vs_s1 = []
        for key in lora_keys:
            if key not in after_state:
                continue
            live = after_state[key].detach().float().cpu()
            ckpt = transfer_state[key].detach().float().cpu()
            lora_vs_ckpt.append(float((live - ckpt).abs().max()))
            if key in before_lora:
                lora_vs_s1.append(float((live - before_lora[key]).abs().max()))
        if lora_vs_ckpt:
            ckpt_max = max(lora_vs_ckpt)
            print(f"  - LoRA vs stage-2 checkpoint: max |delta| = {ckpt_max:.6g}")
            if ckpt_max > 1e-5:
                raise RuntimeError(
                    "Stage-2 LoRA overlay failed: live adapters do not match "
                    f"{checkpoint_path} (max |delta|={ckpt_max:.6g})."
                )
        if lora_vs_s1:
            s1_max = max(lora_vs_s1)
            print(f"  - LoRA vs stage-1 (pre-overlay): max |delta| = {s1_max:.6g}")
            if s1_max < 1e-8:
                print(
                    "  - *** WARNING: live LoRA is identical to stage-1 CPT. "
                    "Stage-2 adapters did not change the weights (or stage 2 "
                    "never moved them). ***"
                )
            else:
                print("  - Stage-2 adapters overlaid on stage-1 CPT successfully.")

        cpt_dir = self.model_config.get("pretrained_checkpoint")
        cpt_file = os.path.join(cpt_dir or "", "adapter_model.safetensors") if cpt_dir else ""
        if lora_keys and cpt_file and os.path.exists(cpt_file):
            try:
                from safetensors.torch import load_file as _st_cpt
                cpt_state = _st_cpt(cpt_file, device="cpu")

                def _lora_tail(name: str) -> str:
                    name = name.replace(".lora_A.default.weight", ".lora_A.weight")
                    name = name.replace(".lora_B.default.weight", ".lora_B.weight")
                    if "layers." in name:
                        return name[name.index("layers."):]
                    return name

                cpt_by_tail = {_lora_tail(k): v for k, v in cpt_state.items() if "lora_" in k}
                file_deltas = []
                for key in lora_keys:
                    tail = _lora_tail(key)
                    if tail not in cpt_by_tail or key not in after_state:
                        continue
                    live = after_state[key].detach().float().cpu()
                    cpt = cpt_by_tail[tail].detach().float().cpu()
                    if live.shape == cpt.shape:
                        file_deltas.append(float((live - cpt).abs().max()))
                if file_deltas:
                    print(
                        f"  - LoRA vs stage-1 adapter file: max |delta| = {max(file_deltas):.6g} "
                        f"({len(file_deltas)} tensors)"
                    )
            except Exception as exc:
                print(f"  - Could not compare against stage-1 adapter file: {exc}")

        logger.info("  - Training will start from step 0 with the above weights.")

    @staticmethod
    def _logits_to_pos_probs(logits_np) -> "np.ndarray":
        """
        Convert raw model logits to positive-class probability array (1-D).

        Handles two output shapes:
          - (N, 1): logistic head → sigmoid, squeeze to (N,)
          - (N, C): linear/mlp head → softmax, take column 1 for binary
        """
        import numpy as np
        t = torch.tensor(logits_np, dtype=torch.float32)
        if t.ndim == 2 and t.shape[1] == 1:
            return torch.sigmoid(t).squeeze(1).numpy()
        probs = torch.softmax(t, dim=-1).numpy()
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs

    def _pred_to_labels_and_probs(self, pred_output):
        """
        Convert a PredictionOutput into 1-D numpy arrays suitable for sklearn
        metrics and the saved ``*_results.json``.

        - label_ids may be 1-D class indices or 2-D one-hot; we always return
          1-D class indices.
        - predictions are raw logits; shape (N,1) for logistic head, (N,C) otherwise.
        """
        import numpy as np

        labels = pred_output.label_ids
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        # One-hot → class indices
        if np.ndim(labels) == 2:
            labels = np.argmax(labels, axis=1)

        probs = self._logits_to_pos_probs(logits)
        return labels, probs

    def _extract_patient_ids(self, dataset):
        """
        Extract patient IDs in prediction order when available.

        Prediction datasets index into ACES `examples` (one row per label), not
        `patient_records` (one trajectory per subject). Length-sorted Subsets
        therefore must resolve subject_id from examples[i].
        """
        def _ids_from_base(base_dataset, indices):
            examples = getattr(base_dataset, "examples", None) or None
            if examples:
                return [
                    str(examples[i].get("subject_id", i))
                    for i in indices
                ]
            if hasattr(base_dataset, "patient_records"):
                return [
                    str(base_dataset.patient_records[i].get("subject_id", i))
                    for i in indices
                ]
            return [str(i) for i in indices]

        if isinstance(dataset, torch.utils.data.Subset):
            return _ids_from_base(dataset.dataset, list(dataset.indices))

        if hasattr(dataset, "examples") and dataset.examples:
            return _ids_from_base(dataset, range(len(dataset)))

        if hasattr(dataset, "patient_records"):
            return _ids_from_base(dataset, range(len(dataset)))

        # Fallback for datasets without explicit subject IDs (e.g. ConcatDataset)
        return [str(i) for i in range(len(dataset))]

    def _extract_example_indices(self, dataset):
        """Return base-dataset example indices in prediction order (for Subsets)."""
        if isinstance(dataset, torch.utils.data.Subset):
            return [int(i) for i in dataset.indices]
        return list(range(len(dataset)))

    def _save_split_predictions(
        self,
        dataset,
        pred_output,
        split_name: str,
        metrics: Dict[str, Any] | None = None,
    ):
        """
        Persist per-sample labels and probabilities for a split.

        Rank 0 only. Under torchrun every rank holds the same gathered
        predictions and used to write this file concurrently; when the ranks'
        payloads differed in length by even one byte the shorter writer left a
        trailing fragment behind the longer one, producing JSON that fails to
        parse with "Extra data".
        """
        if self.local_rank != 0:
            return

        labels, probs = self._pred_to_labels_and_probs(pred_output)
        patient_ids = self._extract_patient_ids(dataset)
        example_indices = self._extract_example_indices(dataset)

        results_payload = {
            "metrics": metrics or {},
            "patient_ids": [str(pid) for pid in patient_ids],
            "example_indices": example_indices,
            "labels": labels.tolist() if hasattr(labels, "tolist") else list(labels),
            "probs": probs.tolist() if hasattr(probs, "tolist") else list(probs),
        }
        save_results(
            results_payload,
            os.path.join(self.training_config["output_dir"], "results"),
            f"{split_name}_results.json",
        )

    def run(self, resume_from_checkpoint: str | None = None):
        return self.run_full_pipeline(resume_from_checkpoint=resume_from_checkpoint)

    def run_full_pipeline(self, resume_from_checkpoint: str | None = None):
        """
        Run the complete classification training pipeline.
        
        Returns:
            Tuple of (eval_results, test_results)
        """
        # Setup
        run_name = self.setup_wandb()
        
        # Load pretrained model
        self.load_pretrained_model()
        
        # Create classifier
        self.create_classifier()

        # Optionally warm-start LoRA adapters + classifier head from a previous
        # checkpoint (same task, new dataset — training restarts from step 0).
        init_ckpt = self.model_config.get("init_from_checkpoint")
        if init_ckpt:
            self.restore_weights_from_checkpoint(init_ckpt)

        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()
        
        # Create collator
        collator = self.create_collator()
        
        # Create trainer
        self.create_trainer(train_dataset, val_dataset, collator, run_name)
        
        # Train
        self.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save
        self.save_model()

        if skip_final_eval(self.training_config):
            print("skip_final_eval: leaving after the training loop (smoke)")
            return None, None
        
        # Evaluate
        eval_results, test_results = self.evaluate_and_visualize(val_dataset, test_dataset)
        
        return eval_results, test_results

    def run_eval_only(self, checkpoint_path: str, run_name: str = "eval-only"):
        """
        Score an existing checkpoint on the held-out test set only, with no training.

        Recovers the metrics for a run whose training was killed after its last
        checkpoint but before the final evaluation — the usual outcome when a
        job hits the wall clock. Writes ``results/test_results.json`` (metrics
        plus labels/probs). No ROC/PR plots.

        The train and tuning splits are never loaded: they cost a full pass over
        hundreds of thousands of examples and are not needed for a test-only score.

        Returns:
            ``(None, test_results)`` — the None stands in for the unused val slot
            so callers that unpack ``eval, test = ...`` keep working.
        """
        print("\n" + "=" * 80)
        print("Evaluation-only run (test set, no training)")
        print(f"  - Stage 1 (CPT / pretrained_checkpoint): {self.model_config.get('pretrained_checkpoint')}")
        print(f"  - Stage 2 (LoRA + classifier head):     {checkpoint_path}")
        print("=" * 80)

        # Guard against any code path that would clear the run directory we are
        # reading the checkpoint from.
        self.training_config["overwrite_output_dir"] = False
        # Original cluster eval used 1 example per GPU. Batch 8 on Unsloth 2026
        # without FA2 can change last-token pooling. Keep 1 unless overridden.
        if not os.environ.get("TEHR_EVAL_BATCH_SIZE", "").strip():
            self.training_config["eval_batch_size"] = 1
            print("  - eval_only: eval_batch_size=1 (matches original cluster)")

        self.load_pretrained_model()
        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"
        self.create_classifier()
        self.restore_weights_from_checkpoint(
            checkpoint_path,
            require_lora=bool(self.model_config.get("train_lora", False)),
        )
        # Do not call FastLanguageModel.for_inference here. Original cluster
        # eval used the Trainer eval path without it; the extra graph rewrite
        # can change last-token states on Unsloth 2026.

        split_env = os.environ.get("EVAL_SPLIT", "").strip().lower()
        score_val = split_env in {"tuning", "validation", "val", "both"}
        score_test = split_env not in {"tuning", "validation", "val"}

        val_dataset = (
            self._prediction_dataset("tuning", "validation") if score_val else None
        )
        test_dataset = (
            self._prediction_dataset("held_out", "test") if score_test else None
        )
        # create_trainer reads len(train_dataset) only to place eval-fraction
        # checkpoints, which never fire here.
        stand_in = val_dataset if val_dataset is not None else test_dataset
        collator = self.create_collator()
        self.create_trainer(stand_in, stand_in, collator, run_name)

        val_results = None
        test_results = None
        if val_dataset is not None:
            val_results = self._score_and_save_split(val_dataset, "validation", "eval")
        if test_dataset is not None:
            test_results = self._score_and_save_split(test_dataset, "test", "test")

        if self.local_rank == 0:
            if score_val and not score_test:
                self._write_validation_provenance(
                    checkpoint_path, n_val=len(val_dataset)
                )
            elif score_val and score_test:
                self._write_validation_provenance(
                    checkpoint_path,
                    n_val=len(val_dataset),
                    n_test=len(test_dataset),
                )
            else:
                self._write_eval_provenance(checkpoint_path, n_test=len(test_dataset))

        return val_results, test_results

    def _prediction_dataset(self, split_name: str, label: str):
        """Load one scored split. Subclasses wrap this when the inputs are not text."""
        print("\n" + "=" * 80)
        print(f"Loading {label} dataset ({split_name})...")
        print("=" * 80)
        dataset = get_dataset(self.data_config["dataset"], self.data_config, split_name)
        if self.data_config.get("sort_by_length", False):
            dataset = self._sort_with_cache(
                dataset, split=split_name, shuffle_buckets=False, num_buckets=20
            )
        print(f"  - {label} dataset: {len(dataset)} examples")
        max_n = self.training_config.get("eval_max_samples")
        if max_n:
            n = min(int(max_n), len(dataset))
            if n < len(dataset):
                dataset = torch.utils.data.Subset(dataset, range(n))
                print(f"  - eval_max_samples={n}")
        self._log_class_distribution(dataset, label)
        return dataset

    def _write_validation_provenance(
        self, checkpoint_path: str, n_val: int, n_test: int | None = None
    ):
        """Record a tuning score without replacing ``eval_provenance.json``."""
        import json

        step = None
        base = os.path.basename(os.path.normpath(checkpoint_path))
        if base.startswith("checkpoint-"):
            try:
                step = int(base.split("-")[-1])
            except ValueError:
                pass
        results_dir = os.path.join(self.training_config["output_dir"], "results")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "validation_provenance.json")
        if n_test is None:
            note = (
                "Tuning split only. test_results.json in this directory was not "
                "rewritten."
            )
        else:
            note = (
                "Tuning and held-out scored from the checkpoint above. "
                "Published classifier test_results.json files were not rewritten."
            )
        payload = {
            "evaluated_checkpoint": checkpoint_path,
            "checkpoint_step": step,
            "produced_by": "src.pipelines.run --eval_only",
            "eval_split": os.environ.get("EVAL_SPLIT", ""),
            "note": note,
            "n_validation": n_val,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if n_test is not None:
            payload["n_test"] = n_test
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  - Validation provenance written to {path}")

    def _write_eval_provenance(self, checkpoint_path: str, n_test: int, n_val: int | None = None):
        """Record which checkpoint produced the results files in this directory."""
        import json

        step = None
        base = os.path.basename(os.path.normpath(checkpoint_path))
        if base.startswith("checkpoint-"):
            try:
                step = int(base.split("-")[-1])
            except ValueError:
                pass

        global_step = None
        trainer_state = os.path.join(checkpoint_path, "trainer_state.json")
        if os.path.exists(trainer_state):
            try:
                with open(trainer_state) as f:
                    global_step = json.load(f).get("global_step")
            except (OSError, ValueError):
                pass

        payload = {
            "evaluated_checkpoint": checkpoint_path,
            "checkpoint_step": step,
            "trainer_global_step": global_step,
            "produced_by": "src.pipelines.run --eval_only",
            "note": (
                "Metrics in this directory come from the checkpoint above, not "
                "from a training run that reached its final step. Test set only."
            ),
            "n_test": n_test,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if n_val is not None:
            payload["n_validation"] = n_val
        results_dir = os.path.join(self.training_config["output_dir"], "results")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "eval_provenance.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  - Provenance written to {path}")
