"""
EHR Classification Trainer - orchestrates classification fine-tuning workflow.
"""
import os
import time
import torch
from typing import Dict, Any
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from src.training.eval_schedule import (
    EvalAtStepsCallback,
    eval_optimizer_steps_from_fractions,
    get_eval_epoch_fractions,
)
import wandb
from torch.utils.data import ConcatDataset

from src.data.classification_collator import ClassificationCollator
from src.training.training_utils import compute_and_sort_by_length
from src.models.llm_classifier import LLMClassifier
from src.training.model_loader import (
    load_pretrained_lora_model,
    load_pretrained_lora_model_hf,
    load_hf_model,
    apply_lora_peft,
)
from src.training.metrics import compute_classification_metrics
from src.evaluations.evaluations import plot_all_curves, save_results

from src.data.dataset_factory import get_dataset
import logging
logger = logging.getLogger(__name__)

class EHRClassificationTrainer:
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
        """
        Setup WandB logging.
        
        Returns:
            Run name for the experiment.
        """
        if self.wandb_config.get('enabled', False):
            run_name = self.wandb_config.get("run_name")
            if run_name is None:
                run_name = f"classifier_{self.config.get('name', 'default')}"
            
            if self.local_rank == 0:
                wandb.init(
                    project=self.wandb_config.get("project", "llm-classification"),
                    name=run_name,
                    config=self.config
                )
                print(f"\nWandB enabled - Project: {self.wandb_config['project']}, Run: {run_name}")
            
            return run_name
        return "classification-run"
    
    def load_pretrained_model(self):
        """
        Load model for classification fine-tuning.

        Supports four modes controlled by config fields:
          1. ``pretrained_checkpoint`` set, ``use_unsloth: true``  (default):
             Load Unsloth checkpoint with LoRA — original behaviour.
          2. ``pretrained_checkpoint`` set, ``use_unsloth: false``:
             Load HF/PEFT checkpoint (e.g. saved MedGemma pretrain).
          3. ``pretrained_checkpoint: null``, ``use_unsloth: true``:
             Load fresh Unsloth base model; apply LoRA if ``train_lora: true``.
             Used for ablation 5a (no pretrain + LoRA) and 5c (no pretrain, frozen).
          4. ``pretrained_checkpoint: null``, ``use_unsloth: false``:
             Load fresh HF base model (e.g. MedGemma ablations).
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
        multi_label = self.training_config.get('multi_label', False)
        
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
            multi_label=multi_label,
            tokenizer=self.tokenizer,
            head_type=self.model_config.get('head_type', 'linear'),
            head_hidden_size=self.model_config.get('head_hidden_size', 512),
            head_dropout=self.model_config.get('head_dropout', 0.1),
            pooling=self.model_config.get('pooling', 'last_token'),
        )

    def _sort_with_cache(self, dataset, split: str, shuffle_buckets: bool, num_buckets: int = 20):
        """
        Sort a dataset by sequence length, caching the sorted indices.

        Cache layout (shared across experiments for the same task+model)::

            {data_cache_dir}/MIMIC_{split}_sorted.pt

        ``data_cache_dir`` is normally set by ``build_mimic_config`` to
        ``/home/ag619/experiments/labels_cache/{task}/{model}``.

        The cache is **tokenizer-specific**: Mistral and MedGemma must not share
        a directory, because token lengths (and thus sort order) differ.
        Filenames no longer embed checkpoint basenames like ``checkpoint-14000``.
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

        # Fast path: cache already exists
        if os.path.exists(cache_path):
            print(
                f"  - Found cached length-sorted indices for split '{split}' "
                f"at {cache_path}"
            )
            indices = _load_indices(cache_path)
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
            # We hold the lock: compute and write, then remove lock
            print(
                f"  - No cached length-sorted indices for split '{split}', "
                f"computing and caching to {cache_path}..."
            )
            sorted_dataset = compute_and_sort_by_length(
                dataset,
                self.tokenizer,
                shuffle_buckets=shuffle_buckets,
                num_buckets=num_buckets,
            )
            if isinstance(sorted_dataset, torch.utils.data.Subset):
                torch.save(sorted_dataset.indices, cache_path)
                print(f"  ✓ Cached sorted indices for split '{split}' to: {cache_path}")
            try:
                os.remove(lock_path)
            except OSError:
                pass
            return sorted_dataset

        # Another process is computing; wait for cache to appear (with timeout)
        for _ in range(14400):
            time.sleep(1)
            if os.path.exists(cache_path):
                print(
                    f"  - Found cached length-sorted indices for split '{split}' "
                    f"(written by another process), loading..."
                )
                indices = _load_indices(cache_path)
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
        
        multi_label = self.training_config.get('multi_label', False)
        binary_classification = not multi_label
        
        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_config.get('max_length'),
            binary_classification=binary_classification,
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
            per_device_eval_batch_size=int(self.training_config.get('eval_batch_size', 8)),
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
        """Run the training loop and report FLOPs on completion."""
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

        # ── FLOPs reporting ─────────────────────────────────────────────────
        try:
            from src.training.flops_counter import report_flops_from_trainer
            report_flops_from_trainer(
                trainer=self.trainer,
                model=self.classifier_model,
                seq_length=int(self.model_config.get("max_length", 8192)),
                output_dir=self.training_config.get("output_dir"),
                local_rank=self.local_rank,
            )
        except Exception as _flops_err:
            logger.warning(f"FLOPs reporting failed (non-fatal): {_flops_err}")
    
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
    
    def evaluate_and_visualize(self, val_dataset, test_dataset):
        """Run evaluation and create visualization plots."""
        combine_eval_splits = self.data_config.get("combine_eval_splits", False)

        if combine_eval_splits:
            print("\n" + "=" * 80)
            print("Running final evaluation on combined validation + test set...")
            print("=" * 80)

            combined_dataset = ConcatDataset([val_dataset, test_dataset])
            print(
                f"  - Combined eval dataset size: {len(combined_dataset)} "
                f"(val={len(val_dataset)} + test={len(test_dataset)})"
            )

            combined_results = self.trainer.evaluate(combined_dataset, metric_key_prefix="combined")
            print("\nFinal Combined Results:")
            for key, value in combined_results.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

            print("\nGenerating ROC and PR curves for combined validation + test set...")
            combined_pred_output = self.trainer.predict(combined_dataset)
            c_labels, c_probs = self._pred_to_labels_and_probs(combined_pred_output)
            plot_all_curves(
                c_labels,
                c_probs,
                os.path.join(self.training_config['output_dir'], 'plots_combined')
            )
            self._save_split_predictions(
                combined_dataset,
                combined_pred_output,
                split_name="combined",
                metrics=combined_results,
            )
            return combined_results, combined_results

        # Evaluate on validation set
        print("\n" + "=" * 80)
        print("Running final evaluation on validation set...")
        print("=" * 80)
        
        eval_results = self.trainer.evaluate()
        print("\nFinal Validation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Generate validation plots
        print("\nGenerating ROC and PR curves for validation set...")
        pred_output = self.trainer.predict(val_dataset)
        v_labels, v_probs = self._pred_to_labels_and_probs(pred_output)
        plot_all_curves(v_labels, v_probs, os.path.join(self.training_config['output_dir'], 'plots_validation'))
        self._save_split_predictions(
            val_dataset,
            pred_output,
            split_name="validation",
            metrics=eval_results,
        )
        
        # Evaluate on test set
        print("\n" + "=" * 80)
        print("Running evaluation on test set...")
        print("=" * 80)
        
        test_results = self.trainer.evaluate(test_dataset, metric_key_prefix="test")
        print("\nFinal Test Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Generate test plots
        print("\nGenerating ROC and PR curves for test set...")
        test_pred_output = self.trainer.predict(test_dataset)
        t_labels, t_probs = self._pred_to_labels_and_probs(test_pred_output)
        plot_all_curves(t_labels, t_probs, os.path.join(self.training_config['output_dir'], 'plots_test'))
        self._save_split_predictions(
            test_dataset,
            test_pred_output,
            split_name="test",
            metrics=test_results,
        )
        
        return eval_results, test_results
    
    def restore_weights_from_checkpoint(self, checkpoint_path: str):
        """
        Warm-start LoRA adapter weights AND the classifier head from a
        previously saved classifier checkpoint, WITHOUT resuming the training
        step counter, optimizer, or scheduler.

        Use this when the task is the same (e.g. pancreatic cancer prediction)
        but the dataset has changed: you want the same trained weights as a
        starting point, but a clean training run from step 0.

        The quantised base-model weights are intentionally skipped — they are
        already loaded correctly from `pretrained_checkpoint` and bitsandbytes
        NF4 state cannot be trivially restored from a safetensors file.

        Args:
            checkpoint_path: Directory of a Trainer checkpoint containing
                             `model.safetensors` (full LLMClassifier state).
        """
        print("\n" + "=" * 80)
        print("Warm-starting weights from checkpoint (LoRA adapters + classifier head)...")
        print("=" * 80)
        print(f"  - Source checkpoint: {checkpoint_path}")

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
            if any(kw in k for kw in lora_keywords) or k.startswith("classifier.")
        }

        lora_keys = [k for k in transfer_state if any(kw in k for kw in lora_keywords)]
        head_keys = [k for k in transfer_state if k.startswith("classifier.")]

        if not lora_keys:
            raise RuntimeError(
                f"No LoRA adapter keys found in {weights_file}. "
                f"Make sure the checkpoint was saved from an LLMClassifier with "
                f"LoRA adapters (keys containing lora_A / lora_B)."
            )
        if not head_keys:
            print(
                "  - No 'classifier.*' keys found in checkpoint; "
                "classifier head will be randomly initialised."
            )

        missing, unexpected = self.classifier_model.load_state_dict(transfer_state, strict=False)
        print(f"  - Loaded {len(lora_keys)} LoRA parameter tensors.")
        print(f"  - Loaded {len(head_keys)} classifier head tensors.")

        # Critical check: unexpected keys mean the key names in transfer_state
        # do NOT exist in the current model — those weights were silently dropped.
        if unexpected:
            print(
                f"  - *** WARNING: {len(unexpected)} keys from the checkpoint could NOT "
                f"be matched to the current model and were DROPPED (weights NOT loaded). "
                f"This likely means a key-name mismatch between the checkpoint and the "
                f"current model (e.g. different Unsloth/PEFT version). ***"
            )
            print("  - First 10 unmatched keys:")
            for k in unexpected[:10]:
                print(f"      {k}")
        else:
            print("  - All transfer keys matched the current model successfully.")

        if missing:
            # Non-LoRA / non-classifier missing keys are expected (base model weights).
            unexpected_missing = [
                k for k in missing
                if any(kw in k for kw in lora_keywords) or k.startswith("classifier.")
            ]
            if unexpected_missing:
                print(
                    f"  - *** WARNING: {len(unexpected_missing)} LoRA/classifier keys are "
                    f"missing from the checkpoint (will use random init): "
                    f"{unexpected_missing[:5]}{'...' if len(unexpected_missing) > 5 else ''} ***"
                )

        # ── Sanity-check: verify a LoRA weight was actually changed ──────────
        # Use std and a specific element — LoRA A means are always ≈ 0 by init.
        if lora_keys:
            probe_key = lora_keys[0]
            if probe_key in model_state and probe_key in transfer_state:
                # Keep everything on CPU for comparison — state_dict() may
                # return GPU tensors depending on where the model lives.
                t_before = model_state[probe_key].float().cpu()
                t_ckpt   = transfer_state[probe_key].float().cpu()
                t_after  = self.classifier_model.state_dict()[probe_key].float().cpu()
                logger.info(
                    f"  - Sanity check on '{probe_key}':\n"
                    f"      std  before={t_before.std().item():.6f}  "
                    f"ckpt={t_ckpt.std().item():.6f}  "
                    f"after={t_after.std().item():.6f}\n"
                    f"      [0,0] before={t_before.flatten()[0].item():.8f}  "
                    f"ckpt={t_ckpt.flatten()[0].item():.8f}  "
                    f"after={t_after.flatten()[0].item():.8f}"
                )
                if not torch.allclose(t_after, t_ckpt, atol=1e-5):
                    print("  - *** Sanity check FAILED: loaded tensor does not match checkpoint! ***")
                elif torch.allclose(t_before, t_after, atol=1e-8):
                    print(
                        "  - *** Sanity check WARNING: tensor unchanged after load. "
                        "The pretrained_checkpoint may already contain these LoRA values "
                        "(expected if both checkpoints share the same pretraining base). ***"
                    )
                else:
                    logger.info("  - Sanity check PASSED: LoRA weight matches checkpoint and differs from pre-load value.")
        # ─────────────────────────────────────────────────────────────────────

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
        metrics and the plot_all_curves helper.

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
        """Persist per-sample labels and probabilities for a split."""
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

    def _generate_plots(self, pred_output, plot_dir_name: str):
        """Helper to generate ROC and PR curves."""
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        
        multi_label = self.training_config.get('multi_label', False)
        if multi_label:
            print("  - Skipping ROC/PR plots for multi-label (not yet supported)")
            return
        
        probs = self._logits_to_pos_probs(logits)
        labels = pred_output.label_ids
        
        # Create plots
        plot_output_dir = os.path.join(self.training_config['output_dir'], plot_dir_name)
        plot_all_curves(labels, probs, plot_output_dir)
        print(f"  - Plots saved to: {plot_output_dir}")
    
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
        
        # Evaluate
        eval_results, test_results = self.evaluate_and_visualize(val_dataset, test_dataset)
        
        return eval_results, test_results
