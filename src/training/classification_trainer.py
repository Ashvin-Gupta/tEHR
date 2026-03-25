"""
EHR Classification Trainer - orchestrates classification fine-tuning workflow.
"""
import os
import time
import torch
from typing import Dict, Any
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import wandb
from torch.utils.data import ConcatDataset

from src.data.classification_collator import ClassificationCollator
from src.training.training_utils import compute_and_sort_by_length
from src.models.llm_classifier import LLMClassifier
from src.training.model_loader import load_pretrained_lora_model
from src.training.metrics import compute_classification_metrics
from src.evaluations.evaluations import plot_all_curves

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
        """Load pretrained model with LoRA adapters from pretraining stage."""
        print("\n" + "=" * 80)
        print("Loading pretrained model...")
        print("=" * 80)
        
        pretrained_checkpoint = self.model_config.get('pretrained_checkpoint')
        if not pretrained_checkpoint:
            raise ValueError("'model.pretrained_checkpoint' must be set in config")
        
        self.base_model, self.tokenizer = load_pretrained_lora_model(
            pretrained_checkpoint=pretrained_checkpoint,
            max_seq_length=self.model_config.get('max_length', 8192),
            load_in_4bit=self.training_config.get('load_in_4bit', True),
            local_rank=self.local_rank
        )
    
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
            head_dropout=self.model_config.get('head_dropout', 0.1)
        )

    def _sort_with_cache(self, dataset, split: str, shuffle_buckets: bool, num_buckets: int = 20):
        """
        Sort a dataset by sequence length, caching the sorted indices so that
        we do not need to recompute lengths every time we run a new model.
        Uses a shared cache dir (data_cache_dir) when set, so all seeds/ranks
        reuse the same cache. Uses a lock file so only one process computes when
        multiple ranks start together.
        """
        if not self.data_config.get('sort_by_length', False):
            return dataset

        dataset_name = self.data_config.get("dataset", "CPRD")
        tokenizer_id = getattr(self.tokenizer, "name_or_path", "tokenizer")
        # Use shared cache root when set (base output dir), so all seeds share one cache
        cache_base = self.training_config.get(
            "data_cache_dir", self.training_config["output_dir"]
        )
        cache_root = os.path.join(cache_base, "data_cache", "length_sorted")
        os.makedirs(cache_root, exist_ok=True)

        cache_filename = f"{dataset_name}_{split}_sorted_{os.path.basename(str(tokenizer_id))}.pt"
        cache_path = os.path.join(cache_root, cache_filename)
        lock_path = cache_path + ".lock"

        def _load_indices(path):
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(path, map_location="cpu")

        # Fast path: cache already exists
        if os.path.exists(cache_path):
            print(f"  - Found cached length-sorted indices for split '{split}', loading from disk...")
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
            print(f"  - No cached length-sorted indices for split '{split}', computing and caching...")
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
                print(f"  - Found cached length-sorted indices for split '{split}' (written by another process), loading...")
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
                labels.append(int(lbl.item()) if hasattr(lbl, 'item') else int(lbl))
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

        training_args_kwargs = dict(
            output_dir=self.training_config['output_dir'],
            overwrite_output_dir=bool(self.training_config.get("overwrite_output_dir", False)),
            run_name=run_name,
            report_to="wandb" if self.wandb_config.get('enabled', False) else "none",
            
            # Training hyperparameters
            num_train_epochs=int(self.training_config.get('epochs', 10)),
            per_device_train_batch_size=int(self.training_config.get('batch_size', 8)),
            per_device_eval_batch_size=int(self.training_config.get('eval_batch_size', 8)),
            learning_rate=float(self.training_config.get('learning_rate', 1e-3)),
            weight_decay=float(self.training_config.get('weight_decay', 0.01)),
            warmup_steps=int(self.training_config.get('warmup_steps', 100)),
            
            # Gradient settings
            gradient_accumulation_steps=int(self.training_config.get('gradient_accumulation_steps', 1)),
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
            eval_steps=int(self.training_config.get('eval_steps', 100)),
            save_strategy="steps",
            save_steps=int(self.training_config.get('save_steps', 500)),
            save_total_limit=int(self.training_config.get('save_total_limit', 2)),
            
            # Best model tracking
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
        """Run the training loop."""
        print("\n" + "=" * 80)
        print("Starting classification training...")
        print("=" * 80)

        if resume_from_checkpoint:
            if not os.path.exists(resume_from_checkpoint):
                raise ValueError(f"Checkpoint not found: {resume_from_checkpoint}")
            print(f"  - Resuming from checkpoint: {resume_from_checkpoint}")

            # Auto-detect checkpoint format: if the checkpoint only contains
            # model.safetensors (and no pytorch_model.bin), force safetensors
            # loading so the Trainer doesn't mistakenly look for the .bin file.
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

    def _pred_to_labels_and_probs(self, pred_output):
        """
        Convert a PredictionOutput into 1-D numpy arrays suitable for sklearn
        metrics and the plot_all_curves helper.

        - label_ids may be 1-D class indices or 2-D one-hot; we always return
          1-D class indices.
        - predictions are raw logits of shape (N, C); we softmax them and
          return the positive-class (index 1) column as a 1-D probability array.
        """
        import numpy as np

        labels = pred_output.label_ids
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        # One-hot → class indices
        if np.ndim(labels) == 2:
            labels = np.argmax(labels, axis=1)

        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
        # For binary classification extract the positive-class column
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs = probs[:, 1]

        return labels, probs

    def _generate_plots(self, pred_output, plot_dir_name: str):
        """Helper to generate ROC and PR curves."""
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        
        multi_label = self.training_config.get('multi_label', False)
        if multi_label:
            print("  - Skipping ROC/PR plots for multi-label (not yet supported)")
            return
        
        # Convert to probabilities.
        # IMPORTANT: keep full [n_samples, n_classes] matrix so that downstream
        # plotting code can handle both binary and multi-class cases.
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        labels = pred_output.label_ids
        
        # Create plots
        plot_output_dir = os.path.join(self.training_config['output_dir'], plot_dir_name)
        plot_classification_performance(labels, probs, plot_output_dir)
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
