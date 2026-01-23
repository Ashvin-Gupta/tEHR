"""
EHR Classification Trainer - orchestrates classification fine-tuning workflow.
"""
import os
import torch
from typing import Dict, Any
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import wandb

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.classification_collator import ClassificationCollator
from src.data.dataset_utils import compute_and_sort_by_length
from src.models.llm_classifier import LLMClassifier
from src.training.model_loader import load_pretrained_lora_model
from src.training.metrics import compute_classification_metrics
from src.evaluations.visualisation import plot_classification_performance


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
    
    def prepare_datasets(self):
        """Prepare train, validation, and test datasets."""
        print("\n" + "=" * 80)
        print("Loading datasets...")
        print("=" * 80)
        
        dataset_args = {
            "data_dir": self.data_config["data_dir"],
            "vocab_file": self.data_config["vocab_filepath"],
            "labels_file": self.data_config["labels_filepath"],
            "medical_lookup_file": self.data_config["medical_lookup_filepath"],
            "lab_lookup_file": self.data_config["lab_lookup_filepath"],
            "region_lookup_file": self.data_config["region_lookup_filepath"],
            "time_lookup_file": self.data_config["time_lookup_filepath"],
            "format": 'text',
            "cutoff_months": self.data_config.get("cutoff_months", 1),
            "max_sequence_length": None,
            "tokenizer": None,
        }

        train_dataset = UnifiedEHRDataset(split="train", **dataset_args)
        val_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
        test_dataset = UnifiedEHRDataset(split="held_out", **dataset_args)

        # Optional: Sort by length for efficient batching
        if self.data_config.get('sort_by_length', False):
            print("\n  - Sorting datasets by sequence length...")
            train_dataset = compute_and_sort_by_length(
                train_dataset, self.tokenizer, shuffle_buckets=True, num_buckets=20
            )
            val_dataset = compute_and_sort_by_length(
                val_dataset, self.tokenizer, shuffle_buckets=False
            )
            test_dataset = compute_and_sort_by_length(
                test_dataset, self.tokenizer, shuffle_buckets=False
            )
            print("  ✓ Datasets sorted by length")
        
        print(f"  - Train dataset: {len(train_dataset)} patients")
        print(f"  - Validation dataset: {len(val_dataset)} patients")
        print(f"  - Test dataset: {len(test_dataset)} patients")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_collator(self):
        """Create data collator for batching."""
        print("\n" + "=" * 80)
        print("Creating data collator...")
        print("=" * 80)
        
        multi_label = self.training_config.get('multi_label', False)
        binary_classification = not multi_label
        
        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.get('max_length'),
            binary_classification=binary_classification,
            truncation=False,
            handle_long_sequences=self.data_config.get('handle_long_sequences', 'warn')
        )
        
        print(f"  - Max length: {self.data_config.get('max_length')}")
        print(f"  - Truncation: False (keeping full patient trajectories)")
        print(f"  - Binary classification: {binary_classification}")
        
        return collator
    
    def create_trainer(self, train_dataset, val_dataset, collator, run_name: str):
        """Create HuggingFace Trainer."""
        print("\n" + "=" * 80)
        print("Setting up trainer...")
        print("=" * 80)
        
        training_args = TrainingArguments(
            output_dir=self.training_config['output_dir'],
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
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Other
            remove_unused_columns=False,
        )
        
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
    
    def train(self):
        """Run the training loop."""
        print("\n" + "=" * 80)
        print("Starting classification training...")
        print("=" * 80)
        
        self.trainer.train()
    
    def save_model(self):
        """Save the final trained model."""
        print("\n" + "=" * 80)
        print("Saving final model...")
        print("=" * 80)
        
        final_model_path = os.path.join(self.training_config['output_dir'], "final_model")
        self.trainer.save_model(final_model_path)
        print(f"  - Model saved to: {final_model_path}")
    
    def evaluate_and_visualize(self, val_dataset, test_dataset):
        """Run evaluation and create visualization plots."""
        # Evaluate on validation set
        print("\n" + "=" * 80)
        print("Running final evaluation on validation set...")
        print("=" * 80)
        
        eval_results = self.trainer.evaluate()
        print("\nFinal Validation Results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Generate validation plots
        print("\nGenerating ROC and PR curves for validation set...")
        pred_output = self.trainer.predict(val_dataset)
        self._generate_plots(pred_output, "plots_validation")
        
        # Evaluate on test set
        print("\n" + "=" * 80)
        print("Running evaluation on test set...")
        print("=" * 80)
        
        test_results = self.trainer.evaluate(test_dataset, metric_key_prefix="test")
        print("\nFinal Test Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")
        
        # Generate test plots
        print("\nGenerating ROC and PR curves for test set...")
        test_pred_output = self.trainer.predict(test_dataset)
        self._generate_plots(test_pred_output, "plots_test")
        
        return eval_results, test_results
    
    def _generate_plots(self, pred_output, plot_dir_name: str):
        """Helper to generate ROC and PR curves."""
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        
        multi_label = self.training_config.get('multi_label', False)
        if multi_label:
            print("  - Skipping ROC/PR plots for multi-label (not yet supported)")
            return
        
        # Convert to probabilities
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        labels = pred_output.label_ids
        
        # Create plots
        plot_output_dir = os.path.join(self.training_config['output_dir'], plot_dir_name)
        plot_classification_performance(labels, probs, plot_output_dir)
        print(f"  - Plots saved to: {plot_output_dir}")
    
    def run_full_pipeline(self):
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
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()
        
        # Create collator
        collator = self.create_collator()
        
        # Create trainer
        self.create_trainer(train_dataset, val_dataset, collator, run_name)
        
        # Train
        self.train()
        
        # Save
        self.save_model()
        
        # Evaluate
        eval_results, test_results = self.evaluate_and_visualize(val_dataset, test_dataset)
        
        return eval_results, test_results
