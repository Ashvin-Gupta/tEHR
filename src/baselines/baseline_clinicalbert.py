"""
Baseline 5: ClinicalBERT Fine-tuning

Fine-tunes emilyalsentzer/Bio_ClinicalBERT on EHR text for cancer prediction.
"""
import os
import torch
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import wandb

from src.baselines.utils import load_baseline_config, setup_output_dir, load_datasets
from src.data.classification_collator import ClassificationCollator
from src.training.metrics import compute_classification_metrics
from src.evaluations.baseline_metrics import compute_baseline_metrics, plot_all_curves, save_results, print_results


class ClinicalBERTBaseline:
    """
    ClinicalBERT fine-tuning baseline for EHR classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ClinicalBERT baseline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        self.wandb_config = config.get('wandb', {})
        self.output_dir = config.get('output_dir', './outputs/clinicalbert')
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def setup_wandb(self) -> str:
        """
        Setup WandB logging for the ClinicalBERT baseline.

        Returns:
            Run name for the experiment.
        """
        if self.wandb_config.get('enabled', False):
            run_name = self.wandb_config.get("run_name")
            if run_name is None:
                run_name = f"clinicalbert_{self.config.get('name', 'default')}"

            if self.local_rank == 0:
                wandb.init(
                    project=self.wandb_config.get("project", "llm-classification"),
                    name=run_name,
                    config=self.config
                )
                print(f"\nWandB enabled - Project: {self.wandb_config.get('project', 'llm-classification')}, Run: {run_name}")

            return run_name
        return "clinicalbert-baseline-run"
    
    def load_model(self):
        """Load ClinicalBERT model and tokenizer."""
        print("\n" + "=" * 80)
        print("Loading ClinicalBERT model...")
        print("=" * 80)
        
        model_name = self.model_config.get('model_name', 'emilyalsentzer/Bio_ClinicalBERT')
        num_labels = self.model_config.get('num_labels', 2)
        max_length = self.data_config.get('max_length', 512)
        
        print(f"  - Model: {model_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - Num labels: {num_labels}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        print(f"  - Model loaded successfully")
        print(f"  - Vocab size: {len(self.tokenizer)}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def create_trainer(self, train_dataset, val_dataset, collator, run_name: str):
        """Create HuggingFace Trainer."""
        print("\n" + "=" * 80)
        print("Setting up training...")
        print("=" * 80)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, 'checkpoints'),
            run_name=run_name,
            overwrite_output_dir=self.training_config.get('overwrite_output_dir', True),
            num_train_epochs=int(self.training_config.get('epochs', 3)),
            per_device_train_batch_size=int(self.training_config.get('batch_size', 16)),
            per_device_eval_batch_size=int(self.training_config.get('eval_batch_size', 16)),
            learning_rate=float(self.training_config.get('learning_rate', 2e-5)),
            weight_decay=float(self.training_config.get('weight_decay', 0.01)),
            warmup_steps=int(self.training_config.get('warmup_steps', 100)),
            gradient_accumulation_steps=int(self.training_config.get('gradient_accumulation_steps', 1)),
            fp16=bool(self.training_config.get('fp16', False)),
            bf16=bool(self.training_config.get('bf16', False)),
            logging_steps=int(self.training_config.get('logging_steps', 50)),
            eval_steps=int(self.training_config.get('eval_steps', 250)),
            save_steps=int(self.training_config.get('save_steps', 500)),
            save_total_limit=int(self.training_config.get('save_total_limit', 2)),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            dataloader_num_workers=int(self.training_config.get('dataloader_num_workers', 8)),
            report_to="wandb" if self.wandb_config.get('enabled', False) else "none",
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_classification_metrics,
        )
        
        print("  - Trainer created successfully")
    
    def train(self):
        """Train the model."""
        print("\n" + "=" * 80)
        print("Training ClinicalBERT...")
        print("=" * 80)
        
        self.trainer.train()
        
        print("\nTraining completed!")
    
    def evaluate(self, dataset, split_name: str = "validation") -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: UnifiedEHRDataset instance
            split_name: Name of the split
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n" + "=" * 80)
        print(f"Evaluating on {split_name} set...")
        print("=" * 80)
        
        # Get predictions
        pred_output = self.trainer.predict(dataset)
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        
        # Convert to probabilities
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        labels = pred_output.label_ids
        
        # Compute metrics
        metrics = compute_baseline_metrics(labels, probs)
        
        # Print results
        print_results(metrics, split_name)
        
        # Save all plots (ROC, PR, and Calibration)
        plot_dir = os.path.join(self.output_dir, 'plots', split_name)
        plot_all_curves(labels, probs, plot_dir)
        
        # Save predictions
        results = {
            'metrics': metrics,
            'labels': labels.tolist(),
            'probs': probs.tolist()
        }
        save_results(results, os.path.join(self.output_dir, 'results'), f'{split_name}_results.json')
        
        return metrics
    
    def save_model(self):
        """Save the trained model."""
        final_model_path = os.path.join(self.output_dir, 'final_model')
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        print(f"\n  - Model saved to: {final_model_path}")
    
    def run(self):
        """Run the complete baseline training and evaluation pipeline."""
        # Setup
        setup_output_dir(self.output_dir, overwrite=self.training_config.get('overwrite_output_dir', False))

        # WandB
        run_name = self.setup_wandb()
        
        # Load model
        self.load_model()
        
        # Load datasets
        datasets = load_datasets(
            self.data_config,
            splits=['train', 'tuning', 'held_out'],
            format='text',
            tokenizer=self.tokenizer
        )

        # ------------------------------------------------------------------
        # Pre-pass: measure how many samples would be dropped (label present
        # but no text) BEFORE training/evaluation.
        # ------------------------------------------------------------------
        debug_collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.get('max_length', 512),
            truncation=True,
        )
        debug_loader = DataLoader(
            datasets['train'],
            batch_size=int(self.training_config.get('batch_size', 16)),
            shuffle=False,
            num_workers=int(self.training_config.get('dataloader_num_workers', 8)),
            collate_fn=debug_collator,
        )
        print("\n" + "=" * 80)
        print("Running pre-pass over training data to count dropped samples...")
        for _ in debug_loader:
            pass
        debug_stats = debug_collator.get_stats()
        print(f"  - Total sequences seen (train): {debug_stats.get('total_sequences', 0)}")
        print(f"  - Samples with label but no text (dropped): {debug_stats.get('missing_text_samples', 0)}")
        print("=" * 80)
        # ------------------------------------------------------------------
        # Now create a fresh collator for actual training/eval
        # ------------------------------------------------------------------

        # Create collator that truncates from the start (keeps most recent events)
        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.get('max_length', 512),
            truncation=True,
        )
        
        # Create trainer
        self.create_trainer(datasets['train'], datasets['tuning'], collator, run_name)
        
        # Train
        self.train()
        
        # Save model
        self.save_model()
        
        # Evaluate
        val_metrics = self.evaluate(datasets['tuning'], 'validation')
        test_metrics = self.evaluate(datasets['held_out'], 'test')
        
        # Save summary
        summary = {
            'validation': val_metrics,
            'test': test_metrics
        }
        save_results(summary, self.output_dir, 'summary.json')

        # Report collator statistics (including missing-text samples)
        if hasattr(collator, "get_stats"):
            stats = collator.get_stats()
            print("\n" + "=" * 80)
            print("Collator statistics:")
            print(f"  - Total sequences seen: {stats.get('total_sequences', 0)}")
            print(f"  - Long sequences (> max_length): {stats.get('long_sequences', 0)} "
                  f"({stats.get('percentage_long', 0):.2f}% of total)")
            print(f"  - Samples with label but no text (dropped): {stats.get('missing_text_samples', 0)}")
            print("=" * 80)
        
        print("\n" + "=" * 80)
        print("ClinicalBERT Baseline Complete!")
        print("=" * 80)
        
        return val_metrics, test_metrics

