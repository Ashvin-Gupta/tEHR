"""
Baseline: Qwen3-Embedding-8B with Linear Classifier

Uses Qwen3-Embedding-8B (text embedding model) as encoder with a linear classification head.
Similar to ClinicalBERT but with the Qwen3 embedding model for better performance.
"""
import os
import torch
import numpy as np
from typing import Dict, Any
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
import wandb

from src.baselines.utils import load_baseline_config, setup_output_dir, load_datasets
from src.models.qwen3_embedding_classifier import Qwen3EmbeddingClassifier
from src.data.classification_collator import ClassificationCollator
from src.training.metrics import compute_classification_metrics
from src.evaluations.baseline_metrics import compute_baseline_metrics, plot_all_curves, save_results, print_results


class Qwen3EmbeddingBaseline:
    """
    Qwen3-Embedding baseline for EHR classification.
    
    Uses Qwen3-Embedding-8B as encoder with last-token pooling and a linear head.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model', {})
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        self.wandb_config = config.get('wandb', {})
        self.output_dir = self.training_config.get('output_dir', './outputs/qwen3_embedding')

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def setup_wandb(self) -> str:
        """Setup WandB logging."""
        if self.wandb_config.get('enabled', False):
            run_name = self.wandb_config.get("run_name") or f"qwen3_embedding_{self.config.get('name', 'default')}"
            if self.local_rank == 0:
                wandb.init(
                    project=self.wandb_config.get("project", "llm-classification"),
                    name=run_name,
                    config=self.config
                )
                print(f"\nWandB enabled - Project: {self.wandb_config.get('project')}, Run: {run_name}")
            return run_name
        return "qwen3-embedding-run"

    def load_model(self):
        """Load Qwen3-Embedding model and tokenizer."""
        print("\n" + "=" * 80)
        print("Loading Qwen3-Embedding model...")
        print("=" * 80)

        model_name = self.model_config.get('model_name', 'Qwen/Qwen3-Embedding-8B')
        hidden_size = self.model_config.get('hidden_size', 4096)
        num_labels = self.model_config.get('num_labels', 2)
        head_dropout = self.model_config.get('head_dropout', 0.1)
        max_length = self.data_config.get('max_length', 8192)

        print(f"  - Model: {model_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Num labels: {num_labels}")

        # Load tokenizer - use left padding as recommended for Qwen3-Embedding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base embedding model
        base_model = AutoModel.from_pretrained(model_name)

        # Wrap with classification head
        self.model = Qwen3EmbeddingClassifier(
            base_model=base_model,
            hidden_size=hidden_size,
            num_labels=num_labels,
            head_dropout=head_dropout,
        )

        print(f"  - Model loaded successfully")
        print(f"  - Vocab size: {len(self.tokenizer)}")

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
            per_device_train_batch_size=int(self.training_config.get('batch_size', 4)),
            per_device_eval_batch_size=int(self.training_config.get('eval_batch_size', 8)),
            learning_rate=float(self.training_config.get('learning_rate', 2e-5)),
            weight_decay=float(self.training_config.get('weight_decay', 0.01)),
            warmup_steps=int(self.training_config.get('warmup_steps', 100)),
            gradient_accumulation_steps=int(self.training_config.get('gradient_accumulation_steps', 2)),
            fp16=bool(self.training_config.get('fp16', False)),
            bf16=bool(self.training_config.get('bf16', True)),
            logging_steps=int(self.training_config.get('logging_steps', 50)),
            eval_steps=int(self.training_config.get('eval_steps', 250)),
            save_steps=int(self.training_config.get('save_steps', 500)),
            save_total_limit=int(self.training_config.get('save_total_limit', 2)),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            dataloader_num_workers=int(self.training_config.get('dataloader_num_workers', 4)),
            report_to="wandb" if self.wandb_config.get('enabled', False) else "none",
            remove_unused_columns=False,
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
        print("Training Qwen3-Embedding classifier...")
        print("=" * 80)
        self.trainer.train()
        print("\nTraining completed!")

    def evaluate(self, dataset, split_name: str = "validation") -> Dict[str, float]:
        """Evaluate model on a dataset."""
        print(f"\n" + "=" * 80)
        print(f"Evaluating on {split_name} set...")
        print("=" * 80)

        pred_output = self.trainer.predict(dataset)
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        labels = pred_output.label_ids

        metrics = compute_baseline_metrics(labels, probs)
        print_results(metrics, split_name)

        plot_dir = os.path.join(self.output_dir, 'plots', split_name)
        plot_all_curves(labels, probs, plot_dir)

        results = {
            'metrics': metrics,
            'labels': labels.tolist(),
            'probs': probs.tolist()
        }
        save_results(results, os.path.join(self.output_dir, 'results'), f'{split_name}_results.json')

        return metrics

    def save_model(self):
        """Save the trained model (custom nn.Module - save state_dict + tokenizer)."""
        final_model_path = os.path.join(self.output_dir, 'final_model')
        os.makedirs(final_model_path, exist_ok=True)
        # Save full model state (base + classifier)
        torch.save(self.model.state_dict(), os.path.join(final_model_path, 'pytorch_model.bin'))
        self.tokenizer.save_pretrained(final_model_path)
        # Save config for loading
        import json
        with open(os.path.join(final_model_path, 'config.json'), 'w') as f:
            json.dump({
                'model_name': self.model_config.get('model_name'),
                'hidden_size': self.model_config.get('hidden_size', 4096),
                'num_labels': self.model_config.get('num_labels', 2),
                'head_dropout': self.model_config.get('head_dropout', 0.1),
            }, f, indent=2)
        print(f"\n  - Model saved to: {final_model_path}")

    def run(self):
        """Run the complete baseline pipeline."""
        setup_output_dir(self.output_dir, overwrite=self.training_config.get('overwrite_output_dir', False))

        run_name = self.setup_wandb()
        self.load_model()

        datasets = load_datasets(
            self.data_config,
            splits=['train', 'tuning', 'held_out'],
            format='text',
            tokenizer=self.tokenizer
        )

        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.get('max_length', 8192),
            truncation=True,
        )

        self.create_trainer(datasets['train'], datasets['tuning'], collator, run_name)
        self.train()
        self.save_model()

        val_metrics = self.evaluate(datasets['tuning'], 'validation')
        test_metrics = self.evaluate(datasets['held_out'], 'test')

        summary = {
            'validation': val_metrics,
            'test': test_metrics
        }
        save_results(summary, self.output_dir, 'summary.json')

        print("\n" + "=" * 80)
        print("Qwen3-Embedding Baseline Complete!")
        print("=" * 80)

        return val_metrics, test_metrics
