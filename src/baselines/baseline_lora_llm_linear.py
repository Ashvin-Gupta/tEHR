"""
Baseline: LLM with LoRA Adapters + Linear Classifier

Uses base Qwen model, adds LoRA adapters, and trains both LoRA and linear head.
"""
import os
import torch
import numpy as np
from typing import Dict, Any
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
import wandb

from src.baselines.utils import load_baseline_config, setup_output_dir, load_datasets
from src.models.llm_classifier import LLMClassifier
from src.data.classification_collator import ClassificationCollator
from src.training.metrics import compute_classification_metrics
from src.evaluations.baseline_metrics import compute_baseline_metrics, plot_all_curves, save_results, print_results


class LoRALLMLinearBaseline:
    """
    Base LLM with LoRA adapters and trainable linear classification head.
    Trains both LoRA adapters and the linear head.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LoRA LLM + linear baseline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.lora_config = config.get('lora', {})
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        self.wandb_config = config.get('wandb', {})
        self.output_dir = self.training_config.get('output_dir', './outputs/lora_llm_linear')
        
        self.base_model = None
        self.tokenizer = None
        self.classifier_model = None
        self.trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def setup_wandb(self) -> str:
        """
        Setup WandB logging (same pattern as main classifier).

        Returns:
            Run name for the experiment.
        """
        if self.wandb_config.get('enabled', False):
            run_name = self.wandb_config.get("run_name")
            if run_name is None:
                run_name = f"lora_llm_linear_{self.config.get('name', 'default')}"

            if self.local_rank == 0:
                wandb.init(
                    project=self.wandb_config.get("project", "llm-classification"),
                    name=run_name,
                    config=self.config
                )
                print(f"\nWandB enabled - Project: {self.wandb_config.get('project', 'llm-classification')}, Run: {run_name}")

            return run_name
        return "lora-llm-linear-run"
    
    def load_model(self):
        """Load base Qwen model (no pretraining)."""
        print("\n" + "=" * 80)
        print("Loading base Qwen model...")
        print("=" * 80)
        
        model_name = self.model_config.get('model_name', 'unsloth/Qwen3-8B-Base-unsloth-bnb-4bit')
        max_length = self.model_config.get('max_length', 8192)
        load_in_4bit = self.training_config.get('load_in_4bit', True)
        
        print(f"  - Model: {model_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - 4-bit quantization: {load_in_4bit}")
        
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map="auto"
        )
        
        print(f"  - Model loaded successfully")
        print(f"  - Vocab size: {len(self.tokenizer)}")
    
    def apply_lora(self):
        """Apply LoRA adapters to the base model."""
        print("\n" + "=" * 80)
        print("Applying LoRA adapters...")
        print("=" * 80)
        
        r = self.lora_config.get('r', 16)
        target_modules = self.lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ])
        lora_alpha = self.lora_config.get('lora_alpha', 16)
        lora_dropout = self.lora_config.get('lora_dropout', 0.05)
        bias = self.lora_config.get('bias', "none")
        use_rslora = self.lora_config.get('use_rslora', True)
        
        print(f"  - LoRA rank (r): {r}")
        print(f"  - LoRA alpha: {lora_alpha}")
        print(f"  - LoRA dropout: {lora_dropout}")
        print(f"  - Target modules: {target_modules}")
        print(f"  - Use RSLoRA: {use_rslora}")
        
        self.base_model = FastLanguageModel.get_peft_model(
            self.base_model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=self.training_config.get('gradient_checkpointing', 'unsloth'),
            random_state=42,
            use_rslora=use_rslora,
            loftq_config=self.lora_config.get('loftq_config', None),
        )
        
        print("  - LoRA adapters applied successfully")
    
    def create_classifier(self):
        """Create LLMClassifier with trainable LoRA adapters and linear head."""
        print("\n" + "=" * 80)
        print("Creating LoRA LLM + linear classifier...")
        print("=" * 80)
        
        hidden_size = self.model_config.get('hidden_size', 4096)
        num_labels = self.model_config.get('num_labels', 2)
        head_type = self.model_config.get('head_type', 'linear')
        head_hidden_size = self.model_config.get('head_hidden_size', 512)
        head_dropout = self.model_config.get('head_dropout', 0.1)
        
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Num labels: {num_labels}")
        print(f"  - Head type: {head_type}")
        print(f"  - Training: LoRA adapters + Linear head")
        
        self.classifier_model = LLMClassifier(
            base_model=self.base_model,
            hidden_size=hidden_size,
            num_labels=num_labels,
            freeze_base=True,  # Freeze base model parameters
            trainable_param_keywords=["lora_"],  # Train LoRA adapters
            multi_label=self.training_config.get('multi_label', False),
            tokenizer=self.tokenizer,
            head_type=head_type,
            head_hidden_size=head_hidden_size,
            head_dropout=head_dropout
        )
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.classifier_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.classifier_model.parameters())
        print(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
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
            per_device_train_batch_size=int(self.training_config.get('batch_size', 2)),
            per_device_eval_batch_size=int(self.training_config.get('eval_batch_size', 1)),
            learning_rate=float(self.training_config.get('learning_rate', 1e-4)),
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
            dataloader_num_workers=int(self.training_config.get('dataloader_num_workers', 8)),
            report_to="wandb" if self.wandb_config.get('enabled', False) else "none",
            remove_unused_columns=False,
        )
        
        self.trainer = Trainer(
            model=self.classifier_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_classification_metrics,
        )
        
        print("  - Trainer created successfully")
    
    def train(self):
        """Train the LoRA adapters and linear classifier head."""
        print("\n" + "=" * 80)
        print("Training LoRA adapters + linear classifier head...")
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
        
        # Apply LoRA adapters
        self.apply_lora()
        
        # Create classifier
        self.create_classifier()
        
        # Load datasets
        datasets = load_datasets(
            self.data_config,
            splits=['train', 'tuning', 'held_out'],
            format='text',
            tokenizer=self.tokenizer
        )
        
        # Create collator
        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.get('max_length', 12000),
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
        
        print("\n" + "=" * 80)
        print("LoRA LLM + Linear Baseline Complete!")
        print("=" * 80)
        
        return val_metrics, test_metrics

