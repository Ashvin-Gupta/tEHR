"""
Pipeline script to evaluate a trained classification model.

This script loads a saved classification model (with classification head) and
runs evaluation on validation and test sets without any training.
"""
import argparse
import yaml
import pprint
import os
import torch
from typing import Dict, Any
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel

from src.training.utils import seed_all
from src.baselines.utils import load_baseline_config, load_datasets
from src.models.llm_classifier import LLMClassifier
from src.data.classification_collator import ClassificationCollator
from src.training.metrics import compute_classification_metrics
from src.evaluations.baseline_metrics import compute_baseline_metrics, plot_all_curves, save_results, print_results
from src.training.model_loader import load_classification_model, load_pretrained_lora_model


class ModelEvaluator:
    """
    Evaluates a saved classification model without training.
    """
    
    def __init__(self, config: Dict[str, Any], model_checkpoint: str):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
            model_checkpoint: Path to saved model checkpoint directory
        """
        self.config = config
        self.model_checkpoint = model_checkpoint
        self.data_config = config['data']
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.output_dir = self.training_config.get('output_dir', './outputs/evaluation')
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    def load_model(self):
        """Load the saved classification model (base LLM + LoRA, not the classifier head)."""
        print("\n" + "=" * 80)
        print("Loading saved base model for classification...")
        print("=" * 80)
        
        max_length = self.model_config.get('max_length', 12000)
        load_in_4bit = self.training_config.get('load_in_4bit', True)
        
        # Case 1: Baselines that use a pretrained checkpoint (e.g. pretrained_llm_linear)
        # In these cases we should load the *pretrained* LLM (with LoRA) as the base,
        # and then separately load the classifier head from the classification checkpoint.
        pretrained_ckpt = self.model_config.get('pretrained_checkpoint')
        if pretrained_ckpt:
            print(f"  - Detected pretrained_checkpoint in config: {pretrained_ckpt}")
            print(f"  - max_length={max_length}, load_in_4bit={load_in_4bit}")
            try:
                self.model, self.tokenizer = load_pretrained_lora_model(
                    pretrained_checkpoint=pretrained_ckpt,
                    max_seq_length=max_length,
                    load_in_4bit=load_in_4bit,
                    local_rank=self.local_rank,
                )
                print(f"  - Base model loaded from pretrained checkpoint (for classification head from {self.model_checkpoint})")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load base model from pretrained_checkpoint={pretrained_ckpt}. "
                    f"Error: {e}\n"
                    f"Make sure this path points to the Stage-1 pretraining checkpoint."
                )
            return
        
        # Case 2: Other models (e.g. off-the-shelf base models saved directly as classifiers)
        # Fall back to loading the full classification model from the checkpoint directory.
        try:
            self.model, self.tokenizer = load_classification_model(
                model_checkpoint=self.model_checkpoint,
                max_seq_length=max_length,
                load_in_4bit=load_in_4bit,
                local_rank=self.local_rank,
            )
            print(f"  - Model loaded from classification checkpoint: {self.model_checkpoint}")
            print(f"  - Max length: {max_length}")
            print(f"  - 4-bit quantization: {load_in_4bit}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.model_checkpoint}. "
                f"Error: {e}\n"
                f"Make sure the checkpoint path is correct and contains valid model files."
            )
    
    def wrap_with_classifier(self):
        """
        Wrap the loaded model with a classification head if needed.
        
        When Trainer saves an LLMClassifier, it saves the full model state.
        We need to reconstruct the LLMClassifier wrapper and load the state so
        that both the base model and the classifier head match training.
        """
        print("\n" + "=" * 80)
        print("Reconstructing classification model...")
        print("=" * 80)
        
        # Check if model already has a classifier head (might be saved as full LLMClassifier)
        has_classifier = any('classifier' in name.lower() for name, _ in self.model.named_modules())
        
        if has_classifier:
            print("  - Model already has a classification head (loaded as LLMClassifier)")
            # Model is already an LLMClassifier, use as-is
            self.model = self.model
        else:
            # Need to wrap with LLMClassifier
            print("  - Wrapping base model with classification head")
            
            hidden_size = self.model_config.get('hidden_size', 4096)
            num_labels = self.model_config.get('num_labels', 2)
            head_type = self.model_config.get('head_type', 'linear')
            head_hidden_size = self.model_config.get('head_hidden_size', 512)
            head_dropout = self.model_config.get('head_dropout', 0.1)
            
            # For evaluation, freeze everything
            trainable_keywords = []
            
            # Create LLMClassifier wrapper with the same config as training.
            # We will then load the *full* state dict from the Trainer checkpoint
            # so that the classifier head matches the trained one.
            classifier_model = LLMClassifier(
                base_model=self.model,
                hidden_size=hidden_size,
                num_labels=num_labels,
                freeze_base=True,
                trainable_param_keywords=trainable_keywords,
                multi_label=self.training_config.get('multi_label', False),
                tokenizer=self.tokenizer,
                head_type=head_type,
                head_hidden_size=head_hidden_size,
                head_dropout=head_dropout
            )
            
            # Try to load the full model state (including classifier) if it exists.
            # Trainer saves the full model state in pytorch_model.bin or model.safetensors
            model_state_path = os.path.join(self.model_checkpoint, 'pytorch_model.bin')
            if not os.path.exists(model_state_path):
                model_state_path = os.path.join(self.model_checkpoint, 'model.safetensors')
            
            if os.path.exists(model_state_path):
                print(f"  - Loading full model state from checkpoint: {model_state_path}")
                try:
                    # Load the state dict saved by Trainer
                    if model_state_path.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(model_state_path)
                    else:
                        state_dict = torch.load(model_state_path, map_location='cpu')

                    # Load the full state dict into the reconstructed LLMClassifier.
                    # Using strict=False allows us to ignore any keys that don't match
                    # exactly (e.g. if the base model checkpoint differs slightly),
                    # while still restoring the trained classifier head weights.
                    missing, unexpected = classifier_model.load_state_dict(state_dict, strict=False)
                    if missing:
                        print(f"  - Warning: {len(missing)} missing keys when loading state dict (first 5): {missing[:5]}")
                    if unexpected:
                        print(f"  - Warning: {len(unexpected)} unexpected keys when loading state dict (first 5): {unexpected[:5]}")
                    print("  - Loaded classifier (and base model where matching) from checkpoint.")
                except Exception as e:
                    print(f"  - ⚠️  Warning: Could not load model state dict: {e}")
                    print("     Using randomly initialized head (this may not be correct!)")
            else:
                print("  - ⚠️  No model state file found. Using randomly initialized head.")
                print("     This is expected if the model was saved without the classifier head.")
            
            self.model = classifier_model
    
    def create_trainer(self, val_dataset, test_dataset, collator):
        """Create HuggingFace Trainer for evaluation only."""
        print("\n" + "=" * 80)
        print("Setting up evaluation trainer...")
        print("=" * 80)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, 'eval_checkpoints'),
            per_device_eval_batch_size=int(self.training_config.get('eval_batch_size', 4)),
            fp16=bool(self.training_config.get('fp16', False)),
            bf16=bool(self.training_config.get('bf16', True)),
            dataloader_num_workers=int(self.training_config.get('dataloader_num_workers', 8)),
            remove_unused_columns=False,
            report_to="none",  # No logging during evaluation
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=val_dataset,  # For compute_metrics compatibility
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_classification_metrics,
        )
        
        print("  - Trainer created successfully")
    
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
    
    def run(self):
        """Run the complete evaluation pipeline."""
        # Setup output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self.load_model()
        
        # Wrap with classifier if needed
        self.wrap_with_classifier()
        
        # Load datasets
        datasets = load_datasets(
            self.data_config,
            splits=['tuning', 'held_out'],  # Only need validation and test
            format='text',
            tokenizer=self.tokenizer
        )
        
        # Create collator
        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.get('max_length', 12000),
        )
        
        # Create trainer
        self.create_trainer(datasets['tuning'], datasets['held_out'], collator)
        
        # Evaluate
        val_metrics = self.evaluate(datasets['tuning'], 'validation')
        test_metrics = self.evaluate(datasets['held_out'], 'test')
        
        # Save summary
        summary = {
            'model_checkpoint': self.model_checkpoint,
            'validation': val_metrics,
            'test': test_metrics
        }
        save_results(summary, self.output_dir, 'summary.json')
        
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        
        return val_metrics, test_metrics


def main(config_path: str, model_checkpoint: str):
    """
    Main function to evaluate a saved classification model.
    
    Args:
        config_path: Path to YAML configuration file (should match the training config).
        model_checkpoint: Path to saved model checkpoint directory (e.g., 
                         "output_dir/final_model" or "output_dir/checkpoints/checkpoint-5000").
    """
    # Set seed for reproducibility
    seed_all(42)
    
    # Load configuration
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    print(f"Loading model from: {model_checkpoint}")
    
    config = load_baseline_config(config_path)
    
    print('Loaded configuration:')
    pprint.pprint(config)
    print("=" * 80)
    
    # Verify checkpoint exists
    if not os.path.exists(model_checkpoint):
        raise ValueError(f"Model checkpoint not found: {model_checkpoint}")
    
    # Create evaluator and run
    evaluator = ModelEvaluator(config, model_checkpoint)
    val_metrics, test_metrics = evaluator.run()
    
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)
    
    return val_metrics, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Saved Classification Model"
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file (should match training config)"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint directory (e.g., output_dir/final_model)"
    )
    args = parser.parse_args()
    
    main(args.config_filepath, args.model_checkpoint)

