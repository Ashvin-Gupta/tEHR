"""
EHR Pretrainer class for managing continued pretraining workflow.
"""
import os
from datetime import datetime
from typing import Dict, Any, Optional
import torch
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer
import wandb

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.preprocessing import extract_text
from src.training.callbacks import InferenceCallback, PackingVerificationCallback


class EHRPretrainer:
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
        """
        Setup WandB logging.
        
        Returns:
            Tuple of (run_name, report_to)
        """
        # Build default run name from hyperparameters
        model_name = self.model_config['model_name'].split('/')[-1]
        default_run_name = (
            f"{model_name}-pretrain-{self.data_config['cutoff_months']}month-cutoff"
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
        
        if self.wandb_config.get('enabled', False):
            run_name = self.wandb_config.get("run_name", default_run_name)
            report_to = "wandb"

            if self.local_rank == 0:
                os.environ["WANDB_PROJECT"] = self.wandb_config.get("project", "ehr-llm-pretraining")
                
                wandb.init(
                    project=self.wandb_config.get("project", "ehr-llm-pretraining"), 
                    config=self.config, 
                    name=run_name
                )
                wandb.config.update(self.config, allow_val_change=True)
        else:
            run_name = default_run_name
            report_to = "none"
        
        return run_name, report_to
    
    def load_model(self):
        """Load model and tokenizer with Unsloth."""
        print("\n" + "=" * 80)
        print(f"Loading model: {self.model_config['model_name']}")
        print("=" * 80)

        if torch.cuda.is_available():
            print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA devices: {torch.cuda.device_count()}")
        else:
            print(f"  - CUDA not available, using CPU")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config['model_name'],
            max_seq_length=self.model_config['max_length'],
            dtype=None,
            load_in_4bit=self.training_config.get('load_in_4bit', True),
            device_map={"": self.local_rank}
        )
        print(f"Loaded model and tokenizer (vocab size: {len(self.tokenizer)})")
        print(f"Requested max_length from config: {self.model_config['max_length']}")
        print(f"Model config max_position_embeddings: {getattr(self.model.config, 'max_position_embeddings', None)}")
        print(f"Tokenizer model_max_length: {self.tokenizer.model_max_length}")
    
    def apply_lora(self):
        """Apply LoRA adapters to the model."""
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.get('r', 16),
            target_modules=self.lora_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj", 
                "embed_tokens", "lm_head"
            ]),
            lora_alpha=self.lora_config.get('lora_alpha', 16),
            lora_dropout=self.lora_config.get('lora_dropout', 0.05),
            bias=self.lora_config.get('bias', "none"),
            use_gradient_checkpointing=self.training_config.get('gradient_checkpointing', 'unsloth'),
            random_state=42,
            use_rslora=self.lora_config.get('use_rslora', True),
            loftq_config=self.lora_config.get('loftq_config', None),
        )
        print("  - Applied LoRA adapters (PEFT) to the model.")
    
    def prepare_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        """
        Prepare training and validation datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
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
            "max_sequence_length": None,  # No truncation - we'll pack sequences
            "data_type": self.training_config.get('input_data', 'binned'),
        }
        
        print("\nLoading training data...")
        train_base_dataset = UnifiedEHRDataset(split="train", **dataset_args)
        print(f"  - Loaded {len(train_base_dataset)} training patients")
        
        print("\nLoading validation data...")
        val_base_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
        print(f"  - Loaded {len(val_base_dataset)} validation patients")

        # Extract text from datasets
        print("  - Extracting text for pretraining...")
        train_text_list = extract_text(train_base_dataset, self.tokenizer)
        val_text_list = extract_text(val_base_dataset, self.tokenizer)

        train_dataset = Dataset.from_dict({"text": train_text_list})
        val_dataset = Dataset.from_dict({"text": val_text_list})
        
        return train_dataset, val_dataset
    
    def create_trainer(
        self, 
        train_dataset: Dataset, 
        val_dataset: Optional[Dataset],
        run_name: str,
        report_to: str,
        inference_prompt: Optional[str] = None
    ):
        """
        Create the SFTTrainer.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset (can be None).
            run_name: Name for the training run.
            report_to: Where to report metrics ("wandb" or "none").
            inference_prompt: Optional prompt for inference callback.
        """
        print("\n" + "=" * 80)
        print("Setting up training...")
        print("=" * 80)
        
        training_args = TrainingArguments(
            dataloader_num_workers=self.training_config.get('dataloader_num_workers', 8),
            output_dir=self.training_config['output_dir'],
            overwrite_output_dir=self.training_config.get('overwrite_output_dir', True),
            
            # Training hyperparameters
            num_train_epochs=self.training_config['epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config.get('eval_batch_size', self.training_config['batch_size']),
            learning_rate=float(self.training_config['learning_rate']),
            weight_decay=float(self.training_config.get('weight_decay', 0.01)),
            warmup_steps=self.training_config.get('warmup_steps', 500),
            
            # Logging and evaluation
            logging_steps=self.training_config.get('logging_steps', 100),
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.training_config.get('eval_steps', 500),
            
            # Saving
            save_strategy="steps",
            save_steps=self.training_config.get('save_steps', 1000),
            save_total_limit=self.training_config.get('save_total_limit', 2),
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="loss" if val_dataset else None,
            
            # Performance
            fp16=self.training_config.get('fp16', False),
            bf16=self.training_config.get('bf16', True),
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 1),
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', False),
            
            # Reporting
            report_to=report_to,
            run_name=run_name,

            # DDP crash fix
            ddp_find_unused_parameters=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            # Other
            remove_unused_columns=False,
        )

        # Create callbacks
        callbacks = []
        if inference_prompt:
            inference_callback = InferenceCallback(self.model, self.tokenizer, inference_prompt)
            callbacks.append(inference_callback)
        
        packing_callback = PackingVerificationCallback(self.tokenizer, num_samples=3)
        callbacks.append(packing_callback)
        
        print("\nInitializing SFTTrainer...")
        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "dataset_text_field": "text",
            "max_seq_length": self.model_config['max_length'],
            "args": training_args,
            "packing": True,  # Efficient sequence packing
            "callbacks": callbacks,
        }
        
        self.trainer = SFTTrainer(**trainer_kwargs)
    
    def train(self):
        """Run the training loop."""
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")
        
        self.trainer.train()
    
    def save_model(self):
        """Save the final trained model."""
        print("\n" + "=" * 80)
        print("Saving final model...")
        print("=" * 80)
        
        final_subdir = self.training_config.get('final_subdir', "final_model")
        final_model_path = os.path.join(self.training_config['output_dir'], final_subdir)
        os.makedirs(final_model_path, exist_ok=True)
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        print(f"  - Model + tokenizer saved to: {final_model_path}")
    
    def run_full_pipeline(self, inference_prompt: Optional[str] = None):
        """
        Run the complete pretraining pipeline.
        
        Args:
            inference_prompt: Optional prompt for inference testing during training.
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
        self.create_trainer(train_dataset, val_dataset, run_name, report_to, inference_prompt)
        
        # Train
        self.train()
        
        # Save
        self.save_model()
