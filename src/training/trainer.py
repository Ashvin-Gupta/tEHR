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
from trl import SFTTrainer, SFTConfig
import wandb

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.preprocessing import extract_text
from src.training.callbacks import InferenceCallback, PackingVerificationCallback, BatchShapeCallback


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
        # Explicitly set the model_max_length and padding_side for the tokenizer
        self.tokenizer.model_max_length = self.model_config['max_length']  
        self.tokenizer.padding_side = "right" # SFTTrainer usually prefers right padding for packing
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
        
        training_args = SFTConfig(
            output_dir=self.training_config['output_dir'],
            overwrite_output_dir=self.training_config.get('overwrite_output_dir', True),

            # --- CRITICAL: Pass SFT params here ---
            max_seq_length=self.model_config['max_length'],
            packing=True,
            dataset_text_field="text",

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
            dataloader_num_workers=self.training_config.get('dataloader_num_workers', 8),
        )

        # Create callbacks
        callbacks = []
        if inference_prompt:
            inference_callback = InferenceCallback(self.model, self.tokenizer, inference_prompt)
            callbacks.append(inference_callback)
        
        # Add batch shape callback (prints every 25 steps, set to 0 for only first batch)
        # batch_callback = BatchShapeCallback(print_every_n_steps=25)
        # callbacks.append(batch_callback)

        packing_callback = PackingVerificationCallback(self.tokenizer, num_samples=3)
        callbacks.append(packing_callback)
        
        print("\nInitializing SFTTrainer...")
       

        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "args": training_args,
            "callbacks": callbacks,
            
            # CRITICAL: Pass these directly to ensure SFTTrainer picks them up
            "packing": True,
            "max_seq_length": self.model_config['max_length'],
            "dataset_text_field": "text",
        }
        
        
        self.trainer = SFTTrainer(**trainer_kwargs)
        train_ds = self.trainer.train_dataset
        print(f"  - Internal train dataset type: {type(train_ds)}")

        print(f"  ✓ SFTTrainer initialized")
        print(f"  - Configured max_seq_length: {self.model_config['max_length']}")
        print(f"  - Configured batch_size: {self.training_config['batch_size']}")

        # Debug: Print actual batch shapes from the dataloader
        print("\n" + "=" * 80)
        print("DEBUGGING: Checking actual batch shapes from dataloader...")
        print("=" * 80)
        try:
            train_dataloader = self.trainer.get_train_dataloader()
            print(f"  - Dataloader type: {type(train_dataloader)}")
            
            # Get first batch
            first_batch = next(iter(train_dataloader))
            print(f"  - First batch keys: {first_batch.keys()}")
            
            if 'input_ids' in first_batch:
                input_ids = first_batch['input_ids']
                print(f"  - input_ids shape: {input_ids.shape}")
                print(f"  - input_ids dtype: {input_ids.dtype}")
                
                # Check sequence lengths
                if 'attention_mask' in first_batch:
                    attention_mask = first_batch['attention_mask']
                    actual_lengths = attention_mask.sum(dim=1)
                    print(f"  - attention_mask shape: {attention_mask.shape}")
                    print(f"  - Actual sequence lengths: min={actual_lengths.min().item()}, max={actual_lengths.max().item()}, mean={actual_lengths.float().mean().item():.1f}")
                
                # Show a sample of the first sequence
                if input_ids.size(0) > 0:
                    first_seq = input_ids[0]
                    print(f"  - First sequence length: {len(first_seq)}")
                    print(f"  - First 10 token IDs: {first_seq[:10].tolist()}")
                    print(f"  - Last 10 token IDs: {first_seq[-10:].tolist()}")
            
            # Get a few more batches to see consistency
            print("\n  Checking next 3 batches...")
            for i, batch in enumerate(train_dataloader):
                if i >= 3:
                    break
                if 'input_ids' in batch:
                    print(f"    Batch {i+1}: shape={batch['input_ids'].shape}")
            
        except Exception as e:
            print(f"  ⚠️  Error accessing dataloader: {e}")
            import traceback
            traceback.print_exc()

        print("=" * 80 + "\n")
    
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
