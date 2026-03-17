"""
EHR Pretrainer class for managing continued pretraining workflow.
"""
import os
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import time
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import wandb

from src.training.training_utils import extract_and_pack
from src.data.dataset_factory import get_dataset

import logging
logger = logging.getLogger(__name__)
    
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
            # f"{model_name}-pretrain-{self.data_config['cutoff_months']}month-cutoff"
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
        logger.info("\n" + "=" * 80)
        logger.info(f"Loading model: {self.model_config['model_name']}")
        logger.info("=" * 80)

        if torch.cuda.is_available():
            logger.info(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  - CUDA devices: {torch.cuda.device_count()}")
        else:
            logger.info(f"  - CUDA not available, using CPU")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config['model_name'],
            max_seq_length=self.model_config['max_length'],
            dtype=None,
            load_in_4bit=self.training_config.get('load_in_4bit', True),
            device_map={"": self.local_rank}
        )
    
    
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
        logger.info("  - Applied LoRA adapters (PEFT) to the model.")
    
    def prepare_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        """
        Prepare training and validation datasets using a Rank-0-only processing pattern.
        """
        # 1. Define paths
        cache_dir = os.path.join(self.training_config['output_dir'], "data_cache")
        train_cache_path = os.path.join(cache_dir, "train_packed")
        val_cache_path = os.path.join(cache_dir, "val_packed")

        if not torch.distributed.is_initialized():
            # Fallback for some environments: ensure we know who we are
            if "WORLD_SIZE" in os.environ:
                torch.distributed.init_process_group(backend="nccl")

        # 2. Only Rank 0 performs the heavy lifting
        if self.local_rank == 0:
            try:
                if not os.path.exists(os.path.join(train_cache_path, "dataset_info.json")):
                    os.makedirs(cache_dir, exist_ok=True)
                    logger.info("\n--- Data Preparation (Rank 0 Only) ---")

                    # Get datasets that returns dict of text,label and subject id
                    train_base = get_dataset(self.data_config['dataset'], self.data_config, "train")
                    val_base = get_dataset(self.data_config['dataset'], self.data_config, "tuning")

                    # Process Training Data
                    extract_and_pack(train_base, self.tokenizer, self.model_config['max_length'], train_cache_path)

                    # Process Validation Data
                    extract_and_pack(val_base, self.tokenizer, self.model_config['max_length'], val_cache_path)
                    
                else:
                    logger.info("\n--- Found existing cached dataset, skipping processing ---")
            
            except Exception as e:
                logger.error(f"Rank {self.local_rank} error: {e}")
                raise

        
        max_idle = 600  # max seconds Rank 0 can be idle before giving up
        heartbeat_file = os.path.join(train_cache_path, "heartbeat_rank0.txt")
        dataset_file = os.path.join(train_cache_path, "dataset_info.json")

        while not os.path.exists(dataset_file):
            if os.path.exists(heartbeat_file):
                # read last heartbeat
                last_time = float(open(heartbeat_file).read())
                idle_time = time.time() - last_time
                if idle_time > max_idle:
                    raise TimeoutError(f"Rank {self.local_rank} timed out: Rank 0 idle for {idle_time:.1f}s")
            else:
                # no heartbeat yet, could be just starting
                idle_time = 0

            logger.info(f"Rank {self.local_rank} waiting... last heartbeat {idle_time:.1f}s ago")
            time.sleep(5)

        # 4. All ranks load from the disk cache
        # .load_from_disk() uses memory mapping, so it's nearly instant and RAM-efficient
        train_dataset = Dataset.load_from_disk(train_cache_path)
        
        val_dataset = None
        if os.path.exists(val_cache_path):
            val_dataset = Dataset.load_from_disk(val_cache_path)

        return train_dataset, val_dataset
    
    def create_trainer(
        self, 
        train_dataset: Dataset, 
        val_dataset: Optional[Dataset],
        run_name: str,
        report_to: str
    ):
        """
        Create the SFTTrainer using the manually packed datasets.
        """
        logger.info("\n" + "=" * 80)
        logger.info("Setting up training...")
        logger.info("=" * 80)
        
        # Define the SFT configuration with all your detailed parameters
        training_args = SFTConfig(
            output_dir=self.training_config['output_dir'],
            overwrite_output_dir=self.training_config.get('overwrite_output_dir', True),
            
            packing=False, # Have manually packed the data ourselves
            max_seq_length=self.model_config['max_length'],
            dataset_num_proc=4,

            # Training Hyperparameters
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
            
            # System
            report_to=report_to,
            run_name=run_name,
            ddp_find_unused_parameters=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            
            # Other
            remove_unused_columns=False,
            # Since we use a standard Dataset now, num_workers is safe to use
            dataloader_num_workers=self.training_config.get('dataloader_num_workers', 8),
            dataloader_drop_last=True,
        )

        
        logger.info("\nInitializing SFTTrainer...")

        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        trainer_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "train_dataset": train_dataset, # This is already packed!
            "eval_dataset": val_dataset,
            "args": training_args,
            "data_collator": collator,
        }
        
        self.trainer = SFTTrainer(**trainer_kwargs)
        
        logger.info(f"  ✓ SFTTrainer initialized")
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                world_size = torch.distributed.get_world_size()
            except Exception:
                world_size = 1

        per_device_bs = self.training_config['batch_size']
        grad_acc = self.training_config.get('gradient_accumulation_steps', 1)
        effective_global_bs = per_device_bs * grad_acc * world_size

    
    def train(self):
        """Run the training loop."""
        logger.info("\n" + "=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80 + "\n")
        
        self.trainer.train()
    
    def save_model(self):
        """Save the final trained model."""
        logger.info("\n" + "=" * 80)
        logger.info("Saving final model...")
        logger.info("=" * 80)
        
        final_subdir = self.training_config.get('final_subdir', "final_model")
        final_model_path = os.path.join(self.training_config['output_dir'], final_subdir)
        os.makedirs(final_model_path, exist_ok=True)
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"  - Model + tokenizer saved to: {final_model_path}")
    
    def run_full_pipeline(self):
        """
        Run the complete pretraining pipeline.
        
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
        self.create_trainer(train_dataset, val_dataset, run_name, report_to)
        
        # Train
        self.train()
        
        # Save
        self.save_model()
