"""
Pipeline script to resume LLM continued pretraining from a checkpoint.

This script loads a saved checkpoint and continues training from where it left off.
"""
import argparse
import yaml
import pprint
import os

from src.training.utils import seed_all, print_sample_translations
from src.training.trainer import EHRPretrainer


# Default inference prompt for testing during training
DEFAULT_INFERENCE_PROMPT = (
    'Given the following EHR medical events, continue generating the narrative in a natural language: '
    'AGE; 65; Demographic Gender Male; Demographic Ethnicity Other; Region Region London; '
    'Antenatal screening result pending; Ultrasonography of soft tissue mass; 4mt-6mt; '
    'QAdmissions emergency admission risk calculator 3.5 %; 8mt-10mt; '
    'Seasonal influenza vaccination; 4mt-6mt; '
    'Bowel cancer screening programme: faecal occult blood result; '
    'Bowel cancer screening programme faecal occult blood test normal; 1d; '
    'Faecal occult blood: negative; 8mt-10mt; '
    'QAdmissions emergency admission risk calculator 4.2 %; 24mt-60mt;'
)


def main(config_path: str, checkpoint_path: str):
    """
    Main function to resume LLM continued pretraining from a checkpoint.
    
    Args:
        config_path: Path to YAML configuration file (same as original training).
        checkpoint_path: Path to checkpoint directory to resume from (e.g., 
                        "output_dir/checkpoints/checkpoint-5000").
    """
    # Set seed for reproducibility
    seed_all(42)
    
    # Load configuration
    print("=" * 80)
    print("Resume LLM Continued Pretraining")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    print('Loaded configuration:')
    pprint.pprint(config)
    print("=" * 80)
    
    # Print sample translations before training
    print_sample_translations(config, num_samples=3)

    # Create pretrainer
    pretrainer = EHRPretrainer(config)
    
    # Setup WandB
    run_name, report_to = pretrainer.setup_wandb()
    
    # Load model
    pretrainer.load_model()
    
    # Apply LoRA
    pretrainer.apply_lora()
    
    # Prepare datasets
    train_dataset, val_dataset = pretrainer.prepare_datasets()
    
    # Create trainer
    pretrainer.create_trainer(train_dataset, val_dataset, run_name, report_to, DEFAULT_INFERENCE_PROMPT)
    
    # Resume training from checkpoint
    print("\n" + "=" * 80)
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    pretrainer.trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # Save final model
    pretrainer.save_model()
    
    print("\n" + "=" * 80)
    print("Resumed training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resume LLM Continued Pretraining from Checkpoint"
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file (same as original training)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory to resume from (e.g., output_dir/checkpoints/checkpoint-5000)"
    )
    args = parser.parse_args()
    
    main(args.config_filepath, args.checkpoint_path)

