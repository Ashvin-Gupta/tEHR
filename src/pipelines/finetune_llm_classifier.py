"""
Main pipeline for LLM-based classification fine-tuning on EHR data.

This script orchestrates the classification training workflow, loading a 
pretrained model from Stage 1 (continued pretraining) and fine-tuning it
with a classification head for binary cancer prediction.
"""
import argparse
import yaml
import pprint

from src.training.utils import seed_all
from src.training.classification_trainer import EHRClassificationTrainer


def main(config_path: str):
    """
    Main function to run LLM classification fine-tuning.
    
    Args:
        config_path: Path to YAML configuration file.
    """
    # Set seed for reproducibility
    seed_all(42)
    
    # Load configuration
    print("=" * 80)
    print("LLM Binary Classification Fine-tuning")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print('Loaded configuration:')
    pprint.pprint(config)
    print("=" * 80)

    # Create classification trainer and run the full pipeline
    trainer = EHRClassificationTrainer(config)
    eval_results, test_results = trainer.run_full_pipeline()
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"\nFinal model saved to: {config['training']['output_dir']}/final_model")
    
    return eval_results, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Binary Classification Fine-tuning"
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()
    
    main(args.config_filepath)
