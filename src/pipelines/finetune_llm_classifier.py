"""
Main pipeline for LLM-based classification fine-tuning on EHR data.

This script orchestrates the classification training workflow, loading a 
pretrained model from Stage 1 (continued pretraining) and fine-tuning it
with a classification head for binary cancer prediction.
"""
import argparse
import os
import pprint
import random
import yaml

from src.training.utils import seed_all
from src.training.classification_trainer import EHRClassificationTrainer


def main(config_path: str):
    """
    Main function to run LLM classification fine-tuning.
    
    Args:
        config_path: Path to YAML configuration file.
    """
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

    # Draw a fresh random seed for this run and apply it everywhere
    seed = random.SystemRandom().randint(0, 2**32 - 1)
    config.setdefault("training", {})["seed"] = seed
    print(f"\nUsing random seed for this run: {seed}")
    seed_all(seed)

    # Make seed-specific output directory so plots (and any artifacts) are separated
    base_output_dir = config["training"].get("output_dir")
    if base_output_dir:
        seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        config["training"]["output_dir"] = seed_output_dir
        print(f"Outputs (including plots) will be written to: {seed_output_dir}")

    # Create classification trainer and run the full pipeline
    trainer = EHRClassificationTrainer(config)
    eval_results, test_results = trainer.run_full_pipeline()
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)

    if config["training"].get("save_model", True):
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
