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

from src.training.training_utils import seed_all
from src.training.classification_trainer import EHRClassificationTrainer
import logging
logger = logging.getLogger(__name__)


def main(config_path: str, seed: int | None = None, checkpoint_path: str | None = None):
    """
    Main function to run LLM classification fine-tuning.
    
    Args:
        config_path: Path to YAML configuration file.
        seed: Optional fixed seed. If not provided, a random seed is sampled.
        checkpoint_path: Optional path to a Trainer checkpoint directory to resume from.
    """
    # Load configuration
    logger.info("=" * 80)
    logger.info("LLM Binary Classification Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"\nLoading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info('Loaded configuration:')
    pprint.pprint(config)
    logger.info("=" * 80)

    # Draw a fresh random seed for this run and apply it everywhere
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
    config.setdefault("training", {})["seed"] = seed
    logger.info(f"\nUsing random seed for this run: {seed}")
    seed_all(seed)

    # Make seed-specific output directory so plots (and any artifacts) are separated
    base_output_dir = config["training"].get("output_dir")
    if base_output_dir:
        seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        config["training"]["output_dir"] = seed_output_dir
        # Length-sorted data cache is shared across all seeds (same data/tokenizer)
        config["training"]["data_cache_dir"] = base_output_dir
        logger.info(f"Outputs (including plots) will be written to: {seed_output_dir}")
        logger.info(f"Data cache (length-sorted indices) shared at: {base_output_dir}")

    # Create classification trainer and run the full pipeline
    trainer = EHRClassificationTrainer(config)
    eval_results, test_results = trainer.run_full_pipeline(resume_from_checkpoint=checkpoint_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)

    if config["training"].get("save_model", True):
        logger.info(f"\nFinal model saved to: {config['training']['output_dir']}/final_model")
    
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed. If omitted, a random seed is sampled and used to create seed_<seed>/ output dir."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a Trainer checkpoint directory (e.g., .../seed_<seed>/checkpoint-1000) to resume from."
    )
    args = parser.parse_args()
    
    main(args.config_filepath, seed=args.seed, checkpoint_path=args.checkpoint_path)
