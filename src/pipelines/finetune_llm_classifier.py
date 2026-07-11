"""
Main pipeline for LLM-based classification fine-tuning on EHR data.

This script orchestrates the classification training workflow, loading a 
pretrained model from Stage 1 (continued pretraining) and fine-tuning it
with a classification head for binary cancer prediction.
"""
import argparse
import pprint

from src.training.training_utils import prepare_run_reproducibility
from src.training.classification_trainer import EHRClassificationTrainer
from src.training.config_utils import load_config
import logging
logger = logging.getLogger(__name__)


def main(config_path: str, seed: int | None = None, checkpoint_path: str | None = None):
    """
    Main function to run LLM classification fine-tuning.
    
    Args:
        config_path: Path to YAML configuration file.
        seed: Optional CLI seed override (else config ``seed`` / ``training.seed``).
        checkpoint_path: Optional explicit checkpoint; else ``training.resume_from_last``.
    """
    logger.info("=" * 80)
    logger.info("LLM Binary Classification Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"\nLoading configuration from: {config_path}")
    
    config = load_config(config_path)
    
    logger.info('Loaded configuration:')
    pprint.pprint(config)
    logger.info("=" * 80)

    seed, resume_ckpt = prepare_run_reproducibility(
        config,
        cli_seed=seed,
        cli_checkpoint=checkpoint_path,
        use_seed_subdir=True,
    )
    logger.info(f"\nUsing seed: {seed}")
    if resume_ckpt:
        logger.info(f"Resuming from: {resume_ckpt}")

    trainer = EHRClassificationTrainer(config)
    eval_results, test_results = trainer.run_full_pipeline(resume_from_checkpoint=resume_ckpt)
    
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
        help="Override config seed (default: config seed, usually 42)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit Trainer checkpoint to resume from (overrides resume_from_last)",
    )
    args = parser.parse_args()
    
    main(args.config_filepath, seed=args.seed, checkpoint_path=args.checkpoint_path)
