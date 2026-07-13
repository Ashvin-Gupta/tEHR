"""
Main pipeline for SFT-based LLM classification on EHR data.

Trains a causal LLM (via LoRA + SFTTrainer) to generate "0" or "1" given a
patient EHR prompt.  Evaluation extracts next-token logits for the tokens "0"
and "1" rather than parsing generated text, giving a calibrated probability.

Usage:
    sbatch src/submissions/mimic/run.sh sft mistral icu_mortality
"""
import argparse
import logging
import pprint

from src.training.training_utils import prepare_run_reproducibility
from src.training.sft_classification_trainer import SFTClassificationTrainer
from src.training.config_utils import load_config

logger = logging.getLogger(__name__)


def main(config_path: str, seed: int | None = None, checkpoint_path: str | None = None):
    logger.info("=" * 80)
    logger.info("SFT LLM Binary Classification")
    logger.info("=" * 80)
    logger.info(f"\nLoading configuration from: {config_path}")

    config = load_config(config_path)

    logger.info("Loaded configuration:")
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

    trainer = SFTClassificationTrainer(config)
    val_metrics, test_metrics = trainer.run_full_pipeline(
        resume_from_checkpoint=resume_ckpt
    )

    logger.info("\n" + "=" * 80)
    logger.info("SFT Classification completed successfully!")
    logger.info("=" * 80)
    logger.info(f"  Validation: {val_metrics}")
    logger.info(f"  Test:       {test_metrics}")

    return val_metrics, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT LLM Binary Classification")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file",
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
