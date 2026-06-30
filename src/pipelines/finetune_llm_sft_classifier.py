"""
Main pipeline for SFT-based LLM classification on EHR data.

Trains a causal LLM (via LoRA + SFTTrainer) to generate "0" or "1" given a
patient EHR prompt.  Evaluation extracts next-token logits for the tokens "0"
and "1" rather than parsing generated text, giving a calibrated probability.

Usage:
    python -m src.pipelines.finetune_llm_sft_classifier \
        --config_filepath src/configs/mimic_sft_classifier.yaml \
        [--seed 42]
"""
import argparse
import logging
import os
import pprint
import random

import yaml

from src.training.training_utils import seed_all
from src.training.sft_classification_trainer import SFTClassificationTrainer

logger = logging.getLogger(__name__)


def main(config_path: str, seed: int | None = None):
    logger.info("=" * 80)
    logger.info("SFT LLM Binary Classification")
    logger.info("=" * 80)
    logger.info(f"\nLoading configuration from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration:")
    pprint.pprint(config)
    logger.info("=" * 80)

    if seed is None:
        seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
    config.setdefault("training", {})["seed"] = seed
    logger.info(f"\nUsing random seed: {seed}")
    seed_all(seed)

    # Per-seed output directory
    base_output_dir = config["training"].get("output_dir")
    if base_output_dir:
        seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        config["training"]["output_dir"] = seed_output_dir
        logger.info(f"Outputs will be written to: {seed_output_dir}")

    trainer = SFTClassificationTrainer(config)
    val_metrics, test_metrics = trainer.run_full_pipeline()

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
        help="Optional fixed seed (random if omitted)",
    )
    args = parser.parse_args()
    main(args.config_filepath, seed=args.seed)
