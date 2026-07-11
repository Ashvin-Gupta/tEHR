"""
Main pipeline for LLM continued pretraining on EHR data.

This script orchestrates the entire pretraining workflow using the refactored modules.
"""
import argparse
import pprint
from src.training.training_utils import prepare_run_reproducibility
from src.training.pretrainer_trainer import EHRPretrainer
from src.training.config_utils import load_config

import logging
logger = logging.getLogger(__name__)


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


def main(config_path: str, seed: int | None = None, checkpoint_path: str | None = None):
    """
    Main function to run LLM continued pretraining.
    
    Args:
        config_path: Path to YAML configuration file.
        seed: Optional CLI seed override.
        checkpoint_path: Optional explicit checkpoint; else ``training.resume_from_last``.
    """
    print("=" * 80)
    print("LLM Continued Pretraining")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    
    config = load_config(config_path)

    seed, resume_ckpt = prepare_run_reproducibility(
        config,
        cli_seed=seed,
        cli_checkpoint=checkpoint_path,
        use_seed_subdir=False,  # keep stable pretrain output paths
    )
    print(f"Using seed: {seed}")
    if resume_ckpt:
        print(f"Resuming from: {resume_ckpt}")
    
    print('Loaded configuration:')
    pprint.pprint(config)
    print("=" * 80)

    pretrainer = EHRPretrainer(config)
    pretrainer.run_full_pipeline(resume_from_checkpoint=resume_ckpt)
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Continued Pretraining on EHR Data"
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
