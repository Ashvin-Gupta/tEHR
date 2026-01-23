"""
Main pipeline for LLM continued pretraining on EHR data.

This script orchestrates the entire pretraining workflow using the refactored modules.
"""
import argparse
import yaml
import pprint

from src.training.utils import seed_all
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


def main(config_path: str):
    """
    Main function to run LLM continued pretraining.
    
    Args:
        config_path: Path to YAML configuration file.
    """
    # Set seed for reproducibility
    seed_all(42)
    
    # Load configuration
    print("=" * 80)
    print("LLM Continued Pretraining")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print('Loaded configuration:')
    pprint.pprint(config)
    print("=" * 80)

    # Create pretrainer and run the full pipeline
    pretrainer = EHRPretrainer(config)
    pretrainer.run_full_pipeline(inference_prompt=DEFAULT_INFERENCE_PROMPT)
    
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
    args = parser.parse_args()
    
    main(args.config_filepath)
