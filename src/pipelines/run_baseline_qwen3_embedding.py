"""
Pipeline script for Qwen3-Embedding Baseline.
"""
import argparse
import yaml
import pprint

from src.training.utils import seed_all
from src.baselines.baseline_qwen3_embedding import Qwen3EmbeddingBaseline
from src.baselines.utils import load_baseline_config


def main(config_path: str):
    """
    Main function to run Qwen3-Embedding baseline.

    Args:
        config_path: Path to YAML configuration file.
    """
    seed_all(42)

    print("=" * 80)
    print("Qwen3-Embedding Baseline")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")

    config = load_baseline_config(config_path)

    print('Loaded configuration:')
    pprint.pprint(config)
    print("=" * 80)

    baseline = Qwen3EmbeddingBaseline(config)
    val_metrics, test_metrics = baseline.run()

    print("\n" + "=" * 80)
    print("Baseline completed successfully!")
    print("=" * 80)

    return val_metrics, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-Embedding Baseline"
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()

    main(args.config_filepath)
