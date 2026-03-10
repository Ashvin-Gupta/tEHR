"""
Simple CLI for generating EHR trajectory continuations with a pretrained
LLM + LoRA adapters.
"""
import argparse
import pprint
import yaml

from src.inference.trajectory_generator import EHRTrajectoryGenerator
from src.data.MIMIC_unified_dataset import MIMICUnifiedDataset


def main(config_path: str, split: str, index: int):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("EHR Trajectory Generation (MIMICUnifiedDataset)")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    print("Loaded configuration:")
    pprint.pprint(config)
    print("=" * 80)

    data_cfg = config.get("data", {})
    data_dir = data_cfg["data_dir"]
    vocab_file = data_cfg["vocab_filepath"]
    aces_label_dir = data_cfg["aces_label_dir"]
    lookup_dir = data_cfg["lookup_dir"]
    task = data_cfg.get("task", "prediction")

    print(f"\nLoading MIMICUnifiedDataset split='{split}' (task={task})...")
    dataset = MIMICUnifiedDataset(
        data_dir=data_dir,
        vocab_file=vocab_file,
        aces_label_dir=aces_label_dir,
        lookup_dir=lookup_dir,
        task=task,
        split=split,
        max_sequence_length=None,
    )
    print(f"  - Dataset size: {len(dataset)}")

    if index < 0 or index >= len(dataset):
        raise ValueError(f"Index {index} out of range for split '{split}' (len={len(dataset)})")

    sample = dataset[index]
    ehr_text = sample["text"]
    subject_id = sample.get("subject_id", None)
    label = sample.get("label", None)

    generator = EHRTrajectoryGenerator(config)
    output = generator.generate(ehr_text=ehr_text)

    print("\n" + "=" * 80)
    print("Sample Info")
    print("=" * 80)
    print(f"Split: {split}")
    print(f"Index: {index}")
    if subject_id is not None:
        print(f"Subject ID: {subject_id}")
    if label is not None:
        try:
            print(f"Label: {label.item()}")
        except Exception:
            print(f"Label: {label}")

    print("\n" + "=" * 80)
    print("Input EHR Narrative")
    print("=" * 80)
    print(ehr_text)

    print("\n" + "=" * 80)
    print("Generated Trajectory")
    print("=" * 80)
    print(output)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate EHR trajectory continuations with a pretrained LLM + LoRA, "
            "using translated narratives from MIMICUnifiedDataset."
        )
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the trajectory generation config YAML file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (train / tuning / held_out). "
             "If not provided, uses data.default_split from the config.",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Patient index within the chosen split.",
    )
    args = parser.parse_args()

    main(
        args.config_filepath,
        split=args.split or "tuning",
        index=args.index,
    )

