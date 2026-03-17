"""
Standalone script to visually inspect patient trajectories from any EHR dataset.

Usage:
    python tests/inspect_trajectories.py --config src/configs/mimic_pretrain.yaml
    python tests/inspect_trajectories.py --config src/configs/mimic_pretrain.yaml \
        --split tuning --n 5 --label 1

This is NOT a pytest test — run it directly to sanity-check translation quality.
It prints the start and end of each trajectory so you can verify that:
  - Demographic headers are correct (age, gender, race…)
  - Clinical events are translated sensibly (lab names, diagnoses…)
  - Time tokens are human-readable
  - The <HEADER_SPLIT> marker is present and in the right position
  - The sequence ends with the EOS token (or is clean text)
"""
import argparse
import random
import yaml
import sys
import os

# Ensure repo root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset_factory import get_dataset


SEPARATOR = "=" * 80
MINI_SEP  = "-" * 60


def truncate(text: str, n: int = 500) -> str:
    if len(text) <= n:
        return text
    return text[:n] + f"\n  ... [{len(text) - n} more characters] ..."


def inspect(dataset, n: int, label_filter=None, seed: int = 42):
    """Sample n trajectories and print start / end snippets."""
    rng = random.Random(seed)

    # Collect valid indices (filter by label if requested)
    indices = []
    for i in range(len(dataset)):
        item = dataset[i]
        if item is None:
            continue
        lab = item["label"].item()
        if label_filter is not None and lab != label_filter:
            continue
        indices.append(i)

    if not indices:
        print(f"\n  No samples found (label_filter={label_filter}, dataset size={len(dataset)})")
        return

    chosen = rng.sample(indices, min(n, len(indices)))

    print(f"\n{'='*80}")
    print(f"  Inspecting {len(chosen)} trajectories  "
          f"(label_filter={label_filter}, split='{dataset.split}')")
    print(f"{'='*80}")

    for rank, idx in enumerate(chosen, 1):
        item = dataset[idx]
        text = item["text"]
        label = item["label"].item()
        subject_id = item["subject_id"]

        print(f"\n{SEPARATOR}")
        print(f"  Sample {rank}/{len(chosen)}   |   idx={idx}   |   subject_id={subject_id}   |   label={label}")
        print(f"{SEPARATOR}")

        # Header / body split info
        if "<HEADER_SPLIT>" in text:
            parts = text.split("<HEADER_SPLIT>", 1)
            header_len = len(parts[0])
            body_len   = len(parts[1]) if len(parts) > 1 else 0
            print(f"  [Has HEADER_SPLIT]  header={header_len} chars  body={body_len} chars  total={len(text)} chars")
        else:
            print(f"  [No HEADER_SPLIT detected]  total={len(text)} chars")

        print(f"\n  {MINI_SEP}")
        print(f"  TEXT START (first 500 chars):")
        print(f"  {MINI_SEP}")
        print("  " + truncate(text, 500).replace("\n", "\n  "))

        print(f"\n  {MINI_SEP}")
        print(f"  TEXT END (last 500 chars):")
        print(f"  {MINI_SEP}")
        tail = text[-500:] if len(text) > 500 else text
        print("  " + tail.replace("\n", "\n  "))

        # Check EOS at end
        eos_present = text.endswith("</s>")
        print(f"\n  Ends with EOS (</s>): {eos_present}")

    print(f"\n{SEPARATOR}")
    print(f"  Done.  Printed {len(chosen)} trajectory samples.")
    print(SEPARATOR)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect patient trajectory translations from an EHR dataset."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a training YAML config (must have a 'data' section)."
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "tuning", "held_out"],
        help="Dataset split to load (default: train)."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of trajectories to print (default: 5)."
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Filter by label value (0=control, 1=case). Default: show both."
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Override the 'task' field in the data config (e.g. pretrain, prediction)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    dataset_name = data_config.get("dataset", "MIMIC")

    if args.task is not None:
        data_config["task"] = args.task

    print(f"\nLoading {dataset_name} dataset  split='{args.split}' ...")
    dataset = get_dataset(dataset_name, data_config, split=args.split)
    print(f"Loaded {len(dataset)} samples.")

    inspect(dataset, n=args.n, label_filter=args.label, seed=args.seed)


if __name__ == "__main__":
    main()