"""
Summarize length-sorted data cache files from classification experiments.

Cache files store sorted dataset indices (e.g. MIMIC_train_sorted_checkpoint-14000.pt).
This script reloads the underlying dataset via a config YAML and reports per-split:
  - number of unique patients
  - number of datapoints
  - number of positive datapoints
  - average / min / max trajectory length (raw tokens up to prediction_time)
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

# Allow running as a script from repo root or src/analysis.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.summarize_mimic_labels import (  # noqa: E402
    SPLITS,
    SplitStats,
    apply_prediction_time_cutoff,
    load_subject_trajectories,
    print_stats,
)
from src.data.dataset_factory import get_dataset  # noqa: E402

CACHE_FILE_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<split>train|tuning|held_out)_sorted_(?P<tokenizer>.+)\.pt$"
)


@dataclass
class CacheFileInfo:
    path: str
    dataset_name: str
    split: str
    tokenizer_basename: str


def resolve_cache_dir(cache_dir: Optional[str], output_dir: Optional[str]) -> str:
    if cache_dir:
        return cache_dir
    if output_dir:
        return os.path.join(output_dir, "data_cache", "length_sorted")
    raise ValueError("Provide either --cache-dir or --output-dir.")


def discover_cache_files(cache_dir: str) -> List[CacheFileInfo]:
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    cache_files: List[CacheFileInfo] = []
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".pt"):
            continue
        match = CACHE_FILE_RE.match(fname)
        if not match:
            print(f"Warning: skipping unrecognized cache filename: {fname}")
            continue
        cache_files.append(
            CacheFileInfo(
                path=os.path.join(cache_dir, fname),
                dataset_name=match.group("dataset"),
                split=match.group("split"),
                tokenizer_basename=match.group("tokenizer"),
            )
        )
    return cache_files


def load_cached_indices(cache_path: str) -> List[int]:
    try:
        indices = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        indices = torch.load(cache_path, map_location="cpu")
    return list(indices)


def _get_example(dataset, idx: int) -> dict:
    if hasattr(dataset, "examples") and dataset.examples:
        return dataset.examples[idx]
    item = dataset[idx]
    return {
        "subject_id": item.get("subject_id"),
        "label": int(item["label"].item()) if hasattr(item.get("label"), "item") else item.get("label", 0),
        "prediction_time": None,
    }


def summarize_cached_split(
    cache_info: CacheFileInfo,
    dataset,
    subject_to_record: Dict[int, dict],
    label_column: str = "label",
) -> Tuple[SplitStats, int]:
    indices = load_cached_indices(cache_info.path)
    dataset_size = len(dataset.examples) if hasattr(dataset, "examples") else len(dataset)

    if len(indices) != dataset_size:
        print(
            f"Warning: cache size ({len(indices)}) != dataset size ({dataset_size}) "
            f"for split '{cache_info.split}'"
        )

    subject_ids: List[int] = []
    labels: List[int] = []
    lengths: List[int] = []
    missing = 0

    for idx in tqdm(indices, desc=f"Summarizing {cache_info.split}"):
        example = _get_example(dataset, idx)
        subject_id = example.get("subject_id")
        if subject_id is not None:
            subject_ids.append(subject_id)

        raw_label = example.get(label_column, 0)
        try:
            label_value = int(raw_label)
        except (TypeError, ValueError):
            label_value = 1 if raw_label else 0
        labels.append(1 if label_value != 0 else 0)

        record = subject_to_record.get(subject_id)
        if record is None:
            missing += 1
            continue

        tokens = apply_prediction_time_cutoff(
            list(record["tokens"]),
            list(record["timestamps"]),
            example.get("prediction_time"),
        )
        lengths.append(len(tokens))

    if lengths:
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
    else:
        avg_length = float("nan")
        min_length = 0
        max_length = 0

    stats = SplitStats(
        split=cache_info.split,
        n_patients=len(set(subject_ids)),
        n_datapoints=len(indices),
        n_positive=sum(labels),
        n_missing_trajectories=missing,
        avg_length=avg_length,
        min_length=min_length,
        max_length=max_length,
    )
    return stats, dataset_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize length-sorted data cache files from a classification run."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to data_cache/length_sorted (e.g. .../mimic-icu-mortality-classification/data_cache/length_sorted).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Experiment output directory; uses <output-dir>/data_cache/length_sorted.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config used to build the cached dataset (e.g. src/configs/mimic_classifier.yaml).",
    )
    args = parser.parse_args()

    cache_dir = resolve_cache_dir(args.cache_dir, args.output_dir)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    dataset_name = data_config.get("dataset", "MIMIC")
    label_column = data_config.get("label_column", "label")

    cache_files = discover_cache_files(cache_dir)
    if not cache_files:
        raise SystemExit(f"No cache .pt files found in {cache_dir}")

    print(f"Cache directory: {cache_dir}")
    print(f"Config: {args.config}")
    print(f"Dataset: {dataset_name}")

    trajectories_by_split = {
        split: load_subject_trajectories(data_config["data_dir"], split) for split in SPLITS
    }

    datasets = {
        split: get_dataset(dataset_name, data_config, split) for split in SPLITS
    }

    split_stats: List[SplitStats] = []
    for cache_info in sorted(cache_files, key=lambda c: SPLITS.index(c.split)):
        dataset = datasets[cache_info.split]
        stats, dataset_size = summarize_cached_split(
            cache_info,
            dataset,
            trajectories_by_split[cache_info.split],
            label_column=label_column,
        )
        split_stats.append(stats)
        print(
            f"\n  [{cache_info.split}] cache={os.path.basename(cache_info.path)} "
            f"tokenizer={cache_info.tokenizer_basename} "
            f"indices={stats.n_datapoints:,} dataset={dataset_size:,}"
        )

    task_name = os.path.basename(os.path.normpath(args.output_dir or cache_dir))
    print_stats(task_name, split_stats)


if __name__ == "__main__":
    main()
