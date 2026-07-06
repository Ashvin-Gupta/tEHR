"""
Summarize MIMIC downstream-task label files.

For each task directory under labels_dir, reports per-split:
  - number of unique patients
  - number of datapoints (ACES label rows)
  - average / min / max trajectory length (tokens up to prediction_time)
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

SPLITS = ("train", "tuning", "held_out")


@dataclass
class SplitStats:
    split: str
    n_patients: int
    n_datapoints: int
    n_positive: int
    n_missing_trajectories: int
    avg_length: float
    min_length: int
    max_length: int


def load_subject_trajectories(data_dir: str, split: str) -> Dict[int, dict]:
    """Load tokenised trajectories for a split, keyed by subject_id."""
    split_dir = os.path.join(data_dir, split)
    pkl_files = sorted(
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".pkl")
    )

    subject_to_record: Dict[int, dict] = {}
    for file_path in tqdm(pkl_files, desc=f"Loading trajectories ({split})"):
        with open(file_path, "rb") as f:
            for record in pickle.load(f):
                subject_id = record["subject_id"]
                if subject_id not in subject_to_record:
                    subject_to_record[subject_id] = record
    return subject_to_record


def apply_prediction_time_cutoff(
    tokens: List[int],
    timestamps: List[float],
    pred_time,
) -> List[int]:
    """Mirror MIMICDataset cutoff logic for trajectory length."""
    if pred_time is None or len(timestamps) == 0:
        return tokens

    first_ts = timestamps[0]

    if isinstance(first_ts, (int, float)):
        numeric_pred_time: Optional[float] = None
        if isinstance(pred_time, (int, float)):
            numeric_pred_time = float(pred_time)
        elif isinstance(pred_time, pd.Timestamp):
            try:
                numeric_pred_time = float(pred_time.timestamp())
            except Exception:
                pass
        if numeric_pred_time is not None:
            return [t for i, t in enumerate(tokens) if timestamps[i] <= numeric_pred_time]

    elif isinstance(first_ts, pd.Timestamp):
        ts_pred_time: Optional[pd.Timestamp] = None
        if isinstance(pred_time, pd.Timestamp):
            ts_pred_time = pred_time
        elif isinstance(pred_time, (int, float)):
            try:
                ts_pred_time = pd.to_datetime(pred_time, unit="s")
            except Exception:
                pass
        if ts_pred_time is not None:
            return [t for i, t in enumerate(tokens) if timestamps[i] <= ts_pred_time]

    return tokens


def summarize_split(
    labels_path: str,
    subject_to_record: Dict[int, dict],
    split: str,
) -> SplitStats:
    labels_df = pd.read_parquet(labels_path)
    lengths: List[int] = []
    missing = 0
    n_positive = int((labels_df["label"] != 0).sum()) if "label" in labels_df.columns else 0

    for _, row in labels_df.iterrows():
        subject_id = row["subject_id"]
        record = subject_to_record.get(subject_id)
        if record is None:
            missing += 1
            continue

        tokens = apply_prediction_time_cutoff(
            list(record["tokens"]),
            list(record["timestamps"]),
            row.get("prediction_time"),
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

    return SplitStats(
        split=split,
        n_patients=labels_df["subject_id"].nunique(),
        n_datapoints=len(labels_df),
        n_positive=n_positive,
        n_missing_trajectories=missing,
        avg_length=avg_length,
        min_length=min_length,
        max_length=max_length,
    )


def discover_tasks(labels_dir: str) -> List[str]:
    return sorted(
        name
        for name in os.listdir(labels_dir)
        if os.path.isdir(os.path.join(labels_dir, name))
    )


def print_stats(task: str, stats: List[SplitStats]) -> None:
    print(f"\n{'=' * 72}")
    print(f"Task: {task}")
    print(f"{'=' * 72}")
    print(
        f"{'Split':<12} {'Patients':>10} {'Datapoints':>12} {'Positive':>10} "
        f"{'Avg Len':>10} {'Min Len':>10} {'Max Len':>10} {'Missing':>10}"
    )
    print("-" * 82)
    for s in stats:
        print(
            f"{s.split:<12} {s.n_patients:>10,} {s.n_datapoints:>12,} {s.n_positive:>10,} "
            f"{s.avg_length:>10.1f} {s.min_length:>10,} {s.max_length:>10,} "
            f"{s.n_missing_trajectories:>10,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize MIMIC label files and trajectory lengths per split."
    )
    parser.add_argument(
        "--labels-dir",
        default="/home/ag619/MIMIC_data/labels",
        help="Root directory containing one subdirectory per downstream task.",
    )
    parser.add_argument(
        "--data-dir",
        default="/home/ag619/MIMIC-IV-tokenisation",
        help="Tokenised MIMIC data directory with train/tuning/held_out splits.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task names to include (default: all subdirectories).",
    )
    args = parser.parse_args()

    tasks = args.tasks or discover_tasks(args.labels_dir)
    if not tasks:
        raise SystemExit(f"No task directories found under {args.labels_dir}")

    trajectories_by_split = {
        split: load_subject_trajectories(args.data_dir, split) for split in SPLITS
    }

    for task in tasks:
        task_dir = os.path.join(args.labels_dir, task)
        if not os.path.isdir(task_dir):
            print(f"Skipping missing task directory: {task_dir}")
            continue

        task_stats: List[SplitStats] = []
        for split in SPLITS:
            labels_path = os.path.join(task_dir, f"{split}_labels.parquet")
            if not os.path.exists(labels_path):
                print(f"Warning: missing label file {labels_path}")
                continue
            task_stats.append(
                summarize_split(labels_path, trajectories_by_split[split], split)
            )

        if task_stats:
            print_stats(task, task_stats)


if __name__ == "__main__":
    main()
