#!/usr/bin/env python
import os
import random
import pickle
from typing import List, Dict, Any

import torch
import pandas as pd

# Adjust this import to match your actual module path
from src.data.MIMIC_unified_dataset import MIMICUnifiedDataset


# --------- CONFIG (EDIT THESE) ---------
DATA_DIR = "/home/ag619/MIMIC-IV tokensiation"         # unified MIMIC tokens dir (train/, tuning/, held_out/...)
VOCAB_FILE = "/home/ag619/MIMIC-IV tokensiation/vocab.csv"
ACES_LABEL_DIR = "/home/ag619/MIMIC_data/labels/icu_mortality"   # ACES labels (train_labels.parquet, tuning_labels.parquet, etc.)
LOOKUP_DIR = "/home/ag619/MIMIC_data/hosp"           # d_labitems.csv.gz, d_icd_diagnoses.csv.gz, d_icd_procedures.csv.gz

# Optional: original MEDS tokenised data to compare against.
# Assumes same pickle structure: list of dicts with keys including 'subject_id', 'tokens', 'timestamps'.
ORIG_DATA_DIR = "/home/ag619/MIMIC-IV tokensiation"  # e.g. "/path/to/original_meds_tokens"

SPLIT = "train"                               # or "tuning", "held_out", etc.
TASK = "prediction"
LABEL_COLUMN = "label"                        # column used as label in ACES parquet

MAX_SCAN_FOR_INCONSISTENCIES = 500
# --------------------------------------


def truncate_by_prediction_time(
    tokens: List[int],
    timestamps: List[Any],
    pred_time: Any,
) -> List[int]:
    """
    Mirror the truncation logic in MIMICUnifiedDataset.__getitem__,
    but keep timestamps so we can inspect before/after.
    """
    if pred_time is None or len(timestamps) == 0:
        return tokens, timestamps

    first_ts = timestamps[0]

    # Case 1: timestamps are numeric (int/float)
    if isinstance(first_ts, (int, float)):
        numeric_pred_time = None
        if isinstance(pred_time, (int, float)):
            numeric_pred_time = float(pred_time)
        elif isinstance(pred_time, pd.Timestamp):
            try:
                numeric_pred_time = float(pred_time.timestamp())
            except Exception:
                numeric_pred_time = None

        if numeric_pred_time is not None:
            keep_idx = [i for i, ts in enumerate(timestamps) if ts <= numeric_pred_time]
        else:
            keep_idx = list(range(len(tokens)))

    # Case 2: timestamps are pandas.Timestamp
    elif isinstance(first_ts, pd.Timestamp):
        ts_pred_time = None
        if isinstance(pred_time, pd.Timestamp):
            ts_pred_time = pred_time
        elif isinstance(pred_time, (int, float)):
            try:
                ts_pred_time = pd.to_datetime(pred_time, unit="s")
            except Exception:
                ts_pred_time = None

        if ts_pred_time is not None:
            keep_idx = [i for i, ts in enumerate(timestamps) if ts <= ts_pred_time]
        else:
            keep_idx = list(range(len(tokens)))

    else:
        # Unknown timestamp type; do not truncate
        keep_idx = list(range(len(tokens)))

    truncated_tokens = [tokens[i] for i in keep_idx]
    truncated_timestamps = [timestamps[i] for i in keep_idx]
    return truncated_tokens, truncated_timestamps


def inspect_single_example(
    ds: MIMICUnifiedDataset,
    idx: int,
    orig_records_by_subject: Dict[Any, List[Dict[str, Any]]] | None = None,
    max_tokens_to_show: int = 50,
) -> None:
    record = ds.patient_records[idx]
    subject_id = record["subject_id"]
    tokens = record["tokens"]
    timestamps = record["timestamps"]
    task_info = ds.aces_data.get(subject_id, {})

    pred_time = task_info.get("prediction_time", None)
    label_raw = task_info.get(LABEL_COLUMN, None)

    # How __getitem__ converts label
    try:
        label_value = int(label_raw) if label_raw is not None else 0
    except Exception:
        label_value = 1 if label_raw else 0

    # Truncate using same logic as dataset
    trunc_tokens, trunc_timestamps = truncate_by_prediction_time(
        tokens, timestamps, pred_time
    )

    print("=" * 80)
    print(f"Index:        {idx}")
    print(f"Subject ID:   {subject_id}")
    print(f"Prediction time: {pred_time!r} (type={type(pred_time)})")
    # Also show prediction_time in the same numeric format as token timestamps (ts)
    pred_time_ts = None
    if isinstance(pred_time, pd.Timestamp):
        try:
            pred_time_ts = float(pred_time.timestamp())
        except Exception:
            pred_time_ts = None
    elif isinstance(pred_time, (int, float)):
        pred_time_ts = float(pred_time)
    print(f"Prediction time as numeric ts: {pred_time_ts!r}")
    print(f"#tokens full (unified): {len(tokens)}   #tokens truncated (unified): {len(trunc_tokens)}")
    if timestamps:
        print(f"First ts:     {timestamps[0]!r} (type={type(timestamps[0])})")
        print(f"Last ts full: {timestamps[-1]!r}")
    if trunc_timestamps:
        print(f"Last ts trunc:{trunc_timestamps[-1]!r}")

    print(f"Raw label value from ACES[{LABEL_COLUMN!r}]: {label_raw!r} -> label: {label_value}")
    print(
        f"MEDS_DEATH:   {task_info.get('MEDS_DEATH', None)!r}, "
        f"HOSPITAL_DISCHARGE//DIED: {task_info.get('HOSPITAL_DISCHARGE//DIED', None)!r}"
    )

    # Basic consistency checks
    if label_value == 1:
        has_death = bool(task_info.get("MEDS_DEATH", 0))
        has_icu = bool(task_info.get("HOSPITAL_DISCHARGE//DIED", 0))
        if not (has_death or has_icu):
            print(">>> WARNING: Positive label but neither MEDS_DEATH nor HOSPITAL_DISCHARGE//DIED is True/1")

    if pred_time is not None and trunc_timestamps:
        # Verify max timestamp in truncated sequence does not exceed prediction time
        try:
            max_trunc_ts = max(trunc_timestamps)
            if max_trunc_ts > pred_time:
                print(">>> ERROR: Found timestamp AFTER prediction_time in truncated sequence!")
        except TypeError:
            print(">>> NOTE: Could not compare timestamps to prediction_time due to type mismatch.")

    # Map IDs to token strings for full unified record
    unified_full_codes = [ds.id_to_token_map.get(tid, "") for tid in tokens]
    unified_trunc_codes = [ds.id_to_token_map.get(tid, "") for tid in trunc_tokens]

    print("-" * 80)
    print("MIMIC unified tokens (FULL) - start/end")
    print("Start:", [repr(c) for c in unified_full_codes[:max_tokens_to_show]])
    print("End:  ", [repr(c) for c in unified_full_codes[-max_tokens_to_show:]]) if unified_full_codes else None

    print("-" * 80)
    print("MIMIC unified tokens (TRUNCATED at prediction_time) - start/end")
    print("Start:", [repr(c) for c in unified_trunc_codes[:max_tokens_to_show]])
    print("End:  ", [repr(c) for c in unified_trunc_codes[-max_tokens_to_show:]]) if unified_trunc_codes else None

    # Check for potential label leakage in truncated unified tokens
    bad_keywords = ["MEDS_DEATH", "HOSPITAL_DISCHARGE//DIED", "DIED"]
    leak = [c for c in unified_trunc_codes if any(k in str(c) for k in bad_keywords)]
    if leak:
        print(">>> WARNING: Found potential label leakage keywords in truncated unified tokens:")
        for c in leak[:20]:
            print("   ", repr(c))

    # Alignment check: prediction_time vs ICU_ADMISSION token in unified record
    icu_indices = [i for i, c in enumerate(unified_full_codes) if "ICU_ADMISSION" in str(c)]
    print("-" * 80)
    if icu_indices:
        print("ICU_ADMISSION alignment (unified record):")
        for i in icu_indices[:10]:
            ts = timestamps[i] if i < len(timestamps) else None
            print(f"  idx {i}: token={repr(unified_full_codes[i])}, ts={ts!r}")
            if pred_time is not None and ts is not None:
                try:
                    print(f"    ts == prediction_time? {ts == pred_time}")
                except Exception:
                    pass
        exact_matches = [
            i for i in icu_indices
            if i < len(timestamps) and pred_time is not None and timestamps[i] == pred_time
        ]
        if exact_matches:
            print(f"Prediction time EXACTLY matches ICU_ADMISSION timestamp at indices: {exact_matches}")
        else:
            print("No ICU_ADMISSION timestamp exactly equal to prediction_time in unified record.")
    else:
        print("No ICU_ADMISSION token found in unified record.")

    # Compare against original MEDS data if available
    print("-" * 80)
    if orig_records_by_subject is not None:
        orig_list = orig_records_by_subject.get(subject_id)
        if not orig_list:
            print("No original MEDS record found for this subject in ORIG_DATA_DIR.")
        else:
            if len(orig_list) > 1:
                print(f"Multiple original MEDS records found for subject_id={subject_id}; inspecting first.")
            orig_rec = orig_list[0]
            orig_tokens = orig_rec.get("tokens", [])
            orig_timestamps = orig_rec.get("timestamps", [])
            orig_codes = [ds.id_to_token_map.get(tid, "") for tid in orig_tokens]

            print("Original MEDS tokens (FULL) - start/end")
            print(f"#tokens full (original MEDS): {len(orig_tokens)}")
            if orig_tokens:
                print("Start:", [repr(c) for c in orig_codes[:max_tokens_to_show]])
                print("End:  ", [repr(c) for c in orig_codes[-max_tokens_to_show:]])
                print(f"First ts (orig): {orig_timestamps[0]!r}")
                print(f"Last ts (orig):  {orig_timestamps[-1]!r}")
            else:
                print("Original MEDS record has no tokens.")
    else:
        print("Original MEDS data mapping not provided; skipping comparison.")


def scan_dataset_for_inconsistencies(ds: MIMICUnifiedDataset, max_samples: int = 500):
    n = len(ds)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[: min(max_samples, n)]

    num_pos = 0
    num_neg = 0
    label_mismatch = 0
    time_violations = 0

    for idx in indices:
        record = ds.patient_records[idx]
        subject_id = record["subject_id"]
        tokens = record["tokens"]
        timestamps = record["timestamps"]
        task_info = ds.aces_data.get(subject_id, {})

        pred_time = task_info.get("prediction_time", None)
        label_raw = task_info.get(LABEL_COLUMN, None)

        # same label logic as __getitem__
        try:
            label_value = int(label_raw) if label_raw is not None else 0
        except Exception:
            label_value = 1 if label_raw else 0

        num_pos += int(label_value == 1)
        num_neg += int(label_value == 0)

        # Check positive label consistency with MEDS_DEATH / ICU_MORTALITY
        if label_value == 1:
            has_death = bool(task_info.get("MEDS_DEATH", 0))
            has_icu = bool(task_info.get("HOSPITAL_DISCHARGE//DIED", 0))
            if not (has_death or has_icu):
                label_mismatch += 1

        # Check time truncation
        trunc_tokens, trunc_timestamps = truncate_by_prediction_time(
            tokens, timestamps, pred_time
        )
        if pred_time is not None and trunc_timestamps:
            try:
                if max(trunc_timestamps) > pred_time:
                    time_violations += 1
            except TypeError:
                # Type mismatch – just note it
                pass

    print("=" * 80)
    print(f"Scanned {len(indices)} examples")
    print(f"Positives: {num_pos}  Negatives: {num_neg}")
    print(f"Positives without MEDS_DEATH/HOSPITAL_DISCHARGE//DIED: {label_mismatch}")
    print(f"Time truncation violations (ts > prediction_time): {time_violations}")
    print("=" * 80)


def load_original_meds_records(data_dir: str, split: str) -> List[Dict[str, Any]]:
    """
    Load original MEDS records from pickled files, if available.
    Expects same structure as unified dataset records.
    """
    if not data_dir:
        print("ORIG_DATA_DIR is empty; skipping load of original MEDS data.")
        return []

    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        print(f"Original MEDS data directory not found for split '{split}': {split_dir}")
        return []

    records: List[Dict[str, Any]] = []
    pkl_files = [f for f in os.listdir(split_dir) if f.endswith(".pkl")]
    if not pkl_files:
        print(f"No .pkl files found in original MEDS split directory: {split_dir}")
        return []

    for fname in pkl_files:
        fpath = os.path.join(split_dir, fname)
        try:
            with open(fpath, "rb") as f_in:
                loaded = pickle.load(f_in)
                if isinstance(loaded, list):
                    records.extend(loaded)
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")

    print(f"Loaded {len(records)} original MEDS records from {split_dir}")
    return records


def main():
    print("Loading MIMICUnifiedDataset with prediction task...")
    ds = MIMICUnifiedDataset(
        data_dir=DATA_DIR,
        vocab_file=VOCAB_FILE,
        aces_label_dir=ACES_LABEL_DIR,
        lookup_dir=LOOKUP_DIR,
        task=TASK,
        split=SPLIT,
        max_sequence_length=512,     # not used in current class, but keep for consistency
        label_column=LABEL_COLUMN,
    )
    print(f"Loaded unified MIMIC dataset split='{SPLIT}' with {len(ds)} records")

    # Load original MEDS data (optional)
    orig_records_by_subject: Dict[Any, List[Dict[str, Any]]] = {}
    orig_records = load_original_meds_records(ORIG_DATA_DIR, SPLIT)
    for rec in orig_records:
        sid = rec.get("subject_id")
        if sid is not None:
            orig_records_by_subject.setdefault(sid, []).append(rec)

    # Find three positive (case) and three negative (control) examples
    pos_indices = []
    neg_indices = []
    for idx, record in enumerate(ds.patient_records):
        subject_id = record["subject_id"]
        task_info = ds.aces_data.get(subject_id, {})
        label_raw = task_info.get(LABEL_COLUMN, None)
        try:
            label_value = int(label_raw) if label_raw is not None else 0
        except Exception:
            label_value = 1 if label_raw else 0

        if label_value == 1 and len(pos_indices) < 5:
            pos_indices.append(idx)
        elif label_value == 0 and len(neg_indices) < 1:
            neg_indices.append(idx)

        if len(pos_indices) >= 5 and len(neg_indices) >= 1:
            break

    print("\n>>> DETAILED INSPECTION OF INDIVIDUAL EXAMPLES (three cases and controls)")
    if pos_indices:
        for case_num, pos_idx in enumerate(pos_indices, 1):
            print(f"\n--- CASE {case_num} (label=1) ---")
            inspect_single_example(ds, pos_idx, orig_records_by_subject=orig_records_by_subject)
    else:
        print("No positive (case) examples found in this split.")

    if neg_indices:
        for ctrl_num, neg_idx in enumerate(neg_indices, 1):
            print(f"\n--- CONTROL {ctrl_num} (label=0) ---")
            inspect_single_example(ds, neg_idx, orig_records_by_subject=orig_records_by_subject)
    else:
        print("No negative (control) examples found in this split.")

    print("\n>>> SCANNING DATASET FOR GLOBAL INCONSISTENCIES")
    scan_dataset_for_inconsistencies(ds, max_samples=MAX_SCAN_FOR_INCONSISTENCIES)


if __name__ == "__main__":
    main()