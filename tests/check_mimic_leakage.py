#!/usr/bin/env python
import os
import sys
import pickle
import torch
import pandas as pd
import polars as pl
import logging

from src.data.mimic_dataset import MIMICDataset

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
CACHE_DIR = "/home/ag619/experiments/Results/mimic-ICU-mortality-classification-mistral-4000/data_cache/length_sorted"
LABEL_DIR = "/home/ag619/MIMIC_data/labels/icu_mortality"
TOKENIZED_DATA_DIR = "/home/ag619/MIMIC-IV tokensiation"
VOCAB_FILE = "/home/ag619/MIMIC-IV tokensiation/vocab.csv"
LOOKUP_DIR = "/home/ag619/MIMIC_data/hosp"

# Map split name -> expected cache filename suffix and label csv name
SPLITS = {
    "train":   {"cache_suffix": "_train_sorted_",   "label_file": "train.csv"},
    "tuning":  {"cache_suffix": "_tuning_sorted_",  "label_file": "tuning.csv"},
    "held_out": {"cache_suffix": "_held_out_sorted_", "label_file": "held_out.csv"},
}

SUBJECT_COL = "subject_id"   # change if your column is named differently

# --------------------------------------------------------------------
# LOGGING (stdout + dedicated file)
# --------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
LOG_PATH = os.path.join("logs", "check_mimic_leakage.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a"),
    ],
)
logger = logging.getLogger("check_mimic_leakage")

logger.info("Starting leakage check")
logger.info("  CACHE_DIR=%s", CACHE_DIR)
logger.info("  TOKENIZED_DATA_DIR=%s", TOKENIZED_DATA_DIR)
logger.info("  LABEL_DIR=%s", LABEL_DIR)
logger.info("  LOG_PATH=%s", LOG_PATH)
MAX_PKL_FILES = int(os.environ.get("MAX_PKL_FILES", "0"))
if MAX_PKL_FILES > 0:
    logger.info("  MAX_PKL_FILES=%d (tokenized-scan will be limited)", MAX_PKL_FILES)

# For the token sanity check:
# - By default (0), we check all label==1 datapoints (can take a while).
# - Set a positive value to cap the number of label==1 datapoints checked.
MAX_LABEL1_CHECKS = int(os.environ.get("MAX_LABEL1_CHECKS", "0"))
MAX_LABEL1_MISMATCHES_LOG = int(os.environ.get("MAX_LABEL1_MISMATCHES_LOG", "5"))

# --------------------------------------------------------------------
# HELPER: find cache file for split
# --------------------------------------------------------------------
def find_cache_file_for_split(split_name: str, suffix: str) -> str:
    """
    Cache filenames are like: MIMIC_{split}_sorted_{tokenizer_id_basename}.pt
    We match using split + substring suffix.
    """
    if not os.path.exists(CACHE_DIR):
        raise FileNotFoundError(f"CACHE_DIR does not exist: {CACHE_DIR}")

    candidates = []
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(".pt") and split_name in fname and suffix in fname:
            candidates.append(fname)

    if not candidates:
        raise FileNotFoundError(
            f"No cache file found for split={split_name} suffix={suffix} in {CACHE_DIR}"
        )
    if len(candidates) > 1:
        logger.warning(
            "Multiple cache candidates found for split=%s suffix=%s; using the first one: %s",
            split_name,
            suffix,
            candidates[0],
        )

    return os.path.join(CACHE_DIR, candidates[0])

def load_subjects_from_tokenized_split(split: str) -> set:
    """
    Count unique subject_ids from the original tokenized pkl files.
    This avoids relying on HF `load_from_disk` (your tokenization is custom).
    """
    split_dir = os.path.join(TOKENIZED_DATA_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Tokenized split dir not found: {split_dir}")

    subject_ids = set()
    pkl_files = [f for f in os.listdir(split_dir) if f.endswith(".pkl")]
    pkl_files = pkl_files[:MAX_PKL_FILES] if MAX_PKL_FILES > 0 else pkl_files
    logger.info("  Tokenized split '%s': %d pkl files", split, len(pkl_files))

    for fname in pkl_files:
        fpath = os.path.join(split_dir, fname)
        with open(fpath, "rb") as f:
            records = pickle.load(f)
        for rec in records:
            sid = rec.get(SUBJECT_COL)
            if sid is not None:
                subject_ids.add(sid)

    return subject_ids

# --------------------------------------------------------------------
# MAIN CHECKS
# --------------------------------------------------------------------
split_subjects_from_indices = {}  # split -> set of subject_ids from length-sorted indices
split_subjects_from_original = {}  # split -> set from tokenized pkl files

logger.info("Loaded configuration; starting per-split checks...")

for split, meta in SPLITS.items():
    logger.info("\n" + "=" * 80)
    logger.info("Processing split: %s", split)

    # 1) Load ACES labels for the split (parquet)
    # Your MIMICDataset expects: {split}_labels.parquet
    # Example: train_labels.parquet, tuning_labels.parquet, held_out_labels.parquet
    aces_path = os.path.join(LABEL_DIR, f"{split}_labels.parquet")
    logger.info("  Loading ACES labels from: %s", aces_path)
    if not os.path.exists(aces_path):
        raise FileNotFoundError(f"ACES label parquet not found: {aces_path}")

    aces_df = pl.read_parquet(aces_path).to_pandas()
    aces_rows = int(len(aces_df))
    if SUBJECT_COL not in aces_df.columns:
        raise ValueError(f"ACES parquet missing '{SUBJECT_COL}' column. Columns={list(aces_df.columns)}")

    split_label_subjects = set(aces_df[SUBJECT_COL].tolist())
    logger.info("  Subjects in ACES labels (%s): %d", split, len(split_label_subjects))
    logger.info("  ACES label rows (%s): %d", split, aces_rows)

    # 2) Load the original tokenized split subject IDs
    #    (this is for "no leakage" sanity and for counting coverage)
    split_subjects_from_original[split] = load_subjects_from_tokenized_split(split)
    logger.info("  Unique subjects in original tokenized split (%s): %d", split, len(split_subjects_from_original[split]))

    # 3) Load sorted indices cache
    cache_path = find_cache_file_for_split(split, meta["cache_suffix"])
    logger.info("  Loading cached indices from: %s", cache_path)
    try:
        indices = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        indices = torch.load(cache_path, map_location="cpu")

    # torch.load may return a tensor or a plain Python list depending on how it was saved.
    indices = indices.tolist() if hasattr(indices, "tolist") else indices
    logger.info("  Number of indices in cached subset (%s): %d", split, len(indices))

    # 4) Re-create the same dataset used to generate the cache,
    #    then map cached indices -> subject_id via dataset.examples[idx].
    #
    #    Important: this mirrors MIMICDataset internals (including any split-specific quirks),
    #    so indices->subject mapping aligns with the cached ordering.
    logger.info("  Re-creating MIMICDataset split=%s task=prediction", split)
    ds = MIMICDataset(
        data_dir=TOKENIZED_DATA_DIR,
        vocab_file=VOCAB_FILE,
        aces_label_dir=LABEL_DIR,
        lookup_dir=LOOKUP_DIR,
        task="prediction",
        split=split,
    )
    logger.info("  Re-created dataset length (%s): %d examples", split, len(ds))
    if not hasattr(ds, "examples") or not ds.examples:
        logger.warning("  Dataset split=%s has 0 examples; subject mapping will be empty.", split)
    if len(indices) != len(ds):
        logger.warning(
            "  [CACHE-SIZE MISMATCH] split=%s cached_indices=%d but current_dataset_len=%d. Cached subset may be stale.",
            split,
            len(indices),
            len(ds),
        )

    # 3b) Token sanity check for label==1 datapoints:
    # Every label-1 example should have a terminal discharge/death marker token
    # in its underlying tokenized trajectory.
    required_token_strings = {"HOSPITAL_DISCHARGE//DIED", "MEDS_DEATH"}
    required_token_ids = {
        int(token_id)
        for token_id, tok in getattr(ds, "id_to_token", {}).items()
        if tok in required_token_strings
    }
    logger.info(
        "  [LABEL1-TOKEN-CHECK] split=%s required_token_ids_found=%d (required=%s)",
        split,
        len(required_token_ids),
        sorted(required_token_strings),
    )

    if not required_token_ids:
        logger.warning(
            "  [LABEL1-TOKEN-CHECK] No required token strings found in vocab for split=%s. "
            "expected=%s",
            split,
            sorted(required_token_strings),
        )

    subject_token_presence_cache = {}  # subject_id -> bool(has_required)
    label1_checked = 0
    label1_missing = 0
    label1_missing_examples_logged = 0

    for ex_idx, ex in enumerate(getattr(ds, "examples", []) or []):
        raw_label = ex.get(getattr(ds, "label_column", "label"))
        try:
            label_value = int(raw_label)
        except Exception:
            label_value = 1 if raw_label else 0

        if label_value != 1:
            continue

        # Optional cap to keep debugging runs quick
        if MAX_LABEL1_CHECKS > 0 and label1_checked >= MAX_LABEL1_CHECKS:
            logger.warning(
                "  [LABEL1-TOKEN-CHECK] Reached MAX_LABEL1_CHECKS=%d for split=%s; stopping early.",
                MAX_LABEL1_CHECKS,
                split,
            )
            break

        subject_id = ex.get("subject_id")
        if subject_id is None:
            # If subject_id missing, we can't verify token stream.
            label1_checked += 1
            label1_missing += 1
            if label1_missing_examples_logged < MAX_LABEL1_MISMATCHES_LOG:
                logger.warning(
                    "  [LABEL1-TOKEN-CHECK] split=%s example_index=%d missing subject_id for label==1",
                    split,
                    ex_idx,
                )
                label1_missing_examples_logged += 1
            continue

        # Compute once per subject (same subject can appear in multiple ACES rows)
        if subject_id not in subject_token_presence_cache:
            records = ds.subject_to_records.get(subject_id) if hasattr(ds, "subject_to_records") else None
            if not records:
                subject_token_presence_cache[subject_id] = False
            else:
                # Mirror MIMICDataset.__getitem__ selection logic for prediction task
                token_ids = list(records[0].get("tokens", []))
                subject_token_presence_cache[subject_id] = any(
                    (tid in required_token_ids) for tid in token_ids
                )

        has_required = subject_token_presence_cache[subject_id]
        label1_checked += 1

        if not has_required:
            label1_missing += 1
            if label1_missing_examples_logged < MAX_LABEL1_MISMATCHES_LOG:
                # Log a tail sample so you can manually confirm the issue.
                # (This converts token ids -> token strings, but only for a few examples.)
                records = ds.subject_to_records.get(subject_id) if hasattr(ds, "subject_to_records") else None
                token_ids = list(records[0].get("tokens", [])) if records else []
                tail_ids = token_ids[-25:]
                tail_tokens = [ds.id_to_token.get(tid, "") for tid in tail_ids]
                logger.warning(
                    "  [LABEL1-TOKEN-CHECK] split=%s label1_example_index=%d subject_id=%s missing_required_token. "
                    "tail_last25_tokens=%s",
                    split,
                    ex_idx,
                    str(subject_id),
                    tail_tokens,
                )
                label1_missing_examples_logged += 1

    logger.info(
        "  [LABEL1-TOKEN-CHECK] split=%s label1_checked=%d label1_missing_required_token=%d",
        split,
        label1_checked,
        label1_missing,
    )

    # Map indices -> subject_ids
    subjects_from_indices = set()
    for idx in indices:
        ex = None
        if hasattr(ds, "examples") and ds.examples:
            i = int(idx)
            if i < 0 or i >= len(ds.examples):
                logger.warning(
                    "  Cached index out of range for split=%s: idx=%d, len(examples)=%d",
                    split,
                    i,
                    len(ds.examples),
                )
                continue
            ex = ds.examples[i]

        if ex is None:
            continue

        sid = ex.get(SUBJECT_COL)
        if sid is not None:
            subjects_from_indices.add(sid)

    split_subjects_from_indices[split] = subjects_from_indices
    logger.info("  Unique subjects from cached indices (%s): %d", split, len(subjects_from_indices))

    # 5) Compare cached-subset subjects vs ACES label subjects
    overlap_with_labels = subjects_from_indices & split_label_subjects
    missing_in_labels = subjects_from_indices - split_label_subjects
    extra_in_labels = split_label_subjects - subjects_from_indices
    logger.info("  Overlap(cached subjects, ACES labels) (%s): %d", split, len(overlap_with_labels))
    logger.info("  Cached subjects NOT in ACES labels (%s): %d", split, len(missing_in_labels))
    logger.info("  ACES label subjects NOT in cached subset (%s): %d", split, len(extra_in_labels))

    # 6) Compare cached-subset subjects vs original tokenized subjects
    overlap_with_original = subjects_from_indices & split_subjects_from_original[split]
    logger.info("  Overlap(cached subjects, original tokenized) (%s): %d", split, len(overlap_with_original))

# --------------------------------------------------------------------
# CROSS-SPLIT LEAKAGE CHECKS
# --------------------------------------------------------------------
logger.info("\n" + "=" * 80)
logger.info("Cross-split subject leakage checks (cached indices -> subject_id sets):")

split_names = list(split_subjects_from_indices.keys())
for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        s1 = split_names[i]
        s2 = split_names[j]
        inter = split_subjects_from_indices[s1] & split_subjects_from_indices[s2]
        logger.info("  Overlap between %s and %s: %d subjects", s1, s2, len(inter))

logger.info("=" * 80)
logger.info("Done.")