"""
Shared test fixtures and utilities.
"""
import pytest
import torch
import numpy as np
import os
import pickle
import gzip
import pandas as pd

from src.data.mimic_translator import MIMICTokenTranslator
from src.data.cprd_translator import CPRDTokenTranslator


# ---------------------------------------------------------------------------
# Mock Tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """
    Simple deterministic character-level tokenizer for unit tests.

    Special IDs:
      0 = pad, 1 = bos, 2 = eos, 3+ = characters
    This avoids loading any real model weights.
    """
    eos_token = "</s>"
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    cls_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        ids = [ord(c) % 250 + 3 for c in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def __call__(self, texts, add_special_tokens=False, truncation=False,
                 padding=False, **kwargs):
        return {
            "input_ids": [
                self.encode(t, add_special_tokens=add_special_tokens)
                for t in texts
            ]
        }


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


# ---------------------------------------------------------------------------
# In-memory translator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mimic_translator():
    """MIMICTokenTranslator backed entirely by in-memory dicts (no files)."""
    lab_lookup = {
        51006: "Urea Nitrogen",
        50912: "Creatinine",
        51265: "Platelet Count",
    }
    diag_lookup = {
        "10_I10": "Essential (primary) hypertension",
        "9_4019": "Unspecified essential hypertension",
    }
    proc_lookup = {
        "10_0BH17EZ": "Insertion of endotracheal airway",
    }
    return MIMICTokenTranslator(lab_lookup, diag_lookup, proc_lookup)


@pytest.fixture
def cprd_translator():
    """
    CPRDTokenTranslator backed entirely by in-memory dicts (no files).
    NOTE: will fail to instantiate until is_numeric_value is added to CPRDTokenTranslator.
    """
    medical_lookup = {
        "11N3200": "Haemoglobin",
        "ABCD": "Haemoglobin",
        "XM0LY": "Systolic blood pressure",
    }
    lab_lookup = {
        "HAB1": "HbA1c",
        "GFR": "eGFR",
    }
    region_lookup = {
        "1": "North East",
        "2": "North West",
    }
    time_lookup = {
        "1M": "1 month",
        "3M": "3 months",
    }
    return CPRDTokenTranslator(medical_lookup, lab_lookup, region_lookup, time_lookup)


# ---------------------------------------------------------------------------
# Fake dataset helpers (for integration tests)
# ---------------------------------------------------------------------------

def write_fake_mimic_data(tmp_path: str, records_per_split: int = 5):
    """
    Build a minimal, self-consistent fake MIMIC dataset on disk so that
    MIMICDataset can be instantiated without real data.

    Returns a dict with all the relevant paths the dataset constructor needs.
    """
    # ---- vocab ----
    vocab_rows = [
        (0, "<pad>"), (1, "<start>"), (2, "AGE"), (3, "55.0"),
        (4, "GENDER//F"), (5, "LAB//51006"), (6, "13.5"), (7, "mmol/L"),
        (8, "DIAGNOSIS//ICD//10_I10"), (9, "TIME 2020-01"),
        (10, "MEDICATION//START//metformin"),
    ]
    vocab_df = pd.DataFrame(vocab_rows, columns=["token", "str"])
    vocab_file = os.path.join(tmp_path, "vocab.csv")
    vocab_df.to_csv(vocab_file, index=False)

    # ---- lookup tables ----
    hosp_dir = os.path.join(tmp_path, "hosp")
    os.makedirs(hosp_dir, exist_ok=True)

    lab_df = pd.DataFrame({"itemid": [51006, 50912], "label": ["Urea Nitrogen", "Creatinine"]})
    lab_df.to_csv(os.path.join(hosp_dir, "d_labitems.csv.gz"), index=False, compression="gzip")

    diag_df = pd.DataFrame({
        "icd_version": [10, 9],
        "icd_code": ["I10", "4019"],
        "long_title": ["Essential hypertension", "Unspecified essential hypertension"],
    })
    diag_df.to_csv(os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"), index=False, compression="gzip")

    proc_df = pd.DataFrame({
        "icd_version": [10],
        "icd_code": ["0BH17EZ"],
        "long_title": ["Insertion of Endotracheal Airway"],
    })
    proc_df.to_csv(os.path.join(hosp_dir, "d_icd_procedures.csv.gz"), index=False, compression="gzip")

    # ---- patient records ----
    base_time = 1577836800  # 2020-01-01 00:00:00 UTC
    splits = ["train", "tuning", "held_out"]
    for split in splits:
        split_dir = os.path.join(tmp_path, "data", split)
        os.makedirs(split_dir, exist_ok=True)
        records = []
        for j in range(records_per_split):
            subject_id = 1000 + j
            # Build a short token sequence with timestamps
            tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            # Timestamps: header tokens have t=0, clinical tokens have real timestamps
            timestamps = [0, 0, 0, 0, base_time, base_time, base_time,
                          base_time + 3600, base_time + 7200]
            records.append({
                "subject_id": subject_id,
                "tokens": tokens,
                "timestamps": timestamps,
            })
        pkl_path = os.path.join(split_dir, "batch_0.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(records, f)

    # ---- ACES labels (parquet) ----
    aces_dir = os.path.join(tmp_path, "aces_labels")
    os.makedirs(aces_dir, exist_ok=True)
    for split in splits:
        rows = []
        for j in range(records_per_split):
            rows.append({
                "subject_id": 1000 + j,
                "label": j % 2,          # alternate 0/1
                "prediction_time": float(base_time + 7200),
            })
        aces_df = pd.DataFrame(rows)
        aces_df.to_parquet(os.path.join(aces_dir, f"{split}_labels.parquet"), index=False)

    return {
        "data_dir": os.path.join(tmp_path, "data"),
        "vocab_filepath": vocab_file,
        "aces_label_dir": aces_dir,
        "lookup_dir": hosp_dir,
    }


def write_fake_cprd_data(tmp_path: str, records_per_split: int = 5):
    """
    Build a minimal, self-consistent fake CPRD dataset on disk.
    """
    # ---- vocab ----
    vocab_rows = [
        (0, "<pad>"), (1, "<start>"), (2, "AGE"), (3, "55.0"),
        (4, "GENDER//F"), (5, "MEDICAL//ABCD"), (6, "13.5"),
        (7, "g/dL"), (8, "<end>"),
    ]
    vocab_df = pd.DataFrame(vocab_rows, columns=["token", "str"])
    vocab_file = os.path.join(tmp_path, "vocab.csv")
    vocab_df.to_csv(vocab_file, index=False)

    # ---- labels ----
    base_time = 1577836800
    label_rows = []
    for j in range(records_per_split):
        subject_id = 2000 + j
        is_case = j % 2
        cancerdate = "2020-06-15" if is_case else None
        site = "pancreas" if is_case else None
        label_rows.append({"subject_id": subject_id, "is_case": is_case,
                            "site": site, "cancerdate": cancerdate})
    labels_df = pd.DataFrame(label_rows)
    labels_file = os.path.join(tmp_path, "labels.csv")
    labels_df.to_csv(labels_file, index=False)

    # ---- lookup files ----
    med_df = pd.DataFrame({"code": ["ABCD", "XM0LY"], "term": ["Haemoglobin", "Systolic BP"]})
    med_file = os.path.join(tmp_path, "medical_lookup.csv")
    med_df.to_csv(med_file, index=False)

    lab_df = pd.DataFrame({"code": ["HAB1"], "term": ["HbA1c"]})
    lab_file = os.path.join(tmp_path, "lab_lookup.csv")
    lab_df.to_csv(lab_file, index=False)

    reg_df = pd.DataFrame({"regionid": ["1", "2"], "Description": ["North East", "North West"]})
    reg_file = os.path.join(tmp_path, "region_lookup.csv")
    reg_df.to_csv(reg_file, index=False)

    time_df = pd.DataFrame({"code": ["1M", "3M"], "term": ["1 month", "3 months"]})
    time_file = os.path.join(tmp_path, "time_lookup.csv")
    time_df.to_csv(time_file, index=False)

    # ---- patient records ----
    splits = ["train", "tuning", "held_out"]
    for split in splits:
        split_dir = os.path.join(tmp_path, "data", split)
        os.makedirs(split_dir, exist_ok=True)
        records = []
        for j in range(records_per_split):
            subject_id = 2000 + j
            tokens = [1, 2, 3, 4, 5, 6, 7, 8]
            timestamps = [0, 0, 0, 0, base_time, base_time, base_time, 0]
            records.append({
                "subject_id": subject_id,
                "tokens": tokens,
                "timestamps": timestamps,
            })
        pkl_path = os.path.join(split_dir, "batch_0.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(records, f)

    return {
        "data_dir": os.path.join(tmp_path, "data"),
        "vocab_filepath": vocab_file,
        "labels_filepath": labels_file,
        "medical_lookup_filepath": med_file,
        "lab_lookup_filepath": lab_file,
        "region_lookup_filepath": reg_file,
        "time_lookup_filepath": time_file,
    }
