"""Tokenised EHRSHOT trajectories + ACES-style prediction labels."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl
import torch

from src.data.base_dataset import BaseDataset
from src.data.ehrshot_translator import EHRSHOTTokenTranslator

logger = logging.getLogger(__name__)


class EHRSHOTDataset(BaseDataset):
    """Same pickle + prediction-time contract as ``MIMICDataset``."""

    def __init__(
        self,
        data_dir: str,
        vocab_file: str,
        aces_label_dir: Optional[str],
        concept_csv: str,
        task: str = "pretrain",
        split: str = "train",
        label_column: str = "label",
        format: str = "text",
        surface_form: str = "text",
        scramble_seed: int = 42,
        drop_time: bool = False,
        drop_values: bool = False,
        max_events: Optional[int] = None,
        file_fraction: Optional[float] = None,
        file_fraction_seed: int = 42,
        **_ignored,
    ):
        self.task = task
        self.split = split
        self.label_column = label_column
        self.format = format
        self.surface_form = surface_form
        self.scramble_seed = scramble_seed
        self.drop_time = bool(drop_time)
        self.drop_values = bool(drop_values)
        self.max_events = int(max_events) if max_events else None

        vocab_df = pd.read_csv(vocab_file, dtype={"str": str})
        self.id_to_token = pd.Series(
            vocab_df["str"].values, index=vocab_df["token"]
        ).to_dict()
        self.id_to_token_map = self.id_to_token

        if not concept_csv or not os.path.isfile(concept_csv):
            raise ValueError(
                "data.concept_csv is required for EHRSHOT "
                "(Athena concept.csv with vocabulary_id / concept_code / concept_name)."
            )
        self.translator = EHRSHOTTokenTranslator.from_concept_csv(
            concept_csv,
            surface_form=surface_form,
            scramble_seed=scramble_seed,
        )

        self.patient_records = self._load_data(
            data_dir,
            split,
            limit=None,
            file_fraction=file_fraction,
            seed=file_fraction_seed,
        )

        self.examples: List[Dict[str, Any]] = []
        self.subject_to_records: Dict[Any, List[Dict]] = {}
        if self.task == "prediction":
            if not aces_label_dir:
                raise ValueError(
                    "aces_label_dir is required when task='prediction'"
                )
            self._build_prediction_examples(aces_label_dir)

        logger.info(
            "EHRSHOTDataset split='%s' task='%s' surface_form='%s' "
            "drop_time=%s drop_values=%s max_events=%s: %d examples",
            split,
            task,
            surface_form,
            self.drop_time,
            self.drop_values,
            self.max_events,
            len(self),
        )

    def _build_prediction_examples(self, aces_label_dir: str) -> None:
        aces_path = os.path.join(aces_label_dir, f"{self.split}_labels.parquet")
        if not os.path.exists(aces_path):
            logger.warning("ACES labels file not found: %s", aces_path)
            return

        print(f"  Building prediction examples from {aces_path}...", flush=True)
        aces_df = pl.read_parquet(aces_path).to_pandas()
        for rec in self.patient_records:
            sid = rec.get("subject_id")
            if sid is not None:
                self.subject_to_records.setdefault(int(sid), []).append(rec)

        skipped = 0
        for _, row in aces_df.iterrows():
            sid = row.get("subject_id")
            if sid is None or int(sid) not in self.subject_to_records:
                skipped += 1
                continue
            self.examples.append(row.to_dict())
        print(
            f"  Loaded {len(self.patient_records):,} patient trajectories → "
            f"{len(self.examples):,} prediction examples "
            f"({skipped:,} label rows skipped, no trajectory found)",
            flush=True,
        )

    def _apply_prediction_time_cutoff(
        self,
        tokens: List[int],
        timestamps: List,
        pred_time,
    ) -> List[int]:
        if pred_time is None or len(timestamps) == 0:
            return tokens
        first_ts = timestamps[0]
        if isinstance(first_ts, (int, float)):
            numeric_pred = None
            if isinstance(pred_time, (int, float)):
                numeric_pred = float(pred_time)
            elif isinstance(pred_time, pd.Timestamp):
                try:
                    numeric_pred = float(pred_time.timestamp())
                except Exception:
                    pass
            if numeric_pred is not None:
                return [
                    t
                    for i, t in enumerate(tokens)
                    if timestamps[i] <= numeric_pred
                ]
        elif isinstance(first_ts, pd.Timestamp) or hasattr(first_ts, "year"):
            ts_pred = None
            if isinstance(pred_time, pd.Timestamp):
                ts_pred = pred_time
            elif hasattr(pred_time, "year"):
                ts_pred = pred_time
            elif isinstance(pred_time, (int, float)):
                try:
                    ts_pred = pd.to_datetime(pred_time, unit="s")
                except Exception:
                    pass
            else:
                ts_pred = pd.to_datetime(pred_time, errors="coerce")
            if ts_pred is not None and not pd.isna(ts_pred):
                return [
                    t
                    for i, t in enumerate(tokens)
                    if pd.Timestamp(timestamps[i]) <= ts_pred
                ]
        return tokens

    def __len__(self) -> int:
        if self.task == "prediction" and self.examples:
            return len(self.examples)
        return len(self.patient_records)

    def __getitem__(self, idx):
        if self.task == "prediction" and self.examples:
            example = self.examples[idx]
            subject_id = int(example["subject_id"])
            records = self.subject_to_records.get(subject_id)
            if not records:
                raise IndexError(
                    f"No patient trajectory found for subject_id={subject_id}"
                )
            record = records[0]
            tokens = list(record["tokens"])
            timestamps = list(record["timestamps"])
            tokens = self._apply_prediction_time_cutoff(
                tokens, timestamps, example.get("prediction_time")
            )
            label_value = 0
            if self.label_column in example:
                try:
                    label_value = int(example[self.label_column])
                except Exception:
                    label_value = 1 if example[self.label_column] else 0
        else:
            record = self.patient_records[idx]
            subject_id = record["subject_id"]
            tokens = list(record["tokens"])
            timestamps = list(record["timestamps"])
            label_value = 0

        if self.format == "tokens":
            tokens = self._apply_max_events_to_ids(tokens)
            return {
                "tokens": tokens,
                "label": torch.tensor(label_value, dtype=torch.long),
                "subject_id": subject_id,
            }

        string_codes = [self.id_to_token.get(tid, "") for tid in tokens]
        string_codes = self._apply_max_events_to_codes(string_codes)
        narrative = "".join(self._tokens_to_phrases(string_codes))
        return {
            "text": narrative,
            "label": torch.tensor(label_value, dtype=torch.long),
            "subject_id": subject_id,
        }
