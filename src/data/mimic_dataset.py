import os
from typing import Optional, Dict, Any, List
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from src.data.base_dataset import BaseDataset
from src.data.mimic_translator import MIMICTokenTranslator
import logging
logger = logging.getLogger(__name__)

class MIMICDataset(BaseDataset):
    '''
    Dataset for tokenised MIMIC-IV EHR data

    In task == 'prediction', each ACES row becomes one datapoint.
    A subject can have multiple ACES rows. 
    In task == 'pretrain', each patient trajectory becomes one datapoint, label = 0.
    '''
    def __init__(
        self,
        data_dir: str,
        vocab_file: str,
        aces_label_dir: str,
        lookup_dir: str,
        task: str = "pretrain",
        split: str = "train",
        label_column: str = "label",
        format: str = "text",
    ):
        self.task = task
        self.split = split
        self.label_column = label_column
        self.format = format
        
        # Vocabulary: token_id -> token_string
        vocab_df = pd.read_csv(vocab_file, dtype={'str': str})
        self.id_to_token = pd.Series(vocab_df['str'].values, index=vocab_df['token']).to_dict()

        # Translator (MIMIC specific)
        self.translator = MIMICTokenTranslator.from_csv_files(
            labitems_file = os.path.join(lookup_dir, "d_labitems.csv.gz"),
            diagnoses_file = os.path.join(lookup_dir, "d_icd_diagnoses.csv.gz"),
            procedures_file = os.path.join(lookup_dir, "d_icd_procedures.csv.gz"),
        )

        # Load patient records
        limit = 5 if split in ['tuning'] else None
        self.patient_records = self._load_data(data_dir, split, limit=limit)

        # For prediction task: build ACES-keyed examples
        self.examples: List[Dict] = []
        self.subject_to_records: Dict[str, List[Dict]] = {}

        if self.task == 'prediction':
            self._build_prediction_examples(aces_label_dir)
        
        logger.info(
            "MIMICDataset split='%s' task='%s': %d examples",
            split, task, len(self),
        )
    
    def _build_prediction_examples(self, aces_label_dir: str):
        '''
        Build examples from ACES labels for prediction task
        '''
        aces_path = os.path.join(aces_label_dir, f"{self.split}_labels.parquet")
        if not os.path.exists(aces_path):
            logger.warning("ACES labels file not found: %s", aces_path)
            return
        
        print(f"  Building prediction examples from {aces_path}...", flush=True)
        aces_df = pl.read_parquet(aces_path).to_pandas()

        #subject_id -> list of records
        for rec in self.patient_records:
            sid = rec.get("subject_id")
            if sid is not None:
                self.subject_to_records.setdefault(sid, []).append(rec)
        
        skipped = 0
        for _, row in aces_df.iterrows():
            sid = row.get("subject_id")
            if sid not in self.subject_to_records:
                skipped += 1
                continue
            self.examples.append(row.to_dict())

        print(
            f"  Loaded {len(self.patient_records):,} patient trajectories → "
            f"{len(self.examples):,} prediction examples "
            f"({skipped:,} ACES rows skipped, no trajectory found)",
            flush=True,
        )
        logger.info(
            "  Loaded %d patient trajectories → %d prediction examples "
            "(%d ACES rows skipped, no trajectory found)",
            len(self.patient_records), len(self.examples), skipped,
        )
    
    def _apply_prediction_time_cutoff(
        self,
        tokens: List[int],
        timestamps: List[float],
        pred_time: float,
    ) -> List[int]:
        '''
        Apply prediction time cutoff to patient trajectory
        '''
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
    
    def __len__(self) -> int:
        if self.task == 'prediction' and self.examples:
            return len(self.examples)
        return len(self.patient_records)
    
    def __getitem__(self, idx):
        if self.task == 'prediction' and self.examples:
            example = self.examples[idx]
            subject_id = example["subject_id"]
            
            records = self.subject_to_records.get(subject_id)
            if not records:
                raise IndexError(f"No patient trajectory found for subject_id={subject_id}")
            record = records[0]

            tokens = list(record['tokens'])
            timestamps = list(record['timestamps'])
            tokens = self._apply_prediction_time_cutoff(tokens, timestamps, example.get("prediction_time"))

            label_value = 0
            if self.label_column in example:
                try:
                    label_value = int(example[self.label_column])
                except Exception:
                    label_value = 1 if example[self.label_column] else 0

        else:
            record = self.patient_records[idx]
            subject_id = record['subject_id']
            tokens = list(record['tokens'])
            timestamps = list(record['timestamps'])
            label_value = 0

        if self.format == "tokens":
            return {
                "tokens": tokens,
                "label": torch.tensor(label_value, dtype=torch.long),
                "subject_id": subject_id,
            }

        string_codes = [self.id_to_token.get(tid, "") for tid in tokens]
        narrative = "".join(self._tokens_to_phrases(string_codes))
        return {
            "text": narrative,
            "label": torch.tensor(label_value, dtype=torch.long),
            "subject_id": subject_id,
        }