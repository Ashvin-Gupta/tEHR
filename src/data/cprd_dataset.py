'''
CPRD Dataset
'''
import logging 
from typing import Optional, Dict, Any, List
import pandas as pd
from tqdm import tqdm
import torch

from src.data.base_dataset import BaseDataset
from src.data.cprd_translator import CPRDTokenTranslator
import logging
logger = logging.getLogger(__name__)

class CPRDDataset(BaseDataset):
    '''
    Dataset for tokenised CPRD EHR data

    Supports two output formats:
    - Text: translated natural language text
    - Tokens: integer token IDs, only really used for custom transformer models

    Labels come from subject information CSV with columns
    - subject_id: unique patient identifier
    - is_case: 1 if patient has cancer, 0 if control
    - site: cancer site if is_case=1
    - cancerdate: date of cancer diagnosis
    '''
    
    def __init__(
        self, 
        data_dir: str,
        vocab_file: str,
        labels_file: str,
        medical_lookup_file: str,
        lab_lookup_file: str,
        region_lookup_file: str,
        time_lookup_file: str,
        cutoff_months: Optional[int] = None,
        format: str = 'text',
        split: str = 'train',
    ):
        self.split = split
        self.cutoff_months = cutoff_months
        self.format = format

        # Vocabulary: token_id -> token_string
        vocab_df = pd.read_csv(vocab_file, dtype={'str': str})
        self.id_to_token = pd.Series(vocab_df['str'].values, index=vocab_df['token']).to_dict()

        # Labels
        self.subject_to_label, self.subject_to_cancer_date, self.label_to_id = self._load_labels(labels_file)

        # Translator (CPRD specific)
        self.translator = CPRDTokenTranslator.from_csv_files(
            medical_lookup_file,
            lab_lookup_file,
            region_lookup_file,
            time_lookup_file,
        )

        # Patient records
        limit = 1 if split in ['tuning', 'held_out', 'train'] else None
        self.patient_records = self._load_data(data_dir, split, limit=limit)

    def _load_labels(self, labels_file:str):
        df = pd.read_csv(
            labels_file,
            dtype={'site': str, 'cancerdate': str},
            low_memory=False,
        )

        df["string_label"] = df.apply(
            lambda row: "Control" if row["is_case"] == 0 else row["site"],
            axis=1,
        )

        unique_labels = sorted([l for l in df["string_label"].unique() if l != "Control"])
        label_to_id = {label: i + 1 for i, label in enumerate(unique_labels)}
        label_to_id["Control"] = 0

        df["label_id"] = df["string_label"].map(label_to_id)
        subject_to_label = pd.Series(df["label_id"].values, index=df["subject_id"]).to_dict()

        df["cancerdate"] = pd.to_datetime(df["cancerdate"], errors="coerce")
        subject_to_cancer_date = pd.Series(df["cancerdate"].values, index=df["subject_id"]).to_dict()

        return subject_to_label, subject_to_cancer_date, label_to_id
    
    def _apply_time_cutoff(
        self,
        token_ids: List[int],
        timestamps: List,
        subject_id: int,
        label: int,
    ) -> List[int]:
        '''
        Remove tokens within cutoff months before the cancer diagnosis date
        '''

        if self.cutoff_months is None:
            return token_ids
        
        cancer_date = self.subject_to_cancer_date.get(subject_id)

        # If cancer_date is not None/NaT, use it; otherwise use the last valid timestamp (if any)
        if pd.isna(cancer_date):
            valid_timestamps = [ts for ts in timestamps if ts is not None and ts > 0]
            if valid_timestamps:
                last_timestamp = max(valid_timestamps)
                last_date = pd.Timestamp.fromtimestamp(last_timestamp)
                cutoff_ts = (last_date - pd.DateOffset(months=self.cutoff_months)).timestamp()
            else:
                # No valid timestamps, can't apply cutoff
                return token_ids
        else:
            cutoff_ts = (cancer_date - pd.DateOffset(months=self.cutoff_months)).timestamp()

        kept = []

        for i, ts in enumerate(timestamps):
            is_end = self.id_to_token.get(token_ids[i], "") == "<end>"
            if ts == 0 or (ts is not None and ts < cutoff_ts) or is_end:
                kept.append(token_ids[i])

        return kept

    def __len__(self) -> int:
        return len(self.patient_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        '''
        Get a single patient record
        '''
        
        record = self.patient_records[idx]
        subject_id = record['subject_id']
        
        label = self.subject_to_label.get(subject_id)
        if label is None or pd.isna(label):
            return None # Skip patients without labels as all patients in CPRD must have a label of cancer or not
        
        token_ids = record['tokens']
        timestamps = record['timestamps']

        token_ids = self._apply_time_cutoff(token_ids, timestamps, subject_id, label)

        if self.format == 'tokens':
            return {
                'tokens': token_ids,
                'label': torch.tensor(label, dtype=torch.long),
                'subject_id': subject_id,
            }
        
        # Text format
        string_codes = [self.id_to_token.get(tid, "") for tid in token_ids]
        narrative = "".join(self._tokens_to_phrases(string_codes))
        return {
            'text': narrative,
            'label': torch.tensor(label, dtype=torch.long),
            'subject_id': subject_id,
        }
    