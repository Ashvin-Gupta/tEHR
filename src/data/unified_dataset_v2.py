"""
Refactored unified dataset for EHR data with improved organization.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import os
from tqdm import tqdm
import random
from typing import List, Dict, Optional, Any

from src.data.token_translator import EHRTokenTranslator


class UnifiedEHRDataset(Dataset):
    """
    A unified, flexible dataset for EHR-based cancer classification.
    
    This 'smart' dataset can:
    1. Provide data as integer tokens for custom models ('tokens' format).
    2. Provide data as natural language text for LLM fine-tuning ('text' format).
    3. Provide data for LLM continued pretraining with random window sampling ('pretrain' format).
    4. Dynamically truncate patient timelines based on a specified cutoff window
       before the cancer diagnosis date.
    """
    
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
        max_sequence_length: int = 512,
        format: str = 'tokens',
        split: str = 'train',
        tokenizer: Optional[Any] = None,
        data_type: str = 'binned'
    ):
        """
        Initialize the EHR dataset.
        
        Args:
            data_dir: Directory containing patient record pickle files.
            vocab_file: Path to vocabulary CSV file.
            labels_file: Path to patient labels CSV file.
            medical_lookup_file: Path to medical codes lookup CSV.
            lab_lookup_file: Path to lab codes lookup CSV.
            region_lookup_file: Path to region codes lookup CSV.
            time_lookup_file: Path to time codes lookup CSV.
            cutoff_months: Number of months before diagnosis to remove (for cancer patients).
            max_sequence_length: Maximum sequence length (unused for text format).
            format: Output format ('tokens', 'text', 'pretrain', 'events').
            split: Data split ('train', 'tuning', 'held_out').
            tokenizer: Optional tokenizer for pretrain format.
            data_type: Type of data ('binned' or other).
        """
        self.format = format
        self.cutoff_months = cutoff_months
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.data_type = data_type
        
        # Load mappings and lookups
        self._load_mappings(
            vocab_file, labels_file,
            medical_lookup_file, lab_lookup_file,
            region_lookup_file, time_lookup_file
        )
        
        # Load patient records
        limit = 1 if split in ['tuning', 'held_out'] else None
        self.patient_records = self._load_data(data_dir, split, limit=limit)
    
    def _load_mappings(
        self,
        vocab_file: str,
        labels_file: str,
        medical_lookup_file: str,
        lab_lookup_file: str,
        region_lookup_file: str,
        time_lookup_file: str
    ):
        """Load all vocabularies, translation lookups, and label information."""
        # Load vocabulary
        vocab_df = pd.read_csv(vocab_file, dtype={'str': str})
        self.id_to_token_map = pd.Series(
            vocab_df['str'].values,
            index=vocab_df['token']
        ).to_dict()
        
        # Load translator for text formats
        if self.format in ['text', 'events']:
            self.translator = EHRTokenTranslator.from_csv_files(
                medical_lookup_file,
                lab_lookup_file,
                region_lookup_file,
                time_lookup_file
            )
        
        # Load labels
        labels_df = pd.read_csv(labels_file)
        labels_df['string_label'] = labels_df.apply(
            lambda row: 'Control' if row['is_case'] == 0 else row['site'],
            axis=1
        )
        
        unique_labels = sorted([
            l for l in labels_df['string_label'].unique()
            if l != 'Control'
        ])
        self.label_to_id_map = {label: i + 1 for i, label in enumerate(unique_labels)}
        self.label_to_id_map['Control'] = 0
        
        labels_df['label_id'] = labels_df['string_label'].map(self.label_to_id_map)
        self.subject_to_label = pd.Series(
            labels_df['label_id'].values,
            index=labels_df['subject_id']
        ).to_dict()
        
        labels_df['cancerdate'] = pd.to_datetime(labels_df['cancerdate'], errors='coerce')
        self.subject_to_cancer_date = pd.Series(
            labels_df['cancerdate'].values,
            index=labels_df['subject_id']
        ).to_dict()
    
    def _load_data(
        self,
        data_dir: str,
        split: str,
        limit: Optional[int] = None,
        seed: int = 42
    ) -> List[Dict]:
        """
        Load patient records from pickle files.
        
        Args:
            data_dir: Base directory containing split subdirectories.
            split: Data split name.
            limit: Optional limit on number of files to load.
            seed: Random seed for sampling (when limit is set for train split).
        
        Returns:
            List of patient record dictionaries.
        """
        data_dir = os.path.join(data_dir, split)
        records = []
        pkl_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.pkl')
        ]

        if limit is not None:
            if split == 'train':
                random.seed(seed)
                pkl_files = random.sample(pkl_files, min(limit, len(pkl_files)))
            else:
                pkl_files = pkl_files[:limit]

        for file_path in tqdm(pkl_files, desc=f"Loading data from {data_dir}"):
            with open(file_path, 'rb') as f:
                records.extend(pickle.load(f))
        
        return records
    
    def _apply_time_cutoff(
        self,
        token_ids: List[int],
        timestamps: List[float],
        subject_id: int,
        label: int
    ) -> List[int]:
        """
        Apply time-based cutoff to patient timeline.
        
        For cancer patients, removes events within N months before diagnosis.
        
        Args:
            token_ids: List of token IDs for the patient.
            timestamps: List of corresponding timestamps.
            subject_id: Patient ID.
            label: Patient label (0 = control, >0 = cancer).
        
        Returns:
            Filtered list of token IDs.
        """
        # Determine cutoff to apply
        if self.format == 'pretrain' and label > 0:
            actual_cutoff = 1  # Always 1 month for pretraining cancer patients
        elif self.cutoff_months is not None:
            actual_cutoff = self.cutoff_months
        else:
            return token_ids  # No cutoff
        
        # Get cancer date
        cancer_date = self.subject_to_cancer_date.get(subject_id)
        if pd.isna(cancer_date):
            return token_ids  # No cancer date, no cutoff
        
        # Calculate cutoff timestamp
        cutoff_date = cancer_date - pd.DateOffset(months=actual_cutoff)
        cutoff_timestamp = cutoff_date.timestamp()
        
        # Filter tokens
        truncated_ids = []
        for i, ts in enumerate(timestamps):
            token_str = self.id_to_token_map.get(token_ids[i], "")
            is_end_token = (token_str == '<end>')
            
            # Keep: special tokens (ts==0), tokens before cutoff, OR the <end> token
            if ts == 0 or (ts is not None and ts < cutoff_timestamp) or is_end_token:
                truncated_ids.append(token_ids[i])
        
        return truncated_ids
    
    def _combine_measurement_tokens(self, string_codes: List[str]) -> List[str]:
        """
        Combine measurement/lab tokens with their values and units.
        
        Converts sequences like ["LAB//glucose", "5.5", "mmol/L"] into
        ["Glucose 5.5 mmol/L; "].
        
        Args:
            string_codes: List of token strings.
        
        Returns:
            List of translated phrases.
        """
        translated_phrases = []
        i = 0
        
        while i < len(string_codes):
            current_code = str(string_codes[i])
            
            # Check if this is a measurable concept
            is_measurable = self.translator.is_measurable_concept(current_code)
            has_next_token = (i + 1 < len(string_codes))
            is_next_a_value = False
            
            if has_next_token:
                next_code = str(string_codes[i + 1])
                is_next_a_value = self.translator.is_numeric_value(next_code)
            
            # Combine measurement + value + optional unit
            if is_measurable and is_next_a_value:
                concept = self.translator.translate(current_code)
                value_bin = self.translator.translate(string_codes[i + 1])
                
                unit_str = ""
                increment = 2  # Default: skip code and value
                
                # Check for unit token
                if i + 2 < len(string_codes):
                    potential_unit = str(string_codes[i + 2])
                    
                    if not self.translator.is_new_event_code(potential_unit):
                        # It's a unit!
                        unit_str = f" {potential_unit}"
                        increment = 3  # Skip code, value, AND unit
                
                # Format the combined measurement
                if concept and value_bin:
                    if unit_str:
                        # Remove trailing "; " from components
                        concept_clean = concept.rstrip('; ').strip()
                        value_clean = value_bin.rstrip('; ').strip()
                        translated_phrases.append(f"{concept_clean} {value_clean}{unit_str}; ")
                    else:
                        # No unit, keep original format
                        translated_phrases.append(f"{concept} {value_bin}")
                
                i += increment
            
            # Single token translation
            else:
                phrase = self.translator.translate(current_code)
                if phrase:
                    translated_phrases.append(phrase)
                i += 1
        
        return translated_phrases
    
    def __len__(self) -> int:
        """Return the number of patient records."""
        return len(self.patient_records)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a single patient record.
        
        Args:
            idx: Index of the patient record.
        
        Returns:
            Dictionary with 'text' and 'label' keys, or None if invalid.
        """
        patient_record = self.patient_records[idx]
        subject_id = patient_record['subject_id']
        
        # Get label
        label = self.subject_to_label.get(subject_id)
        if pd.isna(label):
            return None  # Skip patients without labels
        
        # Get token IDs and timestamps
        token_ids = patient_record['tokens']
        timestamps = patient_record['timestamps']
        
        # Apply time-based cutoff if needed
        token_ids = self._apply_time_cutoff(token_ids, timestamps, subject_id, label)
        
        # Convert to string codes
        string_codes = [self.id_to_token_map.get(tid, "") for tid in token_ids]
        
        # Filter out tokens containing "cancer" (case-insensitive)
        string_codes = [code for code in string_codes if 'cancer' not in str(code).lower()]
        
        # Translate to natural language
        translated_phrases = self._combine_measurement_tokens(string_codes)
        narrative = "".join(translated_phrases)
        
        return {
            "text": narrative,
            "label": torch.tensor(label, dtype=torch.long)
        }
