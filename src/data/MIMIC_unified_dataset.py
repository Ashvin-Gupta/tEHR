import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import os
import polars as pl

class MIMICUnifiedDataset(Dataset):
    def __init__(
        self,
        data_dir,
        vocab_file,
        aces_label_dir,
        lookup_dir,
        task: str = "pretrain",
        split: str = "train",
        max_sequence_length: int = 512,
        label_column: str = "label",
    ):
        self.task = task
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.aces_data = {}
        self.label_column = label_column

        self._load_mappings(vocab_file, lookup_dir)

        if self.task == 'prediction':
            aces_path = os.path.join(aces_label_dir, f"{split}_labels.parquet")
            if os.path.exists(aces_path):
                self.aces_data = self._load_aces(aces_path)

        limit = 1 if split in ['tuning', 'held_out', 'train'] else None
        self.patient_records = self._load_data(data_dir, split, limit=limit)

    def _load_mappings(self, vocab_file, lookup_dir):
        vocab_df = pd.read_csv(vocab_file, dtype={'str': str})
        self.id_to_token_map = pd.Series(vocab_df['str'].values, index=vocab_df['token']).to_dict()

        self.lab_lookup = pd.read_csv(os.path.join(lookup_dir, "d_labitems.csv.gz")).set_index('itemid')['label'].to_dict()
        
        diag_df = pd.read_csv(os.path.join(lookup_dir, "d_icd_diagnoses.csv.gz"))
        diag_df['key'] = diag_df['icd_version'].astype(str) + "_" + diag_df['icd_code'].astype(str)
        self.diag_lookup = diag_df.set_index('key')['long_title'].to_dict()

        proc_df = pd.read_csv(os.path.join(lookup_dir, "d_icd_procedures.csv.gz"))
        proc_df['key'] = proc_df['icd_version'].astype(str) + "_" + proc_df['icd_code'].astype(str)
        self.proc_lookup = proc_df.set_index('key')['long_title'].to_dict()

    def _load_aces(self, path):
        return pl.read_parquet(path).to_pandas().set_index('subject_id').to_dict('index')

    def _is_header_concept(self, token_code):
        token_code = str(token_code).upper()
        if token_code.startswith(('<START>', 'AGE', 'GENDER', 'RACE', 'MARITAL_STATUS')):
            return True
        return False

    def _translate_token(self, token):
        if not isinstance(token, str): return ""
        try:
            # 1. Labs & Procedures & Diagnoses
            if token.startswith('LAB//'):
                parts = token.split('//')
                return f"{self.lab_lookup.get(int(parts[1]), parts[1])}; "
            elif token.startswith('DIAGNOSIS//ICD//'):
                parts = token.split('//')
                return f"{self.diag_lookup.get(f'{parts[2]}_{parts[3]}', parts[3])}; "
            elif token.startswith('PROCEDURE//ICD//'):
                parts = token.split('//')
                return f"{self.proc_lookup.get(f'{parts[2]}_{parts[3]}', parts[3])}; "

            elif token.startswith('TIME '):
                return f"{token_string[5:]}; "

            # 2. Medications (e.g., start Ondansetron)
            elif token.startswith('MEDICATION//'):
                parts = token.split('//')
                if 'START' in parts:
                    return f"start {parts[-1]}; "
                return f"{parts[1]} {parts[-1]}; "

            # 3. Admissions (e.g., hospital admission, emergency room)
            elif any(x in token for x in ['ADMISSION', 'DISCHARGE', 'TRANSFER']):
                return f"{token.replace('//', ', ').replace('_', ' ').lower()}; "

            # 4. Demographics
            elif token.startswith('GENDER//'):
                val = token.split('//')[1]
                return "Female; " if val == 'F' else "Male; "
            elif token.startswith('RACE//'):
                val = token.split('//')[1].replace('RACE_', '').replace('_', ' ').title()
                return f"Race {val}; "
            elif token.startswith('MARITAL_STATUS//'):
                return f"{token.split('//')[1].upper()}; "

            # 5. Fixed Keys
            elif token.startswith('AGE'): return "Age; "
            elif token.startswith('BMI'): return "BMI; "
            elif token.startswith('Blood Pressure'): return "Blood Pressure; "
            elif token == '<start>': return "<start>; "
            
            # 6. Raw Values (handled in pairing loop, but fallback here)
            elif token.replace('.', '', 1).isdigit() or '/' in token:
                return f"{token}; "
            
            return f"{token.replace('//', ' ').lower()}; "
        except:
            return f"{token}; "

    def __len__(self):
        return len(self.patient_records)

    def __getitem__(self, idx):
        record = self.patient_records[idx]
        subject_id = record['subject_id']
        tokens, timestamps = record['tokens'], record['timestamps']

        label_value = 0

        if self.task == 'prediction':
            task_info = self.aces_data.get(subject_id)
            if task_info:
                # Time-based truncation to prediction_time where types are compatible.
                pred_time = task_info.get("prediction_time")
                if pred_time is not None and len(timestamps) > 0:
                    first_ts = timestamps[0]
                    # Only apply comparison when both sides are numeric or both timestamps.
                    if isinstance(first_ts, (int, float)) and isinstance(pred_time, (int, float)):
                        tokens = [
                            t for i, t in enumerate(tokens)
                            if timestamps[i] <= pred_time
                        ]
                    elif isinstance(first_ts, pd.Timestamp) and isinstance(pred_time, pd.Timestamp):
                        tokens = [
                            t for i, t in enumerate(tokens)
                            if timestamps[i] <= pred_time
                        ]
                    # Otherwise, skip time filtering to avoid type errors.
                # Use configured label column if present in ACES row
                if self.label_column in task_info:
                    try:
                        label_value = int(task_info[self.label_column])
                    except Exception:
                        # Fallback: treat any non-zero / truthy value as positive
                        label_value = 1 if task_info[self.label_column] else 0

        string_codes = [self.id_to_token_map.get(tid, "") for tid in tokens]
        translated_phrases = []
        
        split_inserted = False
        i = 0
        while i < len(string_codes):
            code = str(string_codes[i])

            if not split_inserted and not self._is_header_concept(code):
                translated_phrases.append("<HEADER_SPLIT>; ")
                split_inserted = True

            # PAIRING LOGIC: Separates concept and value with a semicolon
            is_meas = code.startswith(('LAB//', 'AGE', 'BMI', 'Blood Pressure'))
            if is_meas and i + 1 < len(string_codes):
                val = str(string_codes[i+1])
                if any(c.isdigit() for c in val):
                    concept = self._translate_token(code) # Already ends in "; "
                    value = self._translate_token(val)    # Already ends in "; "
                    
                    # Look for Unit
                    unit_str = ""
                    increment = 2
                    if i + 2 < len(string_codes):
                        potential_unit = str(string_codes[i+2])
                        if not potential_unit.startswith(('LAB//', 'DIAGNOSIS//', '<time', 'MEDICATION//')):
                            unit_str = self._translate_token(potential_unit)
                            increment = 3
                    
                    translated_phrases.append(f"{concept}{value}{unit_str}")
                    i += increment
                    continue

            phrase = self._translate_token(code)
            if phrase: translated_phrases.append(phrase)
            i += 1

        return {
            "text": "".join(translated_phrases),
            "label": torch.tensor(label_value, dtype=torch.long),
        }

    def _load_data(self, data_dir, split, limit=None):
        path = os.path.join(data_dir, split)
        records = []
        pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
        if limit: pkl_files = pkl_files[:limit]
        for f in pkl_files:
            with open(os.path.join(path, f), 'rb') as f_in:
                records.extend(pickle.load(f_in))
        return records



            