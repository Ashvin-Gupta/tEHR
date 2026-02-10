import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import os
from tqdm import tqdm
import random

class UnifiedEHRDataset(Dataset):
    """
    A unified, flexible dataset for EHR-based cancer classification.

    This 'smart' dataset can:
    1.  Provide data as integer tokens for custom models ('tokens' format).
    2.  Provide data as natural language text for LLM fine-tuning ('text' format).
    3.  Provide data for LLM continued pretraining with random window sampling ('pretrain' format).
    4.  Dynamically truncate patient timelines based on a specified cutoff window
        before the cancer diagnosis date.
    """
    def __init__(self, data_dir, vocab_file, labels_file, medical_lookup_file, lab_lookup_file, region_lookup_file, time_lookup_file,
                 cutoff_months=None, max_sequence_length=512, format='tokens', split='train', tokenizer=None, data_type='binned'):
        
        self.format = format
        self.cutoff_months = cutoff_months
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer  # Store for pretrain format
        
        # Load all necessary mappings and lookup tables
        self._load_mappings(vocab_file, labels_file, medical_lookup_file, lab_lookup_file, region_lookup_file, time_lookup_file)
        self.data_type = data_type
        # Load the patient records from the .pkl files for the specified split
        # self.patient_records = self._load_data(data_dir, split)
        if split == 'tuning' or split:
            self.patient_records = self._load_data(data_dir, split, limit=1)
        else:
            # Chaning to 5 to see result and inference
            self.patient_records = self._load_data(data_dir, split)

    def _load_mappings(self, vocab_file, labels_file, medical_lookup_file, lab_lookup_file, region_lookup_file, time_lookup_file):
        """Loads all vocabularies, translation lookups, and label information."""
        
        vocab_df = pd.read_csv(vocab_file, dtype={'str': str})
        self.id_to_token_map = pd.Series(vocab_df['str'].values, index=vocab_df['token']).to_dict()

        if self.format == 'text' or self.format == 'events':
            medical_df = pd.read_csv(medical_lookup_file)
            self.medical_lookup = pd.Series(medical_df['term'].values, index=medical_df['code'].astype(str).str.upper()).to_dict()
            
            lab_df = pd.read_csv(lab_lookup_file)
            self.lab_lookup = pd.Series(lab_df['term'].values, index=lab_df['code'].astype(str).str.upper()).to_dict()

            region_df = pd.read_csv(region_lookup_file)
            self.region_lookup = pd.Series(region_df['Description'].values, index=region_df['regionid'].astype(str).str.upper()).to_dict()

            time_df = pd.read_csv(time_lookup_file)
            self.time_lookup = pd.Series(time_df['term'].values, index=time_df['code'].astype(str).str.upper()).to_dict()

        labels_df = pd.read_csv(labels_file, dtype={'site': str, 'cancerdate': str}, low_memory=False)
        labels_df['string_label'] = labels_df.apply(lambda row: 'Control' if row['is_case'] == 0 else row['site'], axis=1)
        
        unique_labels = sorted([l for l in labels_df['string_label'].unique() if l != 'Control'])
        self.label_to_id_map = {label: i + 1 for i, label in enumerate(unique_labels)}
        self.label_to_id_map['Control'] = 0
        
        labels_df['label_id'] = labels_df['string_label'].map(self.label_to_id_map)
        self.subject_to_label = pd.Series(labels_df['label_id'].values, index=labels_df['subject_id']).to_dict()
        
        labels_df['cancerdate'] = pd.to_datetime(labels_df['cancerdate'], errors='coerce')
        self.subject_to_cancer_date = pd.Series(labels_df['cancerdate'].values, index=labels_df['subject_id']).to_dict()


    def _load_data(self, data_dir, split, limit=None, seed=42):
        """Loads a limited number of patient records from .pkl files in a directory."""
        data_dir = os.path.join(data_dir, split)
        records = []
        pkl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]

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

    def _is_header_concept(self, token_code):
        """
        Check if a raw token code belongs to the demographic header.
        Includes Age, Gender, Ethnicity, Region, and Vitals (BMI, Height, Weight).
        """
        token_code = str(token_code).upper()
        
        # 1. Standard Demographic Prefixes
        # <start> is included so we don't split immediately at the beginning
        if token_code.startswith(('<START>', 'AGE', 'GENDER', 'ETHNICITY', 'REGION', 'MEDS_BIRTH', 'LIFESTYLE')) or token_code.replace('.', '', 1).isdigit():
            return True
            
        # 2. Vitals (BMI, Height, Weight)
        # Check raw code text first (fastest)
        if 'BMI' in token_code: return True
        
        # Check translation for Height/Weight (handles cases where code is opaque like MEASUREMENT//123)
        # We translate just to check the string content
        trans = self._translate_token(token_code).upper()
        if 'HEIGHT' in trans or 'WEIGHT' in trans or 'BMI' in trans:
            return True
            
        return False

    def _translate_token(self, token_string):
        # This logic is the same as our narrative generator
        if not isinstance(token_string, str): return ""
        try:
            if token_string.startswith('<time_interval_'):
                time_part = token_string.split('_')[-1].strip('>')
                return f"{self.time_lookup.get(time_part, time_part)}; "
            elif token_string.startswith('AGE: ') or token_string.startswith('AGE'):
                return f"{token_string}; "
            elif token_string.startswith('MEDICAL//BMI'):
                return f"{token_string.split('//')[1]}; "
            elif token_string.startswith('MEDICAL//'):
                code = token_string.split('//')[1].upper()
                return f"{self.medical_lookup.get(code, code.replace('_', ' ').title())}; "
            elif token_string.startswith('MEASUREMENT//'):
                code = token_string.split('//')[1].upper()
                description = self.medical_lookup.get(code, code.replace('_', ' ').title())
                return f"{description}; "
            elif token_string.startswith('LAB//'):
                code = token_string.split('//')[1].upper()
                return f"{self.lab_lookup.get(code, code.replace('_', ' ').title())}; "
            # elif token_string.startswith(('BMI//', 'HEIGHT//', 'WEIGHT//')):
            #     return f"{token_string.split('//')[0]}: {token_string.split('//')[1]}"
            elif token_string.startswith(('GENDER//', 'ETHNICITY//')):
                parts = token_string.split('//')
                return f"Demographic {parts[0].title()} {parts[1].title()}; "
            elif token_string.startswith('REGION//'):
                parts = token_string.split('//')
                return f"{parts[0].title()} {self.region_lookup.get(parts[1], parts[1]).title()}; "
            elif token_string.startswith('LIFESTYLE//'):
                code = token_string.split('//')[1].upper()
                return f"{code.title()}; "
            elif token_string.replace('.', '', 1).isdigit():
                return f"{token_string}; "
            elif token_string.startswith('Q') and len(token_string) <= 4 and token_string[1:].isdigit():
                return f"{token_string[1:]}; "
            elif token_string.startswith('low') or token_string.startswith('normal') or token_string.startswith('high') or token_string.startswith('very low') or token_string.startswith('very high') and len(token_string) <= 9:
                return f"{token_string}; "
            elif token_string in ['<start>', '<end>', '<unknown>', 'MEDS_BIRTH']:
                return token_string + "; "
            else:
                return f"Unknown"
        except Exception as e:
            print(f"--- DEBUG: FAILED TO TRANSLATE TOKEN ---")
            print(f"Problematic Token String: '{token_string}'")
            print(f"Error: {e}")
            return "---ERROR_TRANSLATING---"

    def __len__(self):
        return len(self.patient_records)

    def __getitem__(self, idx):
        patient_record = self.patient_records[idx]
        subject_id = patient_record['subject_id']
        
        label = self.subject_to_label.get(subject_id)
        if pd.isna(label):
            return None # Skip patients without labels
            
        token_ids = patient_record['tokens']
        timestamps = patient_record['timestamps']
        
        # --- DYNAMIC TIME TRUNCATION ---
        # Truncate the patient timeline based on the cancer diagnosis date and the cutoff months
        # For pretrain format: always apply 1-month cutoff for cancer patients
        if self.format == 'pretrain' and label > 0:
            actual_cutoff = 1  # Always 1 month for pretraining
        elif self.cutoff_months is not None:
            actual_cutoff = self.cutoff_months
        else:
            actual_cutoff = None
        
        if actual_cutoff is not None:
            cancer_date = self.subject_to_cancer_date.get(subject_id)
            
            # Determine cutoff date - use cancer date if available, otherwise use last timestamp
            if pd.notna(cancer_date):
                cutoff_date = cancer_date - pd.DateOffset(months=actual_cutoff)
            else:
                # For controls or cases without cancer date, use the last timestamp as reference
                valid_timestamps = [ts for ts in timestamps if ts is not None and ts > 0]
                if valid_timestamps:
                    last_timestamp = max(valid_timestamps)
                    last_date = pd.Timestamp.fromtimestamp(last_timestamp)
                    cutoff_date = last_date - pd.DateOffset(months=actual_cutoff)
                else:
                    # No valid timestamps, skip truncation
                    cutoff_date = None
            
            if cutoff_date is not None:
                cutoff_timestamp = cutoff_date.timestamp()
                
                truncated_ids = []
                for i, ts in enumerate(timestamps):
                    token_str = self.id_to_token_map.get(token_ids[i], "")
                    # Keep: special tokens (ts==0), tokens before cutoff, OR the <end> token
                    is_end_token = (token_str == '<end>')
                    if ts == 0 or (ts is not None and ts < cutoff_timestamp) or is_end_token:
                        truncated_ids.append(token_ids[i])
                token_ids = truncated_ids
        
        
   
      
        string_codes = [self.id_to_token_map.get(tid, "") for tid in token_ids]
        translated_phrases = []

        SPLIT_TOKEN = " <HEADER_SPLIT> "
        split_inserted = False

        i = 0
        while i < len(string_codes):
            current_code = str(string_codes[i])

            # Insert the split token if we haven't already and the current token is not a header concept
            if not split_inserted and not self._is_header_concept(current_code):
                translated_phrases.append(SPLIT_TOKEN)
                split_inserted = True
            
            # Check if the current token is a measurable concept
            is_measurable = current_code.startswith(('LAB//', 'MEASUREMENT//', 'MEDICAL//BMI', 'MEDICAL//bp_'))
            has_next_token = (i + 1 < len(string_codes))
            is_next_a_quantile = False
            
            if has_next_token:
                next_code = str(string_codes[i+1])
                # Check if next token is a number (e.g., "45.5", "12")
                safe_code = str(next_code)
                is_numeric = safe_code.replace('.', '', 1).isdigit()
                is_next_a_value = is_numeric
                
            # If we have a measurable concept AND its quantile value, combine them
            if is_measurable and is_next_a_value:
                concept = self._translate_token(current_code) 
                value_bin = self._translate_token(next_code)  
                
                # --- NEW LOGIC TO HANDLE UNITS ---
                unit_str = ""
                increment = 2 # Default: skip Code and Value
                
                # Check if there is a 3rd token (Potential Unit)
                if i + 2 < len(str(string_codes)):
                    potential_unit = str(string_codes[i+2])
                    
                    # A unit is anything that is NOT a new event code or special token
                    # Add any other prefixes you want to exclude here
                    is_new_event = str(potential_unit).startswith((
                        'LAB//', 'MEDICAL//', 'MEASUREMENT//', 
                        'AGE:', 'AGE', 'GENDER//', 'ETHNICITY//', 'REGION//', 
                        'LIFESTYLE//', '<start>', '<end>', '<time', 'MEDS_BIRTH'
                    ))
                    
                    if not is_new_event:
                        # It's a unit! (e.g., "mmol/L", "kg")
                        unit_str = f" {potential_unit}"
                        increment = 3 # Skip Code, Value, AND Unit
                
                if concept and value_bin: 
                    # Remove the trailing "; " from value_bin when we have a unit
                    # This prevents "3.5; %" and makes it "3.5 %"
                    if unit_str:
                        # Remove "; " from both concept and value_bin, then combine with unit
                        concept_clean = concept.rstrip('; ').strip()
                        value_clean = value_bin.rstrip('; ').strip()
                        translated_phrases.append(f"{concept_clean} {value_clean} {unit_str.lstrip()}; ")
                    else:
                        # No unit, keep the original format
                        translated_phrases.append(f"{concept} {value_bin}") 
                
                i += increment
                # ---------------------------------
            
            # Otherwise, just translate the single token as normal
            else:
                phrase = self._translate_token(current_code)
                if phrase: 
                    translated_phrases.append(phrase)
                
                i += 1 
        
        narrative = "".join(translated_phrases)
        
        return {
            "text": narrative,
            "label": torch.tensor(label, dtype=torch.long)
        }
       