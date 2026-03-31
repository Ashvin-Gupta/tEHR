import os
import sys
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# Data processing
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch and transformers
import torch
from transformers import AutoTokenizer

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Add project root to path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("✓ Imports complete")

# Configuration - paths from llm_pretrain.yaml
DATA_DIR = "/gpfs/scratch/qc25022/pancreas/tokenised_data_pancreas_MEDS_new_time/cprd_upgi"
VOCAB_FILE = "/gpfs/scratch/qc25022/pancreas/tokenised_data_pancreas_MEDS_new_time/cprd_upgi/vocab.csv"
LABELS_FILE = "/data/home/qc25022/cancer-extraction-pipeline/output/pancreas_MEDS/subject_information.csv"
MEDICAL_LOOKUP = "/data/home/qc25022/TextCancEHR2/src/resources/MedicalDictTranslation2.csv"
LAB_LOOKUP = "/data/home/qc25022/TextCancEHR2/src/resources/LabLookUP.csv"
REGION_LOOKUP = "src/resources/RegionLookUp.csv"
TIME_LOOKUP = "src/resources/TimeLookUp.csv"

# Model configuration
MODEL_NAME = "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit"

# Splits to analyze
SPLITS = ['train', 'tuning', 'held_out']

print("✓ Configuration loaded")

# Load vocabulary and mappings
print("Loading vocabulary and mappings...")

# Vocabulary
vocab_df = pd.read_csv(VOCAB_FILE, dtype={'str': str})
id_to_token_map = pd.Series(vocab_df['str'].values, index=vocab_df['token']).to_dict()
token_to_id_map = {v: k for k, v in id_to_token_map.items()}
print(f"  ✓ Vocabulary: {len(id_to_token_map)} tokens")

# Labels
labels_df = pd.read_csv(LABELS_FILE)
labels_df['string_label'] = labels_df.apply(
    lambda row: 'Control' if row['is_case'] == 0 else row['site'],
    axis=1
)
unique_labels = sorted([l for l in labels_df['string_label'].unique() if l != 'Control'])
label_to_id_map = {label: i + 1 for i, label in enumerate(unique_labels)}
label_to_id_map['Control'] = 0
id_to_label_map = {v: k for k, v in label_to_id_map.items()}

labels_df['label_id'] = labels_df['string_label'].map(label_to_id_map)
subject_to_label = pd.Series(labels_df['label_id'].values, index=labels_df['subject_id']).to_dict()
subject_to_string_label = pd.Series(labels_df['string_label'].values, index=labels_df['subject_id']).to_dict()
print(f"  ✓ Labels: {len(labels_df)} patients, {len(unique_labels)+1} classes")
print(f"    Classes: {list(label_to_id_map.keys())}")

# Print counts for each label
label_counts = labels_df['string_label'].value_counts()
print("    Label counts per class:")
for label, count in label_counts.items():
    print(f"      {label}: {count}")

# Cancer dates
labels_df['cancerdate'] = pd.to_datetime(labels_df['cancerdate'], errors='coerce')
subject_to_cancer_date = pd.Series(labels_df['cancerdate'].values, index=labels_df['subject_id']).to_dict()

# Lookup tables for translation
medical_df = pd.read_csv(MEDICAL_LOOKUP)
medical_lookup = pd.Series(medical_df['term'].values, index=medical_df['code'].astype(str).str.upper()).to_dict()
print(f"  ✓ Medical lookup: {len(medical_lookup)} codes")

lab_df = pd.read_csv(LAB_LOOKUP)
lab_lookup = pd.Series(lab_df['term'].values, index=lab_df['code'].astype(str).str.upper()).to_dict()
print(f"  ✓ Lab lookup: {len(lab_lookup)} codes")

region_df = pd.read_csv(REGION_LOOKUP)
region_lookup = pd.Series(region_df['Description'].values, index=region_df['regionid'].astype(str).str.upper()).to_dict()
print(f"  ✓ Region lookup: {len(region_lookup)} regions")

time_df = pd.read_csv(TIME_LOOKUP)
time_lookup = pd.Series(time_df['term'].values, index=time_df['code'].astype(str).str.upper()).to_dict()
print(f"  ✓ Time lookup: {len(time_lookup)} intervals")

print("\n✓ All mappings loaded successfully")

def load_patient_records(data_dir: str, split: str) -> List[Dict]:
    """
    Load patient records from pickle files for a given split.
    
    Args:
        data_dir: Base directory containing split subdirectories
        split: Split name (train, tuning, held_out)
    
    Returns:
        List of patient record dictionaries
    """
    split_dir = os.path.join(data_dir, split)
    records = []
    
    pkl_files = [
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith('.pkl')
    ]
    
    print(f"Loading {split} split: {len(pkl_files)} pickle files")
    
    for file_path in tqdm(pkl_files, desc=f"  Loading {split}"):
        with open(file_path, 'rb') as f:
            records.extend(pickle.load(f))
    
    print(f"  ✓ Loaded {len(records)} patient records\n")
    return records

# Load all splits
print("="*60)
print("LOADING PATIENT RECORDS FROM ALL SPLITS")
print("="*60 + "\n")

patient_records = {}
for split in SPLITS:
    patient_records[split] = load_patient_records(DATA_DIR, split)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for split in SPLITS:
    print(f"{split:12s}: {len(patient_records[split]):,} patients")
print(f"{'TOTAL':12s}: {sum(len(patient_records[s]) for s in SPLITS):,} patients")
print("="*60)

# ------------------------------------------------------------------
# Sanity check for MEDICAL//22J..00 occurrences
# ------------------------------------------------------------------
TARGET_TOKEN = "MEDICAL//22J..00"

med22_records = []

for split in SPLITS:
    for record in patient_records[split]:
        subject_id = record['subject_id']
        token_ids = record['tokens']
        token_strings = [id_to_token_map.get(tid, "") for tid in token_ids]

        # All positions where this code appears
        indices = [i for i, t in enumerate(token_strings) if t == TARGET_TOKEN]
        if not indices:
            continue

        # Use first occurrence to summarise "where" in the trajectory
        first_idx = indices[0]
        seq_len = len(token_strings)
        rel_pos = first_idx / max(1, seq_len - 1)  # 0 = very early, 1 = very late

        # Try to find the nearest preceding TIME bucket token
        time_bucket = None
        for j in range(first_idx, -1, -1):
            t = token_strings[j]
            if isinstance(t, str) and t.startswith("TIME//"):
                time_bucket = t
                break

        med22_records.append({
            'subject_id': subject_id,
            'split': split,
            'label': subject_to_string_label.get(subject_id, 'Unknown'),
            'num_tokens': seq_len,
            'first_index': first_idx,
            'first_rel_pos': rel_pos,
            'num_occurrences': len(indices),
            'first_time_bucket': time_bucket,
        })

med22_df = pd.DataFrame(med22_records)

print("\n" + "="*60)
print(f"CHECKING FOR {TARGET_TOKEN}")
print("="*60)

if med22_df.empty:
    print(f"No occurrences of {TARGET_TOKEN} found in any split.")
else:
    print(f"Total patient-split records with {TARGET_TOKEN}: {len(med22_df):,}")
    print(f"Unique patients with {TARGET_TOKEN}: {med22_df['subject_id'].nunique():,}")
    print("\nUnique patients by split (subject-level):")
    print(med22_df.groupby('split')['subject_id'].nunique())

    print("\nUnique patients by cancer/control label:")
    print(med22_df.groupby('label')['subject_id'].nunique())

    print("\nFirst-occurrence relative position in trajectory (0=very early, 1=very late):")
    print(med22_df['first_rel_pos'].describe())

    print("\nCounts by first TIME bucket immediately before the code (if any):")
    print(med22_df['first_time_bucket'].value_counts(dropna=False).head(20))

    # Show a few concrete examples with local context around the code
    print("\nExample contexts around MEDICAL//22J..00 (up to 5 patients):")
    for _, row in med22_df.head(5).iterrows():
        split = row['split']
        subject_id = row['subject_id']
        label = row['label']

        # Reconstruct this patient's tokens
        record = next(r for r in patient_records[split] if r['subject_id'] == subject_id)
        token_ids = record['tokens']
        token_strings = [id_to_token_map.get(tid, "") for tid in token_ids]
        indices = [i for i, t in enumerate(token_strings) if t == TARGET_TOKEN]

        print("-" * 40)
        print(f"Split: {split} | subject_id: {subject_id} | label: {label}")
        print(f"Sequence length: {len(token_strings)} | Occurrences: {len(indices)}")

        for idx in indices:
            start = max(0, idx - 5)
            end = min(len(token_strings), idx + 6)
            window = token_strings[start:end]
            print(f"  Index {idx} (window {start}:{end}):")
            print("   ", window)

print("="*60)


import textwrap

# Map notebook split names to table split labels
split_label_map = {
    "train": "Train",
    "tuning": "Validation",
    "held_out": "Test",
}

# Helper: format n (percent) for counts
def fmt_n_pct(n, total):
    if total == 0 or pd.isna(total):
        return "0 (0.0\\%)"
    pct = 100.0 * n / total
    return f"{int(n)} ({pct:.1f}\\%)"

# Helper: format median [IQR]
def fmt_med_iqr(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "XXX [X--X]"
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    # Age is already in years; token counts are integers
    return f"{q2:.1f} [{q1:.1f}--{q3:.1f}]"

# Build a working DataFrame keyed by (split, is_case)
# We will join demographics and trajectory statistics.

# Build a patient-level analysis frame directly from loaded records.
# This keeps the script standalone (no notebook globals required).
analysis_rows = []
for split in SPLITS:
    for record in patient_records[split]:
        subject_id = record["subject_id"]
        token_ids = record["tokens"]
        token_strings = [id_to_token_map.get(tid, "") for tid in token_ids]

        label_id = subject_to_label.get(subject_id, -1)
        is_case = 1 if label_id > 0 else 0

        age = None
        gender = None
        for i, token in enumerate(token_strings):
            if not isinstance(token, str):
                continue

            if token.startswith("GENDER//") and gender is None:
                gender = token.split("//")[-1]

            if age is None and (token == "AGE" or token.startswith("AGE:") or token.startswith("AGE ")):
                if ":" in token:
                    age_str = token.split(":")[-1].strip()
                    try:
                        age = float(age_str)
                    except ValueError:
                        pass
                elif token.startswith("AGE "):
                    age_str = token.replace("AGE", "").strip()
                    try:
                        age = float(age_str)
                    except ValueError:
                        pass
                elif i + 1 < len(token_strings):
                    next_token = str(token_strings[i + 1])
                    if next_token.replace(".", "", 1).replace("-", "", 1).isdigit():
                        age = float(next_token)

        num_medical = sum(1 for t in token_strings if isinstance(t, str) and t.startswith("MEDICAL//"))
        num_lab = sum(1 for t in token_strings if isinstance(t, str) and t.startswith("LAB//"))

        analysis_rows.append(
            {
                "subject_id": subject_id,
                "split": split,
                "is_case": is_case,
                "gender": gender,
                "age": age,
                "num_medical": num_medical,
                "num_lab": num_lab,
                "trajectory_length_tokens": len(token_ids),
            }
        )

merged = pd.DataFrame(analysis_rows)

# Sanity check: subject/split combinations should line up
# (we won't enforce strict equality here, but will use left joins by subject+split)

# Convenience: harmonise split names
merged["split_label"] = merged["split"].map(split_label_map)

# Precompute total N for denominator by (split, is_case)
N_by_group = merged.groupby(["split_label", "is_case"]).size().to_dict()

# Helper to get a subset for a given split label and case/control
def subset(split_label, is_case):
    return merged[(merged["split_label"] == split_label) & (merged["is_case"] == is_case)]

# Build table entries
rows = {
    "total_patients": {},
    "gender_male": {},
    "gender_female": {},
    "age": {},
    "medcodes": {},
    "labs": {},
    "trajectory_len": {},
}

for split_label in ["Train", "Validation", "Test"]:
    for is_case, col_case in [(1, "Case"), (0, "Control")]:
        key = (split_label, col_case)
        df = subset(split_label, is_case)
        N = len(df)

        # Total patients
        rows["total_patients"][key] = f"{N}"

        # Gender (assuming `gender` is coded 'Male'/'Female' or similar)
        male_n = (df["gender"].str.lower() == "male").sum()
        female_n = (df["gender"].str.lower() == "female").sum()
        rows["gender_male"][key] = fmt_n_pct(male_n, N)
        rows["gender_female"][key] = fmt_n_pct(female_n, N)

        # Age at index (years)
        rows["age"][key] = fmt_med_iqr(df["age"])  # already in years

        # Clinical events counts
        rows["medcodes"][key] = fmt_med_iqr(df["num_medical"])  # total MedCodes
        rows["labs"][key] = fmt_med_iqr(df["num_lab"])          # lab observations

        # Trajectory length in tokens
        rows["trajectory_len"][key] = fmt_med_iqr(df["trajectory_length_tokens"])

# Helper to access a cell for printing
def cell(row_key, split_label, case_label):
    return rows[row_key][(split_label, case_label)]

# Now print the LaTeX body corresponding to the provided table
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{CPRD Patient Demographics and Clinical Event Statistics by Dataset Split. Values are reported as Count (\\%) or Median [IQR].}")
print("\\label{tab:cprd_stats}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{lcccccc}")
print("\\toprule")
print("\\multicolumn{2}{c}{\\textbf{Train}} & \\multicolumn{2}{c}{\\textbf{Validation}} & \\multicolumn{2}{c}{\\textbf{Test}} \\\\")
print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
print("\\textbf{Feature} & \\textbf{Case} & \\textbf{Control} & \\textbf{Case} & \\textbf{Control} & \\textbf{Case} & \\textbf{Control} \\\\")
print("\\midrule")

# Total patients (these should match previously printed counts)
print(
    "Total Patients ($N$) & "
    f"{cell('total_patients', 'Train', 'Case')} & {cell('total_patients', 'Train', 'Control')} & "
    f"{cell('total_patients', 'Validation', 'Case')} & {cell('total_patients', 'Validation', 'Control')} & "
    f"{cell('total_patients', 'Test', 'Case')} & {cell('total_patients', 'Test', 'Control')} \\\\")
print("\\midrule")

# Gender rows
print("\\textbf{Gender} & & & & & & \\\\")
print(
    "\\quad Male & "
    f"{cell('gender_male', 'Train', 'Case')} & {cell('gender_male', 'Train', 'Control')} & "
    f"{cell('gender_male', 'Validation', 'Case')} & {cell('gender_male', 'Validation', 'Control')} & "
    f"{cell('gender_male', 'Test', 'Case')} & {cell('gender_male', 'Test', 'Control')} \\\\")
print(
    "\\quad Female & "
    f"{cell('gender_female', 'Train', 'Case')} & {cell('gender_female', 'Train', 'Control')} & "
    f"{cell('gender_female', 'Validation', 'Case')} & {cell('gender_female', 'Validation', 'Control')} & "
    f"{cell('gender_female', 'Test', 'Case')} & {cell('gender_female', 'Test', 'Control')} \\\\")
print("\\midrule")

# Age at index
print(
    "\\textbf{Age at Index (Years)} & "
    f"{cell('age', 'Train', 'Case')} & {cell('age', 'Train', 'Control')} & "
    f"{cell('age', 'Validation', 'Case')} & {cell('age', 'Validation', 'Control')} & "
    f"{cell('age', 'Test', 'Case')} & {cell('age', 'Test', 'Control')} \\\\")
print("\\midrule")

# Clinical events
print("\\textbf{Clinical Events (Count)} & & & & & & \\\\")
print(
    "\\quad Total MedCodes & "
    f"{cell('medcodes', 'Train', 'Case')} & {cell('medcodes', 'Train', 'Control')} & "
    f"{cell('medcodes', 'Validation', 'Case')} & {cell('medcodes', 'Validation', 'Control')} & "
    f"{cell('medcodes', 'Test', 'Case')} & {cell('medcodes', 'Test', 'Control')} \\\\")
print(
    "\\quad Lab Observations & "
    f"{cell('labs', 'Train', 'Case')} & {cell('labs', 'Train', 'Control')} & "
    f"{cell('labs', 'Validation', 'Case')} & {cell('labs', 'Validation', 'Control')} & "
    f"{cell('labs', 'Test', 'Case')} & {cell('labs', 'Test', 'Control')} \\\\")
print("\\midrule")

# Trajectory length
print(
    "\\textbf{Trajectory Length} & "
    f"{cell('trajectory_len', 'Train', 'Case')} & {cell('trajectory_len', 'Train', 'Control')} & "
    f"{cell('trajectory_len', 'Validation', 'Case')} & {cell('trajectory_len', 'Validation', 'Control')} & "
    f"{cell('trajectory_len', 'Test', 'Case')} & {cell('trajectory_len', 'Test', 'Control')} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")