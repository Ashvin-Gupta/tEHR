"""
Utility functions for training.
"""
from transformers import set_seed as hf_set_seed
import random
import numpy as np
import torch


def seed_all(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def print_sample_translations(config: dict, num_samples: int = 4):
    """
    Print sample patient translations to verify data quality before training.
    
    Shows the first and last 1000 characters of translated text for randomly
    sampled patients (2 controls and 2 cases) to verify the data pipeline is working correctly.
    
    Args:
        config: Configuration dictionary containing data paths
        num_samples: Number of patients to sample (default: 4, split between controls and cases)
    """
    from src.data.unified_dataset_v2 import UnifiedEHRDataset
    
    print("\n" + "=" * 80)
    print("SAMPLE PATIENT TRANSLATIONS")
    print("=" * 80)
    print(f"Showing first and last 1000 characters for {num_samples} patients (2 controls, 2 cases)\n")
    
    data_config = config['data']
    
    # Load a small sample from train split
    dataset = UnifiedEHRDataset(
        data_dir=data_config['data_dir'],
        vocab_file=data_config['vocab_filepath'],
        labels_file=data_config['labels_filepath'],
        medical_lookup_file=data_config['medical_lookup_filepath'],
        lab_lookup_file=data_config['lab_lookup_filepath'],
        region_lookup_file=data_config['region_lookup_filepath'],
        time_lookup_file=data_config['time_lookup_filepath'],
        format='text',
        split='train',
        cutoff_months=data_config.get('cutoff_months', 1),
        data_type=config['training'].get('input_data', 'binned')
    )
    
    print(f"Loaded dataset with {len(dataset)} patients\n")
    
    # Separate indices by label
    control_indices = []
    case_indices = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is not None:
            label = sample['label'].item()
            if label == 0:
                control_indices.append(idx)
            else:
                case_indices.append(idx)
    
    print(f"Found {len(control_indices)} controls and {len(case_indices)} cases\n")
    
    # Sample 2 from each group
    num_per_group = num_samples // 2
    sampled_controls = random.sample(control_indices, min(num_per_group, len(control_indices)))
    sampled_cases = random.sample(case_indices, min(num_per_group, len(case_indices)))
    
    # Combine samples (controls first, then cases)
    sample_indices = sampled_controls + sampled_cases
    sample_labels = ['Control'] * len(sampled_controls) + ['Case'] * len(sampled_cases)
    
    for i, (idx, label_type) in enumerate(zip(sample_indices, sample_labels), 1):
        print("=" * 80)
        print(f"PATIENT SAMPLE {i} - {label_type.upper()} (Index: {idx})")
        print("=" * 80)
        
        sample = dataset[idx]
        if sample is None:
            print("  (No data available for this patient)\n")
            continue
        
        text = sample['text']
        label = sample['label'].item()
        
        print(f"Label: {label} ({label_type})")
        print(f"Total Text Length: {len(text):,} characters")
        print(f"Total Text Length: {len(text.split()):,} words (approx)")
        
        print("\n" + "-" * 80)
        print("FIRST 1000 CHARACTERS:")
        print("-" * 80)
        first_1000 = text[:1000]
        print(first_1000)
        if len(text) > 1000:
            print("...")
        
        print("\n" + "-" * 80)
        print("LAST 1000 CHARACTERS:")
        print("-" * 80)
        if len(text) > 1000:
            last_1000 = text[-1000:]
            print("...")
            print(last_1000)
        else:
            print("(Text is shorter than 1000 characters, see above)")
        
        print("\n")
    
    print("=" * 80)
    print("END OF SAMPLE TRANSLATIONS")
    print("=" * 80 + "\n")
