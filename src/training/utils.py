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


def print_sample_translations(config: dict, num_samples: int = 3):
    """
    Print sample patient translations to verify data quality before training.
    
    Shows the first and last 1000 characters of translated text for randomly
    sampled patients to verify the data pipeline is working correctly.
    
    Args:
        config: Configuration dictionary containing data paths
        num_samples: Number of patients to sample (default: 3)
    """
    from src.data.unified_dataset_v2 import UnifiedEHRDataset
    
    print("\n" + "=" * 80)
    print("SAMPLE PATIENT TRANSLATIONS")
    print("=" * 80)
    print(f"Showing first and last 1000 characters for {num_samples} patients\n")
    
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
    
    # Sample patients randomly
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(sample_indices, 1):
        print("=" * 80)
        print(f"PATIENT SAMPLE {i} (Index: {idx})")
        print("=" * 80)
        
        sample = dataset[idx]
        if sample is None:
            print("  (No data available for this patient)\n")
            continue
        
        text = sample['text']
        label = sample['label'].item()
        
        print(f"Label: {label}")
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
