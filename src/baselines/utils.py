"""
Shared utilities for baseline model implementations.
"""
import os
import yaml
from typing import Dict, Any, List, Tuple
from src.data.unified_dataset import UnifiedEHRDataset


def load_baseline_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate baseline configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config missing required section: {section}")
    
    return config


def setup_output_dir(output_dir: str, overwrite: bool = False) -> str:
    """
    Create output directory structure for baseline results.
    
    Args:
        output_dir: Base output directory
        overwrite: Whether to overwrite existing directory
    
    Returns:
        Path to created output directory
    """
    if os.path.exists(output_dir) and not overwrite:
        print(f"Output directory exists: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Create subdirectories
    subdirs = ['plots', 'models', 'results']
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
    
    return output_dir


def load_datasets(
    data_config: Dict[str, Any],
    splits: List[str] = ['train', 'tuning', 'held_out'],
    format: str = 'text',
    tokenizer: Any = None
) -> Dict[str, UnifiedEHRDataset]:
    """
    Load datasets for specified splits.
    
    Args:
        data_config: Data configuration dictionary
        splits: List of splits to load
        format: Dataset format ('tokens', 'text', etc.)
        tokenizer: Optional tokenizer (for certain formats)
    
    Returns:
        Dictionary mapping split names to datasets
    """
    datasets = {}
    
    for split in splits:
        print(f"\nLoading {split} dataset...")
        dataset = UnifiedEHRDataset(
            data_dir=data_config['data_dir'],
            vocab_file=data_config['vocab_filepath'],
            labels_file=data_config['labels_filepath'],
            medical_lookup_file=data_config['medical_lookup_filepath'],
            lab_lookup_file=data_config['lab_lookup_filepath'],
            region_lookup_file=data_config['region_lookup_filepath'],
            time_lookup_file=data_config['time_lookup_filepath'],
            cutoff_months=data_config.get('cutoff_months', None),
            max_sequence_length=data_config.get('max_length', 512),
            format=format,
            split=split,
            tokenizer=tokenizer,
            data_type=data_config.get('data_type', 'binned')
        )
        datasets[split] = dataset
        print(f"  - Loaded {len(dataset)} samples")
    
    return datasets


def get_labels_from_dataset(dataset: UnifiedEHRDataset) -> Tuple[list, list]:
    """
    Extract labels and subject IDs from dataset.
    
    Args:
        dataset: UnifiedEHRDataset instance
    
    Returns:
        Tuple of (labels_list, subject_ids_list)
    """
    labels = []
    subject_ids = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is not None:
            label = sample['label'].item() if hasattr(sample['label'], 'item') else sample['label']
            labels.append(label)
            subject_id = sample.get('subject_id', i)
            subject_ids.append(subject_id)
    
    return labels, subject_ids

