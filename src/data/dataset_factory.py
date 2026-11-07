'''
Dataset factory: single entry point for creating any EHR dataset.
'''
from typing import Dict, Any
from torch.utils.data import Dataset

from src.data.cprd_dataset import CPRDDataset
from src.data.mimic_dataset import MIMICDataset

ACES_DATASETS = frozenset({"MIMIC"})


def is_aces_dataset(dataset_name: str) -> bool:
    """MIMIC uses ACES parquet labels and a prediction-time cutoff."""
    return str(dataset_name).upper() in ACES_DATASETS


def get_dataset(
    dataset_name: str,
    data_config: Dict[str, Any],
    split: str,
) -> Dataset:
    '''
    Create and return the appropriate dataset for a given split.

    Args:
    - dataset_name: 'CPRD' or 'MIMIC'
    - data_config: The 'data' section from a yaml config dict
    - split: 'train', 'tuning', 'held_out'
    '''
    if dataset_name == 'CPRD':
        return CPRDDataset(
            data_dir=data_config['data_dir'],
            vocab_file=data_config['vocab_filepath'],
            labels_file=data_config['labels_filepath'],
            medical_lookup_file=data_config['medical_lookup_filepath'],
            lab_lookup_file=data_config['lab_lookup_filepath'],
            region_lookup_file=data_config['region_lookup_filepath'],
            time_lookup_file=data_config['time_lookup_filepath'],
            cutoff_months=data_config.get('cutoff_months', None),
            split=split,
            format=data_config.get('format', 'text'),
            drop_time=bool(data_config.get('drop_time', False)),
            drop_values=bool(data_config.get('drop_values', False)),
            max_events=data_config.get('max_events'),
        )
    if dataset_name == 'MIMIC':
        return MIMICDataset(
            data_dir=data_config['data_dir'],
            vocab_file=data_config['vocab_filepath'],
            aces_label_dir=data_config.get('aces_label_dir'),
            lookup_dir=data_config['lookup_dir'],
            split=split,
            label_column=data_config.get('label_column', 'label'),
            task=data_config.get('task', 'pretrain'),
            format=data_config.get('format', 'text'),
            surface_form=data_config.get('surface_form', 'text'),
            scramble_seed=int(data_config.get('scramble_seed', 42)),
            drop_time=bool(data_config.get('drop_time', False)),
            drop_values=bool(data_config.get('drop_values', False)),
            max_events=data_config.get('max_events'),
            file_fraction=data_config.get('file_fraction'),
            file_fraction_seed=int(data_config.get('file_fraction_seed', 42)),
        )
    raise ValueError(f"Unsupported dataset type '{dataset_name}' in data_config.")
