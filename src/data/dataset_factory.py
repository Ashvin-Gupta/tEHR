'''
Dataset factory: single entry point for creating any EHR dataset
'''
from typing import Dict, Any
from torch.utils.data import Dataset

from src.data.cprd_dataset import CPRDDataset
from src.data.mimic_dataset import MIMICDataset
from src.data.ehrshot_dataset import EHRSHOTDataset

def get_dataset(
    dataset_name: str,
    data_config: Dict[str, Any],
    split:str,
) -> Dataset:
    '''
    Create and return the approprate dataset for a given split

    Args:
    - dataset_name: 'CPRD' or 'MIMIC'
    - data_config: The 'data'section from a yaml config dict
    - split: 'train', 'tuning', 'held_out'

    Returns:
    - Initialised Dataset ready for use with a dataloader
    '''
    if dataset_name == 'CPRD':
        return CPRDDataset(
            data_dir = data_config['data_dir'],
            vocab_file = data_config['vocab_filepath'],
            labels_file = data_config['labels_filepath'],
            medical_lookup_file = data_config['medical_lookup_filepath'],
            lab_lookup_file = data_config['lab_lookup_filepath'],
            region_lookup_file = data_config['region_lookup_filepath'],
            time_lookup_file = data_config['time_lookup_filepath'],
            cutoff_months = data_config.get('cutoff_months', None),
            split = split,
            format = data_config.get('format', 'text'), # Only needed as custom transformer uses tokens only
        )
    elif dataset_name == 'MIMIC':
        return MIMICDataset(
            data_dir = data_config['data_dir'],
            vocab_file = data_config['vocab_filepath'],
            aces_label_dir = data_config['aces_label_dir'],
            lookup_dir = data_config['lookup_dir'],
            split = split,
            label_column = data_config.get('label_column', 'label'),
            task = data_config.get('task', 'pretrain'),
        )
    elif dataset_name == 'EHRSHOT':
        return EHRSHOTDataset(
            csv_filepath = data_config['csv_filepath'],
            split = split,
        )
    else:
        raise ValueError(f"Unsupported dataset type '{dataset_name}' in data_config.")