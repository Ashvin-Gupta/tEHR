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