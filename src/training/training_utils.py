"""
Utility functions for training.
"""
from transformers import set_seed as hf_set_seed
import random
import numpy as np
import torch
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from datasets import Dataset
import logging
logger = logging.getLogger(__name__)

def setup_wandb(config):
    """
    Setup WandB logging.
    """
    wandb.init(
        project=config.get("project", "ehr-llm-pretraining"),
        name=config.get("name", "ehr-llm-pretraining"),
        config=config
    )

def extract_and_pack(dataset, tokenizer, max_length, cache_path):
    """
    Extracts all valid text narratives from a dataset for pretraining and packs them into chunks.
    """
    logger.info(f"  - Extracting and packing {len(dataset)} texts to {max_length} tokens...")
    
    # Extract text
    texts = extract_text(dataset, tokenizer)
    # Pack text
    packed_texts = pack_and_chunk_texts(texts, tokenizer, max_length)
    # Save to cache
    Dataset.from_dict({"input_ids": packed_texts}).save_to_disk(cache_path)
    logger.info(f"  - Saved {len(packed_texts)} packed sequences to {cache_path}")
    del texts, packed_texts # Clear memory immediately
    
   

def extract_text(base_dataset, tokenizer) -> List[str]:
    """
    Extracts all valid text narratives from a dataset for pretraining.
    
    This function extracts text from UnifiedEHRDataset and prepares it for SFTTrainer.
    We keep both <start> and <end> tokens as they mark sequence boundaries:
    - <start> marks the beginning of a patient record
    - <end> marks the end of a patient record (the model should learn to predict this)
    
    With packing=True in SFTTrainer, these tokens help the model understand
    where one patient record ends and another begins.
    
    Args:
        base_dataset: The dataset to extract text from.
        tokenizer: The tokenizer (unused but kept for consistency).
    
    Returns:
        List of text strings ready for training.
    """
    text_list = []
    
    # Get EOS token from tokenizer
    eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
    
    total_patients = len(base_dataset)
    logging.info(f"  - Processing {total_patients} patients...")
    if eos_token:
        logging.info(f"  - Adding EOS token '{eos_token}' to each sequence")
    else:
        logging.warning("  ⚠️  WARNING: Tokenizer has no EOS token!")
    
    # Use tqdm for progress bar
    for i in tqdm(range(total_patients), desc="  Extracting text", unit="patient"):
        item = base_dataset[i]
        if item is not None:
            text = item['text']
            # Remove <end> token (if still present)
            text = text.replace('<end>', '').replace(' <HEADER_SPLIT> ', '')
            
            # Clean up leading "; "
            if text.startswith('; '):
                text = text[2:]
            
            # Add EOS token at the end of each sequence
            if eos_token and not text.endswith(eos_token):
                text = text + eos_token
            
            text_list.append(text)
    
    logging.info(f"  ✓ Extracted {len(text_list)} valid narratives (skipped {total_patients - len(text_list)} None items)")
    return text_list

def pack_and_chunk_texts(texts, tokenizer, chunk_size):
    """
    Optimized packing and chunking to prevent CPU RAM OOM.
    """
    print(f"  - Tokenizing and packing {len(texts)} texts into {chunk_size}-token chunks...")
    
    # 1. Use the tokenizer's batch capability (much faster and more memory efficient)
    # We do NOT return tensors here to keep memory low
    tokenized = tokenizer(
        texts, 
        add_special_tokens=False, 
        truncation=False, 
        padding=False
    )["input_ids"]

    # 2. Flatten efficiently
    all_token_ids = []
    eos_id = tokenizer.eos_token_id
    for ids in tokenized:
        all_token_ids.extend(ids)
        if not ids or ids[-1] != eos_id:
            all_token_ids.append(eos_id)
            
    # Free up the intermediate list immediately
    del tokenized 
    
    # 3. Slice into fixed chunks
    chunked_input_ids = [
        all_token_ids[i : i + chunk_size] 
        for i in range(0, len(all_token_ids), chunk_size)
        if (i + chunk_size) <= len(all_token_ids) # Drop partial final chunk for stability
    ]
                
    print(f"  - Created {len(chunked_input_ids)} packed sequences.")
    return chunked_input_ids

def compute_and_sort_by_length(dataset, tokenizer, shuffle_buckets=True, num_buckets=10):
    """
    Sort dataset by sequence length for more efficient batching.
    
    Longer sequences together = less padding waste.
    Optional bucketed shuffling maintains some randomness.
    
    Args:
        dataset: Dataset to sort.
        tokenizer: Tokenizer to compute sequence lengths.
        shuffle_buckets: If True, shuffle within buckets to maintain randomness.
        num_buckets: Number of buckets for shuffling.
    
    Returns:
        Sorted dataset (as torch.utils.data.Subset).
    """
    logging.info(f"  - Computing lengths for {len(dataset)} samples...")
    
    # Compute lengths
    lengths = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is not None and 'text' in sample:
            length = len(tokenizer.encode(sample['text'], add_special_tokens=True))
            lengths.append((idx, length))
        else:
            lengths.append((idx, 0))
    
    # Sort by length
    lengths.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in lengths]
    
    # Optional: Shuffle within buckets to maintain some randomness
    if shuffle_buckets:
        bucket_size = len(sorted_indices) // num_buckets
        bucketed_indices = []
        for i in range(num_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < num_buckets - 1 else len(sorted_indices)
            bucket = sorted_indices[start:end]
            np.random.shuffle(bucket)
            bucketed_indices.extend(bucket)
        sorted_indices = bucketed_indices
    
    # Create sorted dataset
    sorted_dataset = torch.utils.data.Subset(dataset, sorted_indices)
    
    lengths_only = [l for _, l in lengths]
    logging.info(f"  - Length range: {min(lengths_only)} to {max(lengths_only)} tokens")
    logging.info(f"  - Mean length: {np.mean(lengths_only):.0f} tokens")
    
    return sorted_dataset

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
