"""
Dataset utility functions for data preprocessing and manipulation.
"""
import numpy as np
import torch


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
    print(f"  - Computing lengths for {len(dataset)} samples...")
    
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
    print(f"  - Length range: {min(lengths_only)} to {max(lengths_only)} tokens")
    print(f"  - Mean length: {np.mean(lengths_only):.0f} tokens")
    
    return sorted_dataset
