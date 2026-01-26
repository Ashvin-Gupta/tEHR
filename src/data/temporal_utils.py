"""
Utilities for aligning temporal information with tokenized sequences.

When a tokenizer splits text into sub-tokens, we need to map the original
event-level delta times to each sub-token.
"""
import torch
from typing import List, Optional
import numpy as np


def compute_delta_times(timestamps: List[float]) -> List[float]:
    """
    Compute delta times from absolute timestamps.
    
    Args:
        timestamps: List of absolute timestamps (Unix timestamps in seconds).
                   May contain 0 for special tokens or negative values (data errors).
    
    Returns:
        List of delta times where delta_times[i] = timestamps[i] - timestamps[i-1].
        First event has delta_time = 0.
        Negative or zero timestamps are treated as special tokens (delta_time = 0).
    """
    if not timestamps or len(timestamps) == 0:
        return []
    
    delta_times = []
    for i in range(len(timestamps)):
        # Handle special cases: first event, zero timestamp, or negative timestamp
        if i == 0 or timestamps[i] <= 0:
            # First event, special token, or invalid timestamp
            delta_times.append(0.0)
        elif timestamps[i-1] <= 0:
            # Previous timestamp was invalid, treat current as first valid timestamp
            delta_times.append(0.0)
        else:
            # Normal case: time since previous valid event
            delta = float(timestamps[i]) - float(timestamps[i-1])
            # Ensure non-negative (handle any data irregularities)
            delta_times.append(max(0.0, delta))
    
    return delta_times


def align_delta_times_with_tokens(
    texts: List[str],
    original_delta_times: List[List[float]],
    tokenizer,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Align event-level delta times with tokenized sub-tokens.
    
    Challenge: When "lung cancer" is tokenized to ["lung", "can", "cer"],
    all sub-tokens should get the same delta time as the original event.
    
    Args:
        texts: List of text narratives (one per patient).
        original_delta_times: List of delta time arrays (one per patient),
                             where each array aligns with original event tokens.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length (for truncation/padding).
    
    Returns:
        Tensor of shape (batch_size, seq_len) with aligned delta times.
    """
    batch_delta_times = []
    
    for text, delta_times in zip(texts, original_delta_times):
        if delta_times is None or len(delta_times) == 0:
            # No delta times for this sample (shouldn't happen, but handle gracefully)
            encoded = tokenizer(text, add_special_tokens=True, return_tensors='pt')
            seq_len = encoded['input_ids'].shape[1]
            aligned_deltas = [0.0] * seq_len
            batch_delta_times.append(aligned_deltas)
            continue
        
        # Strategy: Use character-level alignment
        # Track which original token each character belongs to
        aligned_deltas = align_single_sequence(text, delta_times, tokenizer)
        
        # Truncate or pad to max_length if specified
        if max_length:
            if len(aligned_deltas) > max_length:
                aligned_deltas = aligned_deltas[:max_length]
            elif len(aligned_deltas) < max_length:
                aligned_deltas.extend([0.0] * (max_length - len(aligned_deltas)))
        
        batch_delta_times.append(aligned_deltas)
    
    # Convert to tensor
    return torch.tensor(batch_delta_times, dtype=torch.float32)


def align_single_sequence(
    text: str,
    delta_times: List[float],
    tokenizer
) -> List[float]:
    """
    Align delta times for a single sequence.
    
    The approach:
    1. Split text by "; " to get original events
    2. Tokenize each event separately
    3. Assign that event's delta time to all its sub-tokens
    4. Handle special tokens (BOS, EOS, padding)
    
    Args:
        text: Full text narrative for one patient.
        delta_times: Delta times for original events.
        tokenizer: Tokenizer.
    
    Returns:
        List of delta times aligned with tokenized sequence.
    """
    # Split text into events (separated by "; ")
    events = text.split('; ')
    events = [e.strip() for e in events if e.strip()]
    
    # Ensure we have delta times for each event
    # If mismatch, pad or truncate
    if len(delta_times) < len(events):
        delta_times = list(delta_times) + [0.0] * (len(events) - len(delta_times))
    elif len(delta_times) > len(events):
        delta_times = delta_times[:len(events)]
    
    aligned_deltas = []
    
    # Add delta time for BOS token (if tokenizer adds it)
    if tokenizer.bos_token:
        aligned_deltas.append(0.0)
    
    # Tokenize each event and assign its delta time
    for event, delta_time in zip(events, delta_times):
        if not event:
            continue
        
        # Tokenize this event (without special tokens to avoid duplicates)
        tokens = tokenizer.encode(event, add_special_tokens=False)
        
        # All sub-tokens of this event get the same delta time
        aligned_deltas.extend([delta_time] * len(tokens))
        
        # The "; " separator also gets the same delta time
        # (it's part of the event structurally)
    
    # Add delta time for EOS token (if tokenizer adds it)
    if tokenizer.eos_token:
        aligned_deltas.append(0.0)
    
    return aligned_deltas


def normalize_delta_times(
    delta_times: torch.Tensor,
    time_scale: float = 86400.0
) -> torch.Tensor:
    """
    Apply log scaling to delta times.
    
    Formula: log(1 + delta_time / time_scale)
    
    Args:
        delta_times: (batch_size, seq_len) tensor of delta times in seconds.
        time_scale: Scaling factor (86400 for days, 3600 for hours, 1 for seconds).
    
    Returns:
        Log-scaled delta times.
    """
    # Ensure non-negative
    delta_times = torch.clamp(delta_times, min=0.0)
    
    # Apply log scaling
    scaled = torch.log1p(delta_times / time_scale)
    
    return scaled
