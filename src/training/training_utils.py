"""
Utility functions for training.
"""
from transformers import set_seed as hf_set_seed
import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple
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

def resolve_binary_label_token_ids(tokenizer) -> tuple[int, int]:
    """
    Resolve vocabulary IDs for the answer tokens ``0`` and ``1``.

    Mistral / Llama SentencePiece tokenizers encode ``"0"`` as ``[▁, 0]``
    (two tokens). Taking ``encode(...)[0]`` incorrectly picks the shared
    spacepiece ``▁`` for both digits, which forces every softmax probability
    to 0.5. Prefer the last non-special piece, then ``convert_tokens_to_ids``.
    """
    def _one(label: str) -> int:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer produced no IDs for label {label!r}")
        # Prefer a single-token encoding; otherwise take the final piece
        # (the digit itself after a leading ▁).
        if len(ids) == 1:
            return int(ids[0])
        tid = int(ids[-1])
        # Sanity: decoded piece should contain the digit character.
        piece = tokenizer.convert_ids_to_tokens(tid)
        if piece is not None and label not in str(piece).replace("▁", ""):
            alt = tokenizer.convert_tokens_to_ids(label)
            if alt is not None and alt != tokenizer.unk_token_id:
                return int(alt)
        return tid

    id0, id1 = _one("0"), _one("1")
    if id0 == id1:
        raise RuntimeError(
            f"Label tokens '0' and '1' resolved to the same id ({id0}). "
            "Logit-based binary scoring would be degenerate (always p=0.5)."
        )
    return id0, id1


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


def resolve_seed(config: Dict[str, Any], cli_seed: Optional[int] = None) -> int:
    """
    Resolve the run seed.

    Priority: CLI ``--seed`` > top-level ``seed`` > ``training.seed`` > ``42``.
    """
    if cli_seed is not None:
        return int(cli_seed)
    if config.get("seed") is not None:
        return int(config["seed"])
    training = config.get("training") or {}
    if training.get("seed") is not None:
        return int(training["seed"])
    return 42


def find_last_checkpoint(output_dir: Optional[str]) -> Optional[str]:
    """
    Return the newest ``checkpoint-*`` under ``output_dir``, or None.
    Also checks ``output_dir/checkpoints`` (used by some baselines).
    """
    if not output_dir:
        return None

    try:
        from transformers.trainer_utils import get_last_checkpoint
    except Exception:
        get_last_checkpoint = None

    candidates = [output_dir, os.path.join(output_dir, "checkpoints")]
    for path in candidates:
        if not path or not os.path.isdir(path):
            continue
        last = None
        if get_last_checkpoint is not None:
            try:
                last = get_last_checkpoint(path)
            except Exception:
                last = None
        if last:
            return last
    return None


def prepare_run_reproducibility(
    config: Dict[str, Any],
    *,
    cli_seed: Optional[int] = None,
    cli_checkpoint: Optional[str] = None,
    cli_resume_from_last: Optional[bool] = None,
    use_seed_subdir: bool = False,
) -> Tuple[int, Optional[str]]:
    """
    Apply seed globally and optionally resolve a resume checkpoint.

    Config knobs
    ------------
    - ``seed`` / ``training.seed``: fixed seed (default 42)
    - ``training.resume_from_last``: if true, resume from newest checkpoint
      under ``training.output_dir`` (or ``.../checkpoints``)
    - CLI ``--seed`` / ``--checkpoint_path`` / ``--resume_from_last`` /
      ``--no-resume_from_last`` override config

    When ``use_seed_subdir`` is true, outputs go to ``<output_dir>/seed_<seed>/``
    (classifier / SFT convention).

    Returns:
        ``(seed, resume_checkpoint_or_None)``
    """
    training = config.setdefault("training", {})
    seed = resolve_seed(config, cli_seed=cli_seed)
    config["seed"] = seed
    training["seed"] = seed
    seed_all(seed)
    logger.info(f"Using seed: {seed}")

    if cli_resume_from_last is not None:
        training["resume_from_last"] = bool(cli_resume_from_last)
        logger.info(f"CLI override: resume_from_last={training['resume_from_last']}")

    if use_seed_subdir:
        base_output_dir = training.get("output_dir")
        if base_output_dir:
            seed_name = f"seed_{seed}"
            # Avoid nesting seed_42/seed_42 if already applied.
            if os.path.basename(os.path.normpath(base_output_dir)) != seed_name:
                training["data_cache_dir"] = training.get("data_cache_dir", base_output_dir)
                training["output_dir"] = os.path.join(base_output_dir, seed_name)
            logger.info(f"Outputs will be written to: {training['output_dir']}")

    resume_path = cli_checkpoint
    if resume_path is None and bool(training.get("resume_from_last", False)):
        resume_path = find_last_checkpoint(training.get("output_dir"))
        if resume_path:
            logger.info(f"resume_from_last=true → resuming from {resume_path}")
            # Never wipe an output dir we are about to resume into.
            training["overwrite_output_dir"] = False
        else:
            logger.info(
                "resume_from_last=true but no checkpoint found under "
                f"{training.get('output_dir')}; starting fresh"
            )

    if resume_path and not os.path.exists(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    return seed, resume_path
