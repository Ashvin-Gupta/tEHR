"""
Model loading utilities for pretrained models with LoRA adapters.
"""
import torch
from unsloth import FastLanguageModel
from typing import Tuple, Any


def load_pretrained_lora_model(
    pretrained_checkpoint: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    local_rank: int = 0
) -> Tuple[Any, Any]:
    """
    Load a pretrained model with LoRA adapters.
    
    This function loads a model that was previously trained with LoRA
    (e.g., from continued pretraining stage) and prepares it for further
    fine-tuning or inference.
    
    Args:
        pretrained_checkpoint: Path to pretrained model checkpoint directory.
        max_seq_length: Maximum sequence length.
        load_in_4bit: Whether to load model in 4-bit quantization.
        local_rank: Local rank for distributed training.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading pretrained model from: {pretrained_checkpoint}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA devices: {torch.cuda.device_count()}")
    else:
        print("  - CUDA not available, using CPU")
    
    # Load model and tokenizer with LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=pretrained_checkpoint,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        device_map={"": local_rank}
    )
    
    print(f"  - Loaded model and tokenizer (vocab size: {len(tokenizer)})")
    print(f"  - Model has LoRA adapters from pretraining")
    
    return model, tokenizer
