"""
Model loading utilities for pretrained models with LoRA adapters.
"""
import os
import json
import torch
from unsloth import FastLanguageModel
from typing import Tuple, Any, Optional


# def _load_base_plus_adapter_from_checkpoint(
#     checkpoint_dir: str,
#     max_seq_length: int,
#     load_in_4bit: bool,
#     local_rank: int,
# ) -> Tuple[Any, Any]:
#     """
#     Load a base model and attach a LoRA adapter from an Unsloth/PEFT checkpoint.
    
#     This is used when the checkpoint directory only contains LoRA weights
#     (e.g. `adapter_model.safetensors` + `adapter_config.json`) but no
#     `config.json` for a full HF model.
#     """
#     adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
#     if not os.path.exists(adapter_config_path):
#         raise RuntimeError(
#             f"Checkpoint at '{checkpoint_dir}' does not contain 'config.json' or "
#             f"'adapter_config.json'. Cannot infer base model to load."
#         )

#     with open(adapter_config_path, "r") as f:
#         adapter_cfg = json.load(f)

#     base_model_name = adapter_cfg.get("base_model_name_or_path") or adapter_cfg.get("base_model_name")
#     if not base_model_name:
#         raise RuntimeError(
#             f"'adapter_config.json' at '{adapter_config_path}' does not specify "
#             f"'base_model_name_or_path'. Available keys: {list(adapter_cfg.keys())}"
#         )

#     print(f"  - Detected base model from adapter config: {base_model_name}")

#     # 1) Load the base model with Unsloth
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=base_model_name,
#         max_seq_length=max_seq_length,
#         dtype=None,
#         load_in_4bit=load_in_4bit,
#         device_map={"": local_rank},
#     )

#     # 2) Attach the LoRA adapter weights
#     if hasattr(model, "load_adapter"):
#         # Preferred Unsloth/PEFT API – this reads adapter_config.json and adapter_model.safetensors
#         print(f"  - Loading LoRA adapter from checkpoint: {checkpoint_dir}")
#         model.load_adapter(checkpoint_dir)
#     else:
#         # Fallback to PEFT directly if needed
#         try:
#             from peft import PeftModel  # type: ignore
#         except ImportError as e:
#             raise RuntimeError(
#                 "PEFT is required to load LoRA adapters from checkpoint, but it is "
#                 "not installed or importable."
#             ) from e

#         print(f"  - Wrapping base model with PEFT adapter from: {checkpoint_dir}")
#         model = PeftModel.from_pretrained(model, checkpoint_dir)

#     return model, tokenizer


# def load_pretrained_lora_model(
#     pretrained_checkpoint: str,
#     max_seq_length: int = 8192,
#     load_in_4bit: bool = True,
#     local_rank: int = 0
# ) -> Tuple[Any, Any]:
#     """
#     Load a pretrained model with LoRA adapters.
    
#     This function handles two cases:
#     1) `pretrained_checkpoint` is a full HF/Unsloth model directory containing
#        `config.json` + weights.
#     2) `pretrained_checkpoint` is a LoRA-only checkpoint directory containing
#        `adapter_model.safetensors` + `adapter_config.json`, but no `config.json`.
    
#     Args:
#         pretrained_checkpoint: Path to pretrained model or LoRA checkpoint directory.
#         max_seq_length: Maximum sequence length.
#         load_in_4bit: Whether to load model in 4-bit quantization.
#         local_rank: Local rank for distributed training.
    
#     Returns:
#         Tuple of (model, tokenizer)
#     """
#     print(f"\nLoading pretrained model from: {pretrained_checkpoint}")

#     if not os.path.exists(pretrained_checkpoint):
#         raise ValueError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")
    
#     # Check CUDA availability
#     if torch.cuda.is_available():
#         print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
#         print(f"  - CUDA devices: {torch.cuda.device_count()}")
#     else:
#         print("  - CUDA not available, using CPU")

#     config_path = os.path.join(pretrained_checkpoint, "config.json")
#     adapter_config_path = os.path.join(pretrained_checkpoint, "adapter_config.json")

#     if os.path.exists(config_path):
#         # Standard HF/Unsloth model directory
#         print("  - Detected full model directory with 'config.json'")
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name=pretrained_checkpoint,
#             max_seq_length=max_seq_length,
#             dtype=None,
#             load_in_4bit=load_in_4bit,
#             device_map={"": local_rank},
#         )
#     else:
#         # Fall back to adapter-only checkpoint
#         print("  - No 'config.json' found; attempting to load as LoRA adapter checkpoint")
#         model, tokenizer = _load_base_plus_adapter_from_checkpoint(
#             checkpoint_dir=pretrained_checkpoint,
#             max_seq_length=max_seq_length,
#             load_in_4bit=load_in_4bit,
#             local_rank=local_rank,
#         )
    
#     print(f"  - Loaded model and tokenizer (vocab size: {len(tokenizer)})")
#     print(f"  - Model has LoRA adapters from pretraining")
    
#     return model, tokenizer

def load_pretrained_lora_model(
    pretrained_checkpoint: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    local_rank: int = 0
) -> Tuple[Any, Any]:
    """
    Load a pretrained model with LoRA adapters using Unsloth's native loader.
    """
    print(f"\nLoading pretrained model from: {pretrained_checkpoint}")

    if not os.path.exists(pretrained_checkpoint):
        raise ValueError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  - CUDA not available, using CPU")

    # Unsloth's from_pretrained is smart enough to detect if a directory 
    # is a base model or a PEFT adapter directory. If it's a PEFT directory,
    # it reads adapter_config.json, loads the correct base model, applies
    # the optimized kernels, and loads the LoRA weights perfectly.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=pretrained_checkpoint,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        device_map={"": local_rank},
    )
    
    print(f"  - Loaded model and tokenizer via Unsloth (vocab size: {len(tokenizer)})")
    
    # Verify LoRA weights actually loaded
    has_lora = any('lora' in name.lower() for name, _ in model.named_parameters())
    if has_lora:
        print(f"  - SUCCESS: Model has LoRA adapters applied with Unsloth kernels.")
    else:
        print(f"  - WARNING: No LoRA adapters detected in the loaded model!")
    
    return model, tokenizer


def load_classification_model(
    model_checkpoint: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    local_rank: int = 0,
    model_type: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Load a saved classification model (LLMClassifier).
    
    This function can load models saved from:
    - Baseline models (frozen_llm_linear, pretrained_llm_linear, lora_llm_linear)
    - Main classification trainer
    
    Args:
        model_checkpoint: Path to saved model directory (should contain model files).
        max_seq_length: Maximum sequence length.
        load_in_4bit: Whether to load model in 4-bit quantization.
        local_rank: Local rank for distributed training.
        model_type: Optional hint about model type. If None, will try to auto-detect.
                    Options: 'base', 'pretrained_lora', 'lora', 'clinicalbert'
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading classification model from: {model_checkpoint}")
    
    # Check if checkpoint exists
    if not os.path.exists(model_checkpoint):
        raise ValueError(f"Model checkpoint not found: {model_checkpoint}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  - CUDA not available, using CPU")
    
    # Try to load using FastLanguageModel (works for Unsloth models with/without LoRA)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_checkpoint,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map={"": local_rank}
        )
        
        print(f"  - Loaded model and tokenizer (vocab size: {len(tokenizer)})")
        
        # Check if model has LoRA adapters
        has_lora = any('lora' in name.lower() for name, _ in model.named_parameters())
        if has_lora:
            print(f"  - Model has LoRA adapters")
        else:
            print(f"  - Model does not have LoRA adapters (base model only)")
        
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {model_checkpoint}. "
            f"Error: {e}\n"
            f"Make sure the checkpoint contains valid model files."
        )
