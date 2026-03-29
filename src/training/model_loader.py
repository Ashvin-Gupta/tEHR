"""
Model loading utilities for pretrained models with LoRA adapters.
"""
import os
import json
import torch
from unsloth import FastLanguageModel
from typing import Tuple, Any, Optional, Dict, List

from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from src.models.qwen3_embedding_classifier import Qwen3EmbeddingClassifier


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


def is_qwen3_embedding_baseline_checkpoint(checkpoint_dir: str) -> bool:
    """
    Detect ``final_model`` dirs written by ``baseline_qwen3_embedding.save_model``:
    minimal JSON (model_name / hidden_size / num_labels), ``pytorch_model.bin`` as
    ``torch.save(state_dict)``, not an Unsloth causal LM tree.
    """
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not (os.path.exists(cfg_path) and os.path.exists(weights_path)):
        return False
    try:
        with open(cfg_path, "r") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(d, dict):
        return False
    if any(k not in d for k in ("model_name", "hidden_size", "num_labels")):
        return False
    if d.get("model_type") is not None or d.get("architectures"):
        return False
    return True


def load_qwen3_embedding_classifier_from_checkpoint(
    checkpoint_dir: str,
    model_config: Dict[str, Any],
    local_rank: int = 0,
) -> Tuple[Any, Any]:
    """
    Rebuild ``Qwen3EmbeddingClassifier`` and load weights from a baseline save dir.

    ``use_lora`` and ``lora`` are taken from ``checkpoint_dir/config.json`` when
    present (newer saves); otherwise from ``model_config`` (training YAML).
    """
    print(f"\nLoading Qwen3-Embedding classifier from: {checkpoint_dir}")

    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        ckpt_meta = json.load(f)

    merged: Dict[str, Any] = dict(model_config)
    merged.update(ckpt_meta)

    model_name = merged["model_name"]
    hidden_size = int(merged.get("hidden_size", 4096))
    num_labels = int(merged.get("num_labels", 2))
    head_dropout = float(merged.get("head_dropout", 0.1))
    use_lora = bool(merged.get("use_lora", False))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if hasattr(base_model, "resize_token_embeddings"):
        base_model.resize_token_embeddings(len(tokenizer))

    trainable_param_keywords: List[str]
    if use_lora:
        lora_settings = merged.get("lora") or {}
        target_modules = lora_settings.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        peft_config = LoraConfig(
            r=int(lora_settings.get("r", 16)),
            target_modules=target_modules,
            lora_alpha=int(lora_settings.get("lora_alpha", 16)),
            lora_dropout=float(lora_settings.get("lora_dropout", 0.05)),
            bias=str(lora_settings.get("bias", "none")),
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
        )
        base_model = get_peft_model(base_model, peft_config)
        trainable_param_keywords = ["lora_"]
        print(f"  - LoRA enabled (r={peft_config.r}, target_modules={target_modules})")
    else:
        trainable_param_keywords = []
        print("  - LoRA disabled (head-only checkpoint)")

    for p in base_model.parameters():
        p.requires_grad = False
    if use_lora:
        for name, p in base_model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    full_model = Qwen3EmbeddingClassifier(
        base_model=base_model,
        hidden_size=hidden_size,
        num_labels=num_labels,
        head_dropout=head_dropout,
        freeze_base=True,
        trainable_param_keywords=trainable_param_keywords,
        tokenizer=tokenizer,
    )

    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")

    missing, unexpected = full_model.load_state_dict(state, strict=False)
    if missing:
        print(f"  - Warning: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  - Warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    full_model.eval()
    print("  - Qwen3EmbeddingClassifier weights loaded.")
    return full_model, tokenizer


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
    
    Does **not** load Qwen3-Embedding baseline saves; use
    ``load_qwen3_embedding_classifier_from_checkpoint`` for those.
    
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
