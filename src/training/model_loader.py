"""
Model loading utilities for pretrained models with LoRA adapters.

Two loading backends are supported:
  - Unsloth (default, use_unsloth=True): fast kernels, 4-bit NF4, best for
    supported models (Mistral, Llama, Qwen, Gemma 2, …).
  - HuggingFace + BitsAndBytes (use_unsloth=False): compatible with any model,
    including multimodal ones such as google/medgemma-1.5-4b-it whose
    Gemma3ForConditionalGeneration architecture is not registered with Unsloth.
    The language-model backbone is extracted automatically when a multimodal
    checkpoint is loaded.
"""
import os
import json
import torch
from unsloth import FastLanguageModel
from typing import Tuple, Any, Optional, Dict, List

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
# For explicit Levenshtein compatibility with different PEFT import names if needed
from peft import LoraConfig as PeftLoraConfig, get_peft_model as peft_get_peft_model, TaskType as PeftTaskType

from src.models.qwen3_embedding_classifier import Qwen3EmbeddingClassifier


# ---------------------------------------------------------------------------
# HuggingFace + BitsAndBytes backend (use_unsloth=False)
# ---------------------------------------------------------------------------

def load_hf_model(
    model_name: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    local_rank: int = 0,
) -> Tuple[Any, Any]:
    """
    Load a model using standard HuggingFace transformers + BitsAndBytes 4-bit
    quantisation.  Supports any architecture including multimodal models such
    as google/medgemma-1.5-4b-it: if AutoModelForCausalLM fails the full
    multimodal model is loaded and its ``language_model`` backbone is returned.
    """

    print(f"\nLoading model (HF backend): {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config: Optional[Any] = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    device = {"": local_rank} if torch.cuda.is_available() else "auto"
    extra_kwargs: Dict[str, Any] = dict(
        quantization_config=quantization_config,
        device_map=device,
        trust_remote_code=True,
    )
    if not load_in_4bit:
        extra_kwargs["torch_dtype"] = torch.bfloat16

    # --- Try CausalLM first ---
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **extra_kwargs)
        print(f"  - Loaded as AutoModelForCausalLM (vocab size: {len(tokenizer)})")
        return model, tokenizer
    except Exception as causal_err:
        print(f"  - AutoModelForCausalLM failed ({type(causal_err).__name__}), "
              "trying multimodal path...")

    # --- Fallback: multimodal model, extract language_model component ---
    try:
        
        full_model = AutoModelForImageTextToText.from_pretrained(model_name, **extra_kwargs)
    except Exception:
        full_model = AutoModel.from_pretrained(model_name, **extra_kwargs)

    if hasattr(full_model, "language_model"):
        model = full_model.language_model
        print(f"  - Extracted language_model component from multimodal model "
              f"(type: {type(model).__name__})")
        # Resize token embeddings to match tokenizer if necessary
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    raise RuntimeError(
        f"Cannot load {model_name} as a language model: "
        f"AutoModelForCausalLM failed and the multimodal fallback model "
        f"({type(full_model).__name__}) has no 'language_model' attribute."
    )


def apply_lora_peft(model: Any, lora_config: Dict[str, Any]) -> Any:
    """
    Apply LoRA adapters to *model* using standard PEFT (no Unsloth).
    Handles BitsAndBytes-quantised models via ``prepare_model_for_kbit_training``.
    """
    

    # Prepare quantised model (enable gradient checkpointing, cast norms, …)
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("  - prepare_model_for_kbit_training applied")
    except Exception:
        pass  # Not quantised or already prepared

    peft_cfg = PeftLoraConfig(
        r=int(lora_config.get("r", 16)),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=int(lora_config.get("lora_alpha", 32)),
        lora_dropout=float(lora_config.get("lora_dropout", 0.05)),
        bias=str(lora_config.get("bias", "none")),
        task_type=PeftTaskType.CAUSAL_LM,
    )

    model = peft_get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    print(f"  - Applied LoRA via PEFT (r={peft_cfg.r}, alpha={peft_cfg.lora_alpha})")
    return model


def load_pretrained_lora_model_hf(
    pretrained_checkpoint: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    local_rank: int = 0,
) -> Tuple[Any, Any]:
    """
    Load a pretrained model+LoRA checkpoint saved by the HF/PEFT backend.
    Reads ``adapter_config.json`` to find the base model, loads it via
    ``load_hf_model``, then wraps it with ``PeftModel.from_pretrained``.
    """

    if not os.path.exists(pretrained_checkpoint):
        raise ValueError(f"Checkpoint not found: {pretrained_checkpoint}")

    adapter_cfg_path = os.path.join(pretrained_checkpoint, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        raise ValueError(f"No adapter_config.json in {pretrained_checkpoint}")

    with open(adapter_cfg_path) as fh:
        adapter_cfg = json.load(fh)

    base_model_name = adapter_cfg.get("base_model_name_or_path")
    print(f"  - Base model (from adapter_config): {base_model_name}")

    base_model, tokenizer = load_hf_model(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        local_rank=local_rank,
    )

    model = PeftModel.from_pretrained(base_model, pretrained_checkpoint)
    print(f"  - Loaded LoRA adapters from {pretrained_checkpoint}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Unsloth backend (original, use_unsloth=True)
# ---------------------------------------------------------------------------

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


def is_qwen3_embedding_trainer_checkpoint(checkpoint_dir: str) -> bool:
    """
    Detect HF Trainer checkpoint dirs for qwen3 baseline (checkpoint-*/ with tokenizer/trainer state).
    """
    base = os.path.basename(os.path.normpath(checkpoint_dir))
    return (
        base.startswith("checkpoint-")
        and os.path.exists(os.path.join(checkpoint_dir, "trainer_state.json"))
        and os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json"))
    )


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

    cfg_path = os.path.join(checkpoint_dir, "config.json")
    ckpt_meta: Dict[str, Any] = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
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

    pt_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    sf_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(pt_path):
        try:
            state = torch.load(pt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(pt_path, map_location="cpu")
    elif os.path.exists(sf_path):
        from safetensors.torch import load_file
        state = load_file(sf_path)
    else:
        raise RuntimeError(
            f"No classifier weights found in checkpoint dir: {checkpoint_dir}. "
            "Expected `pytorch_model.bin` or `model.safetensors`. "
            "If this is an older qwen3 baseline Trainer checkpoint, re-run with the "
            "updated checkpoint exporter or use a checkpoint directory that contains full weights."
        )

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
