"""
Load causal LMs with LoRA adapters.

Two backends:
  - Unsloth (default, ``use_unsloth=True``): fast kernels, 4-bit NF4.
  - HuggingFace + BitsAndBytes (``use_unsloth=False``): any causal LM.
"""
import os
import json
import torch
from unsloth import FastLanguageModel
from typing import Tuple, Any, Optional, Dict

from transformers import (
    AutoModelForCausalLM,
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


def load_hf_model(
    model_name: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    local_rank: int = 0,
) -> Tuple[Any, Any]:
    """Load a causal LM with HuggingFace transformers + optional 4-bit BitsAndBytes."""
    print(f"\nLoading model (HF backend): {model_name}")

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
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
        token=hf_token,
    )
    if not load_in_4bit:
        extra_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **extra_kwargs)
    print(f"  - Loaded as AutoModelForCausalLM (vocab size: {len(tokenizer)})")
    return model, tokenizer


def apply_lora_peft(model: Any, lora_config: Dict[str, Any]) -> Any:
    """Apply LoRA adapters with PEFT (used when Unsloth is off)."""
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("  - prepare_model_for_kbit_training applied")
    except Exception:
        pass

    peft_cfg = LoraConfig(
        r=int(lora_config.get("r", 16)),
        target_modules=lora_config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=int(lora_config.get("lora_alpha", 32)),
        lora_dropout=float(lora_config.get("lora_dropout", 0.05)),
        bias=str(lora_config.get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    print(f"  - Applied LoRA via PEFT (r={peft_cfg.r}, alpha={peft_cfg.lora_alpha})")
    return model


def load_pretrained_lora_model_hf(
    pretrained_checkpoint: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    local_rank: int = 0,
) -> Tuple[Any, Any]:
    """Load a PEFT LoRA checkpoint saved by the HuggingFace backend."""
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


def load_pretrained_lora_model(
    pretrained_checkpoint: str,
    max_seq_length: int = 8192,
    load_in_4bit: bool = True,
    local_rank: int = 0,
) -> Tuple[Any, Any]:
    """Load a pretrained model with LoRA adapters using Unsloth."""
    print(f"\nLoading pretrained model from: {pretrained_checkpoint}")

    if not os.path.exists(pretrained_checkpoint):
        raise ValueError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")

    if torch.cuda.is_available():
        print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  - CUDA not available, using CPU")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=pretrained_checkpoint,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        device_map={"": local_rank},
    )

    print(f"  - Loaded model and tokenizer via Unsloth (vocab size: {len(tokenizer)})")

    has_lora = any("lora" in name.lower() for name, _ in model.named_parameters())
    if has_lora:
        print("  - SUCCESS: Model has LoRA adapters applied with Unsloth kernels.")
    else:
        print("  - WARNING: No LoRA adapters detected in the loaded model!")

    return model, tokenizer
