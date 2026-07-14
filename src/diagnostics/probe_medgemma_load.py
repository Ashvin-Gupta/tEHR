#!/usr/bin/env python3
"""
MedGemma load-path probe (text-only EHR).

Compares:
  A) Current production path: AutoModelForCausalLM + AutoTokenizer + 4-bit
  B) HF card path: AutoModelForImageTextToText + AutoProcessor + bf16
     with a *text-only* chat message (no image) — EHR understanding use case.

For each path, prints:
  - model / processor types
  - logits shape, % NaN/Inf, min/max
  - logits for digit tokens "0" / "1"
  - top-8 next tokens
  - a short greedy generation decode

Submit:
  sbatch src/submissions/mimic/probe_medgemma_load.sh
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch


MODEL_ID = "google/medgemma-1.5-4b-it"

TOY_EHR = (
    "Age; 78.0; Male; Race Unknown; Married; <HEADER_SPLIT> "
    "gallbladder disease; ICU admission, medical intensive care unit."
)

BINARY_USER = f"""You are a clinical risk model.
Given this patient's medical history, predict whether this patient will die in the ICU.
Answer with a single digit: 0 (survive) or 1 (die).

Patient's medical history:
{TOY_EHR}

Answer:
"""


def _hf_token():
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or True
    )


def _resolve_digit_ids(tokenizer) -> Tuple[int, int]:
    sys.path.insert(0, os.getcwd())
    from src.training.training_utils import resolve_binary_label_token_ids

    return resolve_binary_label_token_ids(tokenizer)


def _logit_report(logits_1d: torch.Tensor, tokenizer, id0: int, id1: int, tag: str):
    x = logits_1d.detach().float().cpu()
    n = x.numel()
    n_nan = int(torch.isnan(x).sum().item())
    n_inf = int(torch.isinf(x).sum().item())
    finite = x[torch.isfinite(x)]
    print(f"\n[{tag}] logit vector")
    print(f"  shape={tuple(logits_1d.shape)}  dtype={logits_1d.dtype}")
    print(f"  nan={n_nan}/{n} ({100.0 * n_nan / n:.2f}%)  inf={n_inf}/{n}")
    if finite.numel():
        print(f"  finite min={finite.min().item():.4f}  max={finite.max().item():.4f}")
    else:
        print("  NO finite logits")

    l0 = float(x[id0].item()) if id0 < n else float("nan")
    l1 = float(x[id1].item()) if id1 < n else float("nan")
    print(f"  logit_0 (id={id0}, piece={tokenizer.convert_ids_to_tokens(id0)!r}) = {l0}")
    print(f"  logit_1 (id={id1}, piece={tokenizer.convert_ids_to_tokens(id1)!r}) = {l1}")
    if math.isfinite(l0) and math.isfinite(l1):
        p1 = float(torch.softmax(torch.tensor([l0, l1]), dim=0)[1].item())
        print(f"  softmax P(1) over {{0,1}} = {p1:.4f}")
    else:
        print("  softmax P(1) over {0,1} = <non-finite>")

    # top-k among finite entries only if possible
    k = min(8, n)
    vals, idxs = torch.topk(torch.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9), k=k)
    print("  top-8 (nan_to_num):")
    for v, i in zip(vals.tolist(), idxs.tolist()):
        piece = tokenizer.convert_ids_to_tokens(i)
        print(f"    id={i:6d}  logit={v:9.4f}  {piece!r}")


def _last_token_logits_from_causal(model, tokenizer, prompt: str) -> Tuple[torch.Tensor, str]:
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    last_id = int(inputs["input_ids"][0, -1].item())
    last_tok = tokenizer.decode([last_id], skip_special_tokens=False)
    return logits, last_tok


def _generate_causal(model, tokenizer, prompt: str, max_new: int = 16) -> str:
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [eos] if eos is not None else []
    if (
        isinstance(eot, int)
        and eot >= 0
        and tokenizer.convert_ids_to_tokens(eot) == "<end_of_turn>"
        and eot != eos
    ):
        eos_ids.append(eot)
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=pad,
        eos_token_id=eos_ids or eos,
    )
    if pad is not None and pad not in (eos_ids or []):
        gen_kwargs["suppress_tokens"] = [int(pad)]
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new, skip_special_tokens=False)


def probe_current_causallm_4bit(model_id: str, max_new: int):
    print("\n" + "=" * 80)
    print("PATH A — current production: CausalLM + Tokenizer + 4-bit")
    print("=" * 80)
    sys.path.insert(0, os.getcwd())
    from src.training.model_loader import load_hf_model

    model, tok = load_hf_model(
        model_name=model_id,
        max_seq_length=2048,
        load_in_4bit=True,
        local_rank=0,
    )
    model.eval()
    print(f"  model type = {type(model).__name__}")
    id0, id1 = _resolve_digit_ids(tok)

    # Match production: chat wrap + Answer: prefill
    wrapped = tok.apply_chat_template(
        [{"role": "user", "content": BINARY_USER[: BINARY_USER.rfind("Answer:")].rstrip() + "\n"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = wrapped + "Answer:"
    print(f"  prompt tail = {prompt[-120:]!r}")

    logits, last_tok = _last_token_logits_from_causal(model, tok, prompt)
    print(f"  last_tok = {last_tok!r}")
    _logit_report(logits, tok, id0, id1, "A-causallm-4bit")
    try:
        gen = _generate_causal(model, tok, prompt, max_new=max_new)
        print(f"  generate raw = {gen!r}")
    except Exception as exc:
        print(f"  generate FAILED: {type(exc).__name__}: {exc}")

    # Free VRAM before path B
    del model
    torch.cuda.empty_cache()


def probe_official_image_text_bf16(model_id: str, max_new: int, load_in_4bit: bool):
    label = "4-bit" if load_in_4bit else "bf16"
    print("\n" + "=" * 80)
    print(f"PATH B — HF card: ImageTextToText + Processor + {label} (text-only EHR)")
    print("=" * 80)
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    token = _hf_token()
    processor = AutoProcessor.from_pretrained(model_id, token=token)

    kwargs: Dict[str, Any] = dict(
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    print(f"  model type = {type(model).__name__}")
    print(f"  processor type = {type(processor).__name__}")

    # Text-only chat (no image) — EHR understanding
    messages = [{"role": "user", "content": [{"type": "text", "text": BINARY_USER}]}]
    # Prefer processor chat template; fall back to tokenizer if needed.
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as exc:
        print(f"  processor.apply_chat_template failed ({exc}); trying text-only string API")
        # Some processors want role/content strings for text-only
        messages_alt = [{"role": "user", "content": BINARY_USER}]
        inputs = processor.apply_chat_template(
            messages_alt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    # Move tensors; dtype cast like the card for bf16 path
    device = next(model.parameters()).device
    moved = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            moved[k] = v
            continue
        if torch.is_floating_point(v) and not load_in_4bit:
            moved[k] = v.to(device, dtype=torch.bfloat16)
        else:
            moved[k] = v.to(device)
    inputs = moved

    tok = getattr(processor, "tokenizer", None) or processor
    id0, id1 = _resolve_digit_ids(tok)
    input_len = inputs["input_ids"].shape[-1]
    last_id = int(inputs["input_ids"][0, -1].item())
    print(f"  input_len = {input_len}")
    print(f"  last_tok = {tok.decode([last_id], skip_special_tokens=False)!r}")
    # show prompt decode tail
    full = tok.decode(inputs["input_ids"][0], skip_special_tokens=False)
    print(f"  prompt tail = {full[-160:]!r}")

    with torch.inference_mode():
        out = model(**inputs)
    # ImageTextToText logits: typically (B, T, V)
    logits = out.logits[0, -1, :]
    _logit_report(logits, tok, id0, id1, f"B-image_text-{label}")

    try:
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        new = gen[0][input_len:]
        decoded = tok.decode(new, skip_special_tokens=False)
        print(f"  generate raw = {decoded!r}")
        print(f"  generate skip_special = {tok.decode(new, skip_special_tokens=True)!r}")
    except Exception as exc:
        print(f"  generate FAILED: {type(exc).__name__}: {exc}")

    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--skip-a",
        action="store_true",
        help="Skip current CausalLM+4bit path",
    )
    p.add_argument(
        "--skip-b-bf16",
        action="store_true",
        help="Skip official bf16 ImageTextToText path",
    )
    p.add_argument(
        "--also-b-4bit",
        action="store_true",
        help="Also try ImageTextToText + 4-bit",
    )
    args = p.parse_args()

    print("=" * 80)
    print("MedGemma load-path probe (text-only EHR)")
    print(f"  model_id = {args.model_id}")
    print(f"  cuda = {torch.cuda.is_available()}  "
          f"device = {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print("=" * 80)

    if not args.skip_a:
        probe_current_causallm_4bit(args.model_id, args.max_new_tokens)
    if not args.skip_b_bf16:
        probe_official_image_text_bf16(
            args.model_id, args.max_new_tokens, load_in_4bit=False
        )
    if args.also_b_4bit:
        probe_official_image_text_bf16(
            args.model_id, args.max_new_tokens, load_in_4bit=True
        )

    print("\nDONE")


if __name__ == "__main__":
    main()
