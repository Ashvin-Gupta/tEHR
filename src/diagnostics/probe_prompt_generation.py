#!/usr/bin/env python3
"""
Cluster-side prompt/generation probe for instruct LLMs (MedGemma, Mistral, …).

Runs a handful of short synthetic EHR prompts under several chat-formatting
strategies and prints:
  - raw newly-generated token ids
  - decode with / without skip_special_tokens
  - next-token top-5 under logit_01 (no generation)

Submit via:
  sbatch src/submissions/mimic/probe_prompt.sh medgemma
  sbatch src/submissions/mimic/probe_prompt.sh mistral
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


TOY_EHR = (
    "Age; 78.0; Male; Race Unknown; Married; <HEADER_SPLIT> "
    "gallbladder disease; ICU admission, medical intensive care unit;<end>"
)

USER_TEMPLATE = """You are a clinical risk model.
Given this patient's medical history, estimate the probability that this patient will die in the ICU, given that roughly 7% of patients die in the ICU.

Output EXACTLY one number on a single line:
  - A probability between 0 and 1 inclusive
  - With exactly 2 digits after the decimal point (examples: 0.03, 0.25, 0.80)
Do NOT output any words, explanations, percentages, or extra symbols.
Only output the probability.

Patient's medical history:
{ehr}

Answer:
"""

BINARY_USER = """You are a clinical risk model.
Given this patient's medical history, predict whether this patient will die in the ICU.
Answer with a single digit: 0 (survive) or 1 (die).

Patient's medical history:
{ehr}

Answer:
"""


def _load(model_name: str, use_unsloth: bool, load_in_4bit: bool):
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        FastLanguageModel.for_inference(model)
        return model, tok

    # Same path as production MedGemma loads.
    sys.path.insert(0, os.getcwd())
    from src.training.model_loader import load_hf_model

    return load_hf_model(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=load_in_4bit,
        local_rank=0,
    )


def _chat_wrap(tok, user_text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _strategies(tok, ehr: str) -> dict[str, str]:
    """Named prompt variants to compare."""
    user_full = USER_TEMPLATE.format(ehr=ehr)
    # Mirror production: strip trailing "Answer:" from user turn, wrap, re-append.
    answer_suffix = "Answer:"
    stripped = user_full.rstrip()
    if stripped.endswith(answer_suffix):
        user_body = stripped[: -len(answer_suffix)].rstrip() + "\n"
    else:
        user_body = user_full

    wrapped = _chat_wrap(tok, user_body)
    wrapped_answer_in_user = _chat_wrap(tok, user_body + answer_suffix + "\n")
    binary_body = BINARY_USER.format(ehr=ehr)
    if binary_body.rstrip().endswith(answer_suffix):
        binary_body = binary_body.rstrip()[: -len(answer_suffix)].rstrip() + "\n"
    wrapped_binary = _chat_wrap(tok, binary_body)

    return {
        "A_prod_prefill_Answer": wrapped + answer_suffix,  # current PureLLMBaseline
        "B_gen_prompt_only": wrapped,  # no assistant prefill
        "C_Answer_inside_user": wrapped_answer_in_user,
        "D_raw_no_chat": user_full,
        "E_binary_gen_prompt": wrapped_binary + answer_suffix,
        "F_binary_gen_prompt_only": wrapped_binary,
    }


def _generate(model, tok, prompt: str, max_new_tokens: int):
    prev = tok.padding_side
    tok.padding_side = "left"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    tok.padding_side = prev

    eos = tok.eos_token_id
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [eos] if eos is not None else []
    if (
        isinstance(eot, int)
        and eot >= 0
        and tok.convert_ids_to_tokens(eot) == "<end_of_turn>"
        and eot != eos
    ):
        eos_ids.append(eot)

    pad = tok.pad_token_id
    suppress = []
    if pad is not None and pad not in eos_ids:
        suppress = [int(pad)]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad,
        eos_token_id=eos_ids if eos_ids else eos,
    )
    if suppress:
        gen_kwargs["suppress_tokens"] = suppress

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = out[0, prompt_len:].tolist()
    return new_ids, inputs["input_ids"].shape[1]


def _logit_topk(model, tok, prompt: str, k: int = 8):
    prev = tok.padding_side
    tok.padding_side = "left"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    tok.padding_side = prev
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    vals, idxs = torch.topk(logits, k=k)
    rows = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        piece = tok.decode([i], skip_special_tokens=False)
        rows.append((i, v, repr(piece)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--use-unsloth", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=16)
    args = p.parse_args()

    print("=" * 80)
    print("prompt generation probe")
    print(f"  model_name  = {args.model_name}")
    print(f"  use_unsloth = {args.use_unsloth}")
    print(f"  load_in_4bit= {not args.no_4bit}")
    print("=" * 80)

    model, tok = _load(args.model_name, args.use_unsloth, not args.no_4bit)
    model.eval()

    print(f"\neos_token={tok.eos_token!r} id={tok.eos_token_id}")
    print(f"pad_token={tok.pad_token!r} id={tok.pad_token_id}")
    print(f"padding_side={tok.padding_side}")
    for special in ("<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>"):
        try:
            sid = tok.convert_tokens_to_ids(special)
        except Exception:
            sid = None
        print(f"  {special!r} → {sid}")

    strategies = _strategies(tok, TOY_EHR)
    for name, prompt in strategies.items():
        print("\n" + "=" * 80)
        print(f"STRATEGY: {name}")
        print("-" * 80)
        print("prompt tail:")
        print(repr(prompt[-200:]))
        new_ids, n_prompt = _generate(model, tok, prompt, args.max_new_tokens)
        print(f"prompt_tokens={n_prompt}  new_ids={new_ids}")
        print(f"decode skip_special = {tok.decode(new_ids, skip_special_tokens=True)!r}")
        print(f"decode keep_special = {tok.decode(new_ids, skip_special_tokens=False)!r}")
        top = _logit_topk(model, tok, prompt, k=8)
        print("next-token top-8 (before generate):")
        for tid, logit, piece in top:
            print(f"  id={tid:6d}  logit={logit:8.3f}  {piece}")

    print("\nDONE")


if __name__ == "__main__":
    main()
