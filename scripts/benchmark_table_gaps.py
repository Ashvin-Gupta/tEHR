#!/usr/bin/env python3
"""Fill the MIMIC compute-table cells the 8 Oct benchmark left empty.

Same method as ``benchmark_icu_compute.py``: one warmup step plus 10
full-context steps on a synthetic batch, then linear extrapolation to
GPU-hours. MIMIC is not loaded. Full-length steps are an upper bound.

The tEHR-Mistral row (85.11 GPU-h, 26.09 GPU-h, 301 ms) is the context-8192
measurement. Qwen rows use that same context, the same stage-1 token budget,
and the configured microbatches (4 and 8) so the three tEHR rows compare
model speed rather than a different sequence length.

Atomic-code Mistral is the event-token run: context 2,048, stage-1 cohort
65,856 (the completed log), stage-2 cohort 75,050 ICU-mortality examples.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_icu_compute import (  # noqa: E402
    N_TRAIN,
    OUT_DIR,
    TEHR_S1_SEQUENCES_AT_8192,
    _enter_training,
    _is_oom,
    _load_unsloth,
    cfg,
    count_params,
    gpu_hours,
    release,
    timed,
)

STATE: dict = {}

CONTEXT = 8192
EVENT_CONTEXT = 2048
# Completed event-token stage-1 log: 65,856 examples, 1 epoch, microbatch 4.
EVENT_S1_EXAMPLES = 65_856
EVENT_VOCAB = 72_119
SUMMARY_PATH = OUT_DIR / "table_gaps.json"


def _emit(bucket: dict, key: str, row: dict) -> None:
    bucket[key] = row
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(STATE, indent=2, default=str))
    print(f"  saved {key}", flush=True)
    interesting = {
        k: row.get(k)
        for k in (
            "context",
            "stage1_gpu_h",
            "stage2_gpu_h",
            "latency_ms",
            "train_microbatch",
            "peak_vram_gib",
            "error",
        )
        if k in row
    }
    print(f"  {key} {json.dumps(interesting)}", flush=True)


def _train_lm(model, vocab: int, length: int, batch: int):
    import torch

    ids = torch.randint(0, vocab, (batch, length), device="cuda")
    mask = torch.ones(batch, length, dtype=torch.long, device="cuda")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
    model.train()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
        loss.backward()
        opt.step()

    stats = timed(step)
    del opt
    return stats


def _train_cls(clf, vocab: int, length: int, batch: int):
    import torch

    ids = torch.randint(0, vocab, (batch, length), device="cuda")
    mask = torch.ones(batch, length, dtype=torch.long, device="cuda")
    y = torch.randint(0, 2, (batch,), device="cuda")
    opt = torch.optim.AdamW((p for p in clf.parameters() if p.requires_grad), lr=1e-6)
    clf.train()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = clf(input_ids=ids, attention_mask=mask, labels=y)["loss"]
        loss.backward()
        opt.step()

    stats = timed(step)
    del opt
    return stats, ids, mask


def _patch_torchao_symbol() -> None:
    """PEFT imports a torchao class this environment no longer exports.

    Qwen3 leaves a few MLP linears in bf16. Wrapping those reaches PEFT's
    torchao dispatcher, which crashes on the missing symbol before it can
    fall through to a normal linear. The symbol is only needed for an
    isinstance check against torchao weights, which these layers are not.
    """
    import torchao.quantization as quantization

    if hasattr(quantization, "LinearActivationQuantizedTensor"):
        return

    class LinearActivationQuantizedTensor:
        pass

    quantization.LinearActivationQuantizedTensor = LinearActivationQuantizedTensor


def bench_tehr(model_key: str, bucket: dict) -> None:
    import torch

    from src.models.llm_classifier import LLMClassifier

    _patch_torchao_symbol()

    s1_cfg = cfg("tehr_stage1", model_key)
    s2_cfg = cfg("tehr_stage2", model_key)
    s1_batch = int(s1_cfg["training"]["batch_size"])
    s2_batch = int(s2_cfg["training"]["batch_size"])
    name = s1_cfg["model"]["model_name"]
    tag = model_key.replace(".", "")
    print(f"\n===== tEHR {model_key} @ {CONTEXT} ({name}) =====", flush=True)

    FastLanguageModel, model, tokenizer = _load_unsloth(name, CONTEXT)
    vocab = int(getattr(tokenizer, "vocab_size", 32768))
    _enter_training(FastLanguageModel, model)
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(s1_cfg["lora"].get("r", 16)),
        target_modules=s1_cfg["lora"].get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=int(s1_cfg["lora"].get("lora_alpha", 32)),
        lora_dropout=float(s1_cfg["lora"].get("lora_dropout", 0.05)),
        bias=s1_cfg["lora"].get("bias", "none"),
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
    )
    model.train()
    used_s1 = s1_batch
    try:
        s1_stats = _train_lm(model, vocab, CONTEXT, s1_batch)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        used_s1 = max(1, s1_batch // 2)
        print(f"  stage-1 OOM at batch {s1_batch}; retrying {used_s1}", flush=True)
        s1_stats = _train_lm(model, vocab, CONTEXT, used_s1)
    total, trainable = count_params(model)
    _emit(bucket, f"tehr_{tag}_stage1_ctx{CONTEXT}", {
        "model": model_key,
        "model_name": name,
        "context": CONTEXT,
        "stage1_gpu_h": gpu_hours(
            TEHR_S1_SEQUENCES_AT_8192, 1, used_s1, s1_stats["sec_per_step_median"]
        ),
        "train_microbatch": used_s1,
        "sequences": TEHR_S1_SEQUENCES_AT_8192,
        "total_params": total,
        "trainable_params": trainable,
        "peak_vram_gib": s1_stats["peak_allocated_gib"],
        "sec_per_step_median": s1_stats["sec_per_step_median"],
        "notes": (
            f"Same stage-1 token budget as the tEHR-Mistral row "
            f"({TEHR_S1_SEQUENCES_AT_8192} sequences at context {CONTEXT})."
        ),
    })

    hidden = int(s2_cfg["model"]["hidden_size"])
    clf = LLMClassifier(
        base_model=model,
        hidden_size=hidden,
        num_labels=2,
        freeze_base=True,
        trainable_param_keywords=["lora_"],
        pooling=s2_cfg["model"].get("pooling", "last_token"),
    ).to("cuda")
    used_s2 = s2_batch
    try:
        s2_stats, ids, mask = _train_cls(clf, vocab, CONTEXT, s2_batch)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        used_s2 = max(1, s2_batch // 2)
        print(f"  stage-2 OOM at batch {s2_batch}; retrying {used_s2}", flush=True)
        s2_stats, ids, mask = _train_cls(clf, vocab, CONTEXT, used_s2)
    _emit(bucket, f"tehr_{tag}_stage2_ctx{CONTEXT}", {
        "model": model_key,
        "context": CONTEXT,
        "stage2_gpu_h": gpu_hours(N_TRAIN, 1, used_s2, s2_stats["sec_per_step_median"]),
        "train_microbatch": used_s2,
        "n_examples": N_TRAIN,
        "epochs": 1,
        "peak_vram_gib": s2_stats["peak_allocated_gib"],
        "sec_per_step_median": s2_stats["sec_per_step_median"],
        "notes": f"1 epoch on {N_TRAIN} ICU-mortality examples, context {CONTEXT}.",
    })

    clf.eval()
    one_ids = ids[:1]
    one_mask = mask[:1]

    def infer():
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                clf(input_ids=one_ids, attention_mask=one_mask)

    inf = timed(infer)
    _emit(bucket, f"tehr_{tag}_latency_bs1_ctx{CONTEXT}", {
        "model": model_key,
        "context": CONTEXT,
        "infer_batch_size": 1,
        "latency_ms": inf["sec_per_step_median"] * 1000.0,
        "peak_vram_gib": inf["peak_allocated_gib"],
    })
    del clf, model, FastLanguageModel, ids, mask, one_ids, one_mask
    release()


def bench_event_token(bucket: dict) -> None:
    import torch

    from src.models.llm_classifier import LLMClassifier

    s1_cfg = cfg("event_token_stage1", "mistral")
    s2_cfg = cfg("event_token_stage2", "mistral")
    name = s1_cfg["model"]["model_name"]
    print(f"\n===== atomic codes (event-token Mistral) @ {EVENT_CONTEXT} =====", flush=True)
    FastLanguageModel, model, tokenizer = _load_unsloth(name, EVENT_CONTEXT)
    _enter_training(FastLanguageModel, model)
    model.resize_token_embeddings(EVENT_VOCAB)
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(s1_cfg["lora"].get("r", 16)),
        target_modules=s1_cfg["lora"].get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=int(s1_cfg["lora"].get("lora_alpha", 32)),
        lora_dropout=float(s1_cfg["lora"].get("lora_dropout", 0.05)),
        bias=s1_cfg["lora"].get("bias", "none"),
        modules_to_save=["embed_tokens", "lm_head"],
        ensure_weight_tying=True,
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
    )
    vocab = int(model.get_input_embeddings().weight.shape[0])
    model.train()
    s1_batch = int(s1_cfg["training"]["batch_size"])
    s1_stats = _train_lm(model, vocab, EVENT_CONTEXT, s1_batch)
    _emit(bucket, "event_token_stage1", {
        "context": EVENT_CONTEXT,
        "vocab_size": vocab,
        "stage1_gpu_h": gpu_hours(
            EVENT_S1_EXAMPLES, 1, s1_batch, s1_stats["sec_per_step_median"]
        ),
        "train_microbatch": s1_batch,
        "n_examples": EVENT_S1_EXAMPLES,
        "peak_vram_gib": s1_stats["peak_allocated_gib"],
        "sec_per_step_median": s1_stats["sec_per_step_median"],
        "notes": (
            "4-bit Mistral + LoRA + trainable event embeddings, context 2048, "
            f"microbatch {s1_batch}, 1 epoch on {EVENT_S1_EXAMPLES} packed chunks."
        ),
    })

    # Stage 2 trains LoRA and the head. Embeddings stay frozen.
    for param_name, param in model.named_parameters():
        if "embed" in param_name or "lm_head" in param_name:
            param.requires_grad_(False)
    hidden = int(s2_cfg["model"]["hidden_size"])
    clf = LLMClassifier(
        base_model=model,
        hidden_size=hidden,
        num_labels=2,
        freeze_base=True,
        trainable_param_keywords=["lora_"],
        pooling="last_token",
    ).to("cuda")
    s2_batch = int(s2_cfg["training"]["batch_size"])
    s2_stats, ids, mask = _train_cls(clf, vocab, EVENT_CONTEXT, s2_batch)
    _emit(bucket, "event_token_stage2", {
        "context": EVENT_CONTEXT,
        "stage2_gpu_h": gpu_hours(N_TRAIN, 1, s2_batch, s2_stats["sec_per_step_median"]),
        "train_microbatch": s2_batch,
        "n_examples": N_TRAIN,
        "epochs": 1,
        "peak_vram_gib": s2_stats["peak_allocated_gib"],
        "sec_per_step_median": s2_stats["sec_per_step_median"],
        "notes": f"1 epoch on {N_TRAIN} ICU-mortality examples, context {EVENT_CONTEXT}.",
    })
    clf.eval()
    one_ids = ids[:1]
    one_mask = mask[:1]

    def infer():
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                clf(input_ids=one_ids, attention_mask=one_mask)

    inf = timed(infer)
    _emit(bucket, "event_token_latency_bs1", {
        "context": EVENT_CONTEXT,
        "infer_batch_size": 1,
        "latency_ms": inf["sec_per_step_median"] * 1000.0,
        "peak_vram_gib": inf["peak_allocated_gib"],
    })
    del clf, model, FastLanguageModel
    release()


def bench_clmbr_frozen_bs1(bucket: dict) -> None:
    import torch
    from src.clmbr.embed import ATTENTION_WIDTH, causal_local_mask_padded
    from src.clmbr.model import load_clmbr_transformer

    print("\n===== CLMBR-T frozen batch-1 =====", flush=True)
    device = torch.device("cuda")
    ft_cfg = cfg("clmbr_ft")
    model_dir = cfg("clmbr")["clmbr"]["model_dir"]
    seq = int(ft_cfg["clmbr"]["max_seq_len"])
    backbone = load_clmbr_transformer(model_dir, device)
    backbone.eval()
    tokens = torch.randint(0, 65536, (1, seq), device=device)
    ages = torch.rand(1, seq, device=device) * 30000
    lengths = torch.ones(1, dtype=torch.long, device=device) * seq
    mask = causal_local_mask_padded(lengths, seq, ATTENTION_WIDTH, device)
    use_bf16 = torch.cuda.is_bf16_supported()

    def infer():
        with torch.no_grad():
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else torch.autocast("cuda", enabled=False)
            )
            with ctx:
                backbone(tokens, ages, mask)

    stats = timed(infer)
    _emit(bucket, "clmbr_frozen_latency_bs1", {
        "context": seq,
        "infer_batch_size": 1,
        "latency_ms": stats["sec_per_step_median"] * 1000.0,
        "peak_vram_gib": stats["peak_allocated_gib"],
        "notes": "Batch-1 bf16 forward of the frozen CLMBR-T backbone, length 496.",
    })
    del backbone
    release()


def _recovered() -> dict:
    """Cells already measured on 8 Oct. Not re-timed."""
    path = OUT_DIR / "results.json"
    models = json.loads(path.read_text())["models"] if path.exists() else {}
    xgb = models.get("xgboost", {})
    xgb_lat = models.get("xgboost_latency_bs1", {})
    frozen = models.get("clmbr_frozen", {})
    return {
        "xgboost_stage2_cpu_h": xgb.get("stage2_cpu_h"),
        "xgboost_latency_ms_bs1": xgb_lat.get("latency_ms"),
        "xgboost_note": (
            "CPU hist trees, not GPU-hours. Stage 1 does not exist. "
            "Latency is one row after a 100-tree fit."
        ),
        "clmbr_frozen_stage2_gpu_h": frozen.get("stage2_gpu_h"),
        "clmbr_frozen_stage2_note": frozen.get("notes"),
    }


def main() -> None:
    import argparse

    import unsloth  # noqa: F401
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset: clmbr event qwen0.6 qwen8",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Submit scripts/site_c/benchmark_table_gaps.sh.")
    print("device", torch.cuda.get_device_name(0), flush=True)
    STATE.clear()
    STATE["recovered"] = _recovered()
    previous = {}
    if SUMMARY_PATH.exists():
        try:
            previous = json.loads(SUMMARY_PATH.read_text()).get("measured", {})
        except json.JSONDecodeError:
            previous = {}
    STATE["measured"] = previous
    measured = STATE["measured"]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(STATE, indent=2))

    jobs = [
        ("clmbr", lambda: bench_clmbr_frozen_bs1(measured)),
        ("event", lambda: bench_event_token(measured)),
        ("qwen0.6", lambda: bench_tehr("qwen0.6", measured)),
        ("qwen8", lambda: bench_tehr("qwen8", measured)),
    ]
    if args.only:
        wanted = set(args.only)
        jobs = [job for job in jobs if job[0] in wanted]
    for name, fn in jobs:
        try:
            fn()
        except Exception as exc:
            print(f"\nFAILED {name}: {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            _emit(measured, f"{name}_error", {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            release()
    print("\nWrote", SUMMARY_PATH, flush=True)


if __name__ == "__main__":
    main()
