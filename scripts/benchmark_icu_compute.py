#!/usr/bin/env python3
"""
10-step compute estimate for the ICU-mortality column of the compute table.

Does not load MIMIC. Each configuration times 1 warmup step plus 10 steps on a
synthetic batch at that configuration's maximum context. GPU-hours extrapolate
as (examples * epochs / microbatch) * seconds_per_step / 3600, which is the
GPU-hour total under linear scaling. Full-context steps are an upper bound:
real batches are shorter, and data loading is not included.

The code transformer is measured at 2,048 events (encoder_ctx2048).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Cohort sizes taken from the completed ICU-mortality logs. Do not reload data.
N_TRAIN = 75_050
N_TUNE = 9_213
N_TEST = 9_343
# Encoder stage-1 MLM uses every MIMIC trajectory (train + tuning + held_out).
N_MLM = 291_702 + 36_463 + 36_462  # 364_627
# Phrase-embedding cache shape from the completed Qwen3-Embedding run.
N_PHRASES = 50_000
# Completed tEHR stage-1 run: checkpoint-14000, max_steps 14426, 2 GPUs,
# per-device batch 8, context 8192. Used only to scale the token budget.
TEHR_S1_SEQUENCES_AT_8192 = 14_426 * 2 * 8
TEHR_S1_CONTEXT_COMPLETED = 8192
# Paper ICU encoder vocabulary (including specials), from the 512-event log.
ENCODER_VOCAB = 45_733

STEPS = 10
WARMUP = 1
OUT_DIR = Path(
    os.environ.get(
        "COMPUTE_BENCH_OUT",
        "/hx2-weka/home/ag619/TextCancEHR2/tEHR-Data/Results/icu_compute_benchmark",
    )
)


def gpu_hours(n_examples: float, epochs: float, microbatch: int, sec_per_step: float) -> float:
    """GPU-hours for n_examples * epochs microsteps. Independent of GPU count."""
    return (n_examples * epochs / microbatch) * sec_per_step / 3600.0


def latency_ms(patients_per_s: float) -> float:
    return 1000.0 / patients_per_s if patients_per_s else float("nan")


def count_params(module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return int(total), int(trainable)


def save(results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "results.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2, default=str))
    tmp.replace(path)
    print(f"  wrote {path}", flush=True)


def release(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def timed(step_fn, steps: int = STEPS, warmup: int = WARMUP) -> dict:
    import torch

    torch.cuda.synchronize()
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step_fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return {
        "sec_per_step_median": statistics.median(times),
        "sec_per_step_mean": statistics.mean(times),
        "sec_per_step_min": min(times),
        "sec_per_step_max": max(times),
        "peak_allocated_gib": peak,
        "n_timed_steps": steps,
        "n_warmup": warmup,
    }


def run_one(results: dict, name: str, fn, *, split: bool = False) -> None:
    print(f"\n===== {name} =====", flush=True)
    t0 = time.perf_counter()
    try:
        row = fn()
        row["wall_sec"] = time.perf_counter() - t0
        row["status"] = "ok"
    except Exception as exc:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_sec": time.perf_counter() - t0,
        }
        print(row["traceback"], flush=True)
    if split and row.get("status") == "ok":
        wall = row.pop("wall_sec", None)
        row.pop("status", None)
        for key, sub in row.items():
            if isinstance(sub, dict):
                sub["group_wall_sec"] = wall
                results["models"][key] = sub
                print(key, json.dumps({
                    k: sub.get(k) for k in (
                        "context", "stage1_gpu_h", "stage2_gpu_h",
                        "peak_vram_gib", "patients_per_s", "latency_ms",
                    ) if k in sub
                }), flush=True)
    else:
        results["models"][name] = row
        print(json.dumps({k: row.get(k) for k in (
            "status", "context", "stage1_gpu_h", "stage2_gpu_h",
            "peak_vram_gib", "patients_per_s", "latency_ms", "error",
        ) if k in row or k == "status"}, indent=2), flush=True)
    save(results)
    release()


def cfg(experiment: str, model: str | None = None) -> dict:
    from src.training.config_utils import build_config

    return build_config("mimic", experiment, model=model, task="icu_mortality")


def bench_xgboost() -> dict:
    import numpy as np
    from scipy import sparse
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    n_features = 40_000
    nnz = 200
    rows = np.repeat(np.arange(N_TRAIN), nnz)
    cols = rng.integers(0, n_features, size=rows.shape[0], dtype=np.int32)
    data = rng.random(rows.shape[0]).astype(np.float32)
    x_train = sparse.csr_matrix((data, (rows, cols)), shape=(N_TRAIN, n_features))
    y_train = rng.integers(0, 2, size=N_TRAIN)
    n_trees_timed = 10
    n_trees = 100
    model = XGBClassifier(
        n_estimators=n_trees_timed,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=8,
        random_state=42,
    )
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_sec = time.perf_counter() - t0
    # One more fit of the same 10 trees is not a median-of-steps, but tree
    # building is one shot. Scale trees linearly; the matrix is already the
    # full ICU train size.
    train_sec_full = train_sec * (n_trees / n_trees_timed)

    rows_te = np.repeat(np.arange(N_TEST), nnz)
    cols_te = rng.integers(0, n_features, size=rows_te.shape[0], dtype=np.int32)
    data_te = rng.random(rows_te.shape[0]).astype(np.float32)
    x_test = sparse.csr_matrix((data_te, (rows_te, cols_te)), shape=(N_TEST, n_features))
    _ = model.predict_proba(x_test[:64])
    t1 = time.perf_counter()
    model.predict_proba(x_test)
    infer_sec = time.perf_counter() - t1
    patients_per_s = N_TEST / infer_sec
    return {
        "total_params": None,
        "trainable_params": None,
        "context": None,
        "context_unit": "bag-of-words",
        "stage1_gpu_h": None,
        "stage2_gpu_h": None,
        "stage2_cpu_h": train_sec_full / 3600.0,
        "peak_vram_gib": None,
        "patients_per_s": patients_per_s,
        "latency_ms": latency_ms(patients_per_s),
        "device": "cpu",
        "notes": (
            f"CPU hist, {n_trees_timed} trees on {N_TRAIN} synthetic rows "
            f"({n_features} features, {nnz} nnz) scaled by {n_trees}/{n_trees_timed}. "
            "Not GPU-hours. Tree cost is not perfectly linear in n_estimators."
        ),
        "detail": {"timed_train_sec": train_sec, "infer_sec": infer_sec},
    }


def bench_encoder(num_layers: int | None = None) -> dict:
    import torch
    import torch.nn as nn

    from src.baselines.encoder import EHRTransformerEncoder, EHRTransformerMLMHead

    device = torch.device("cuda")
    conf = cfg("encoder_ctx2048")
    model_cfg = conf["model"]
    pre = conf["pretrain"]
    train = conf["training"]
    seq = int(model_cfg["max_seq_length"])
    if seq != 2048:
        raise RuntimeError(f"encoder_ctx2048 max_seq_length is {seq}, expected 2048")
    vocab = ENCODER_VOCAB
    batch = int(pre["batch_size"])
    epochs_s1 = int(pre["epochs"])
    epochs_s2 = int(train["epochs"])
    mlm_p = float(pre.get("mlm_probability", 0.15))
    layers = int(num_layers if num_layers is not None else model_cfg["num_layers"])
    heads = int(model_cfg["num_heads"])

    encoder = EHRTransformerEncoder(
        vocab_size=vocab,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=layers,
        num_heads=heads,
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        max_seq_length=seq,
        num_labels=2,
        head_type=model_cfg.get("head_type", "linear"),
        head_hidden_size=int(model_cfg.get("head_hidden_size", 256)),
        pooling=model_cfg.get("pooling", "mean"),
    ).to(device)
    mlm = EHRTransformerMLMHead(encoder, vocab).to(device)
    ids = torch.randint(0, vocab, (batch, seq), device=device)
    mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    labels = torch.full((batch, seq), -100, dtype=torch.long, device=device)
    keep = torch.rand(batch, seq, device=device) < mlm_p
    labels = torch.where(keep, ids, labels)
    opt = torch.optim.AdamW(mlm.parameters(), lr=1e-4, weight_decay=0.01)
    mlm.train()

    def train_mlm():
        opt.zero_grad(set_to_none=True)
        loss, _ = mlm(ids, mask, labels)
        loss.backward()
        opt.step()

    s1 = timed(train_mlm)
    total_s1, trainable_s1 = count_params(mlm)
    del opt, mlm, labels, keep
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 2 reuses the same encoder weights. A fresh head matches the recipe
    # (MLM head is dropped; the classification head is trained).
    y = torch.randint(0, 2, (batch,), device=device)
    opt2 = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=0.01)
    encoder.train()

    def train_cls():
        opt2.zero_grad(set_to_none=True)
        logits = encoder(ids, mask)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt2.step()

    s2 = timed(train_cls)
    total_s2, trainable_s2 = count_params(encoder)
    del opt2
    encoder.eval()

    def infer():
        with torch.no_grad():
            encoder(ids, mask)

    inf = timed(infer)
    sec = inf["sec_per_step_median"]
    pps = batch / sec
    return {
        "total_params": total_s2,
        "trainable_params_stage1": trainable_s1,
        "trainable_params": trainable_s2,
        "total_params_stage1_with_mlm_head": total_s1,
        "context": seq,
        "context_unit": "events",
        "vocab_size": vocab,
        "stage1_gpu_h": gpu_hours(N_MLM, epochs_s1, batch, s1["sec_per_step_median"]),
        "stage2_gpu_h": gpu_hours(N_TRAIN, epochs_s2, batch, s2["sec_per_step_median"]),
        "peak_vram_gib": max(s1["peak_allocated_gib"], s2["peak_allocated_gib"], inf["peak_allocated_gib"]),
        "peak_train_stage1_gib": s1["peak_allocated_gib"],
        "peak_train_stage2_gib": s2["peak_allocated_gib"],
        "peak_infer_gib": inf["peak_allocated_gib"],
        "patients_per_s": pps,
        "latency_ms": latency_ms(pps),
        "train_batch_size": batch,
        "infer_batch_size": batch,
        "num_layers": layers,
        "num_heads": heads,
        "notes": (
            f"{layers}-layer {heads}-head encoder, context {seq} events. "
            f"Stage-1 is {epochs_s1} MLM epochs on {N_MLM} trajectories, batch {batch}. "
            f"Stage-2 is {epochs_s2} epochs on {N_TRAIN} ICU-mortality examples. "
            f"Vocab fixed at the paper ICU size {vocab}. Full-length batches, so times are an upper bound."
        ),
        "detail": {"stage1": s1, "stage2": s2, "inference": inf},
    }


def bench_text_embedding() -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from src.baselines.text_embedding import TextEmbeddingTransformer
    from src.pipelines.precompute_token_embeddings import last_token_pool

    device = torch.device("cuda")
    conf = cfg("text_embedding")
    embedder = conf["embedder"]["model_name"]
    model_cfg = conf["model"]
    train = conf["training"]
    seq = int(model_cfg["max_seq_length"])
    batch = int(train["batch_size"])
    epochs = int(train["epochs"])
    embed_dim = 1024

    tokenizer = AutoTokenizer.from_pretrained(embedder, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModel.from_pretrained(embedder, trust_remote_code=True, torch_dtype=torch.float32).to(device)
    qwen.eval()
    phrase = "serum glucose 120 mg/dl " * 30
    enc = tokenizer(
        [phrase] * 256,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    def embed_step():
        with torch.no_grad():
            out = qwen(**enc)
            last_token_pool(out.last_hidden_state, enc["attention_mask"])

    s1 = timed(embed_step)
    qwen_total, _ = count_params(qwen)
    n_embed_steps = (N_PHRASES + 255) // 256
    stage1 = n_embed_steps * s1["sec_per_step_median"] / 3600.0
    del qwen, enc, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    net = TextEmbeddingTransformer(
        embed_dim=embed_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        max_seq_length=seq,
        num_labels=2,
        head_type=model_cfg.get("head_type", "linear"),
        head_hidden_size=int(model_cfg.get("head_hidden_size", 256)),
        pooling=model_cfg.get("pooling", "mean"),
        pos_init_std=float(model_cfg.get("pos_init_std", 0.02)),
    ).to(device)
    emb = torch.randn(batch, seq, embed_dim, device=device)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    y = torch.randint(0, 2, (batch,), device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    net.train()

    def train_step():
        opt.zero_grad(set_to_none=True)
        logits = net(emb, mask)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

    s2 = timed(train_step)
    total, trainable = count_params(net)
    del opt
    net.eval()

    def infer():
        with torch.no_grad():
            net(emb, mask)

    inf = timed(infer)
    pps = batch / inf["sec_per_step_median"]
    return {
        "total_params": qwen_total + total,
        "embedder_params": qwen_total,
        "transformer_params": total,
        "trainable_params": trainable,
        "context": seq,
        "context_unit": "events",
        "stage1_gpu_h": stage1,
        "stage1_note": "frozen phrase embedding (dagger); not continued pretraining",
        "stage2_gpu_h": gpu_hours(N_TRAIN, epochs, batch, s2["sec_per_step_median"]),
        "peak_vram_gib": max(s1["peak_allocated_gib"], s2["peak_allocated_gib"], inf["peak_allocated_gib"]),
        "patients_per_s": pps,
        "latency_ms": latency_ms(pps),
        "infer_batch_size": batch,
        "notes": (
            f"Stage-1 is {N_PHRASES} phrases through {embedder} at batch 256, "
            f"max length 128, float32, matching precompute_token_embeddings.py. "
            f"Stage-2 is the 512-event transformer on frozen 1024-d vectors, "
            f"batch {batch}, {epochs} epoch."
        ),
        "detail": {"stage1": s1, "stage2": s2, "inference": inf},
    }


def bench_clmbr() -> dict:
    import torch
    from src.clmbr.embed import ATTENTION_WIDTH, causal_local_mask
    from src.clmbr.model import ClmbrClassifier, load_clmbr_transformer

    device = torch.device("cuda")
    frozen_cfg = cfg("clmbr")
    ft_cfg = cfg("clmbr_ft")
    model_dir = frozen_cfg["clmbr"]["model_dir"]
    seq = int(ft_cfg["clmbr"]["max_seq_len"])
    ft_batch = int(ft_cfg["training"]["batch_size"])
    ft_epochs = int(ft_cfg["training"]["epochs"])
    eval_batch = int(ft_cfg["clmbr"].get("eval_batch_size", 16))
    lr = float(ft_cfg["training"]["learning_rate"])
    wd = float(ft_cfg["training"]["weight_decay"])

    backbone = load_clmbr_transformer(model_dir, device)
    backbone.eval()
    tokens = torch.randint(0, 65536, (seq,), device=device)
    ages = torch.rand(seq, device=device) * 30000
    mask = causal_local_mask(seq, ATTENTION_WIDTH, device)

    def frozen_forward():
        with torch.no_grad():
            backbone(tokens, ages, mask)

    frozen = timed(frozen_forward)
    # Frozen probe embeds every labeled split once, one sequence at a time.
    n_embed = N_TRAIN + N_TUNE + N_TEST
    frozen_gpu_h = n_embed * frozen["sec_per_step_median"] / 3600.0
    frozen_pps = 1.0 / frozen["sec_per_step_median"]
    total_backbone, _ = count_params(backbone)

    classifier = ClmbrClassifier(backbone).to(device)
    from src.clmbr.embed import causal_local_mask_padded

    b_tokens = torch.randint(0, 65536, (ft_batch, seq), device=device)
    b_ages = torch.rand(ft_batch, seq, device=device) * 30000
    lengths = torch.full((ft_batch,), seq, dtype=torch.long, device=device)
    b_mask = causal_local_mask_padded(lengths, seq, ATTENTION_WIDTH, device)
    y = torch.randint(0, 2, (ft_batch,), device=device).float()
    opt = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)
    use_bf16 = torch.cuda.is_bf16_supported()
    classifier.train()

    def ft_step():
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
        with ctx:
            logits = classifier(b_tokens, b_ages, b_mask, lengths)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

    ft = timed(ft_step)
    total_ft, trainable_ft = count_params(classifier)
    del opt
    classifier.eval()

    e_tokens = torch.randint(0, 65536, (eval_batch, seq), device=device)
    e_ages = torch.rand(eval_batch, seq, device=device) * 30000
    e_lengths = torch.full((eval_batch,), seq, dtype=torch.long, device=device)
    e_mask = causal_local_mask_padded(e_lengths, seq, ATTENTION_WIDTH, device)

    def ft_infer():
        with torch.no_grad():
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
            with ctx:
                classifier(e_tokens, e_ages, e_mask, e_lengths)

    inf = timed(ft_infer)
    ft_pps = eval_batch / inf["sec_per_step_median"]
    return {
        "clmbr_frozen": {
            "total_params": total_backbone,
            "trainable_params": 0,
            "context": seq,
            "context_unit": "events",
            "stage1_gpu_h": None,
            "stage1_note": "External",
            "stage2_gpu_h": frozen_gpu_h,
            "peak_vram_gib": frozen["peak_allocated_gib"],
            "patients_per_s": frozen_pps,
            "latency_ms": latency_ms(frozen_pps),
            "infer_batch_size": 1,
            "notes": (
                f"Stage-1 is the public CLMBR-T checkpoint ({model_dir}). "
                f"Stage-2 GPU time is one forward per labeled example "
                f"({n_embed} = train+tuning+test), matching embed_sequences "
                f"(unbatched, length {seq}). Logistic regression is CPU and is not included."
            ),
            "detail": frozen,
        },
        "clmbr_finetuned": {
            "total_params": total_ft,
            "trainable_params": trainable_ft,
            "context": seq,
            "context_unit": "events",
            "stage1_gpu_h": None,
            "stage1_note": "External",
            "stage2_gpu_h": gpu_hours(N_TRAIN, ft_epochs, ft_batch, ft["sec_per_step_median"]),
            "peak_vram_gib": max(ft["peak_allocated_gib"], inf["peak_allocated_gib"]),
            "peak_train_gib": ft["peak_allocated_gib"],
            "peak_infer_gib": inf["peak_allocated_gib"],
            "patients_per_s": ft_pps,
            "latency_ms": latency_ms(ft_pps),
            "infer_batch_size": eval_batch,
            "notes": (
                f"{ft_epochs} epochs, batch {ft_batch}, length {seq}, bf16 autocast, "
                f"AdamW. Early stopping is ignored, so this is a full {ft_epochs}-epoch upper bound. "
                f"Inference batch {eval_batch}."
            ),
            "detail": {"stage2": ft, "inference": inf},
        },
    }


def bench_clmbr_matched_infer() -> dict:
    """Inference for frozen and fine-tuned CLMBR-T at the same batch."""
    import torch
    from src.clmbr.embed import ATTENTION_WIDTH, causal_local_mask_padded
    from src.clmbr.model import ClmbrClassifier, load_clmbr_transformer

    device = torch.device("cuda")
    frozen_cfg = cfg("clmbr")
    ft_cfg = cfg("clmbr_ft")
    model_dir = frozen_cfg["clmbr"]["model_dir"]
    seq = int(ft_cfg["clmbr"]["max_seq_len"])
    batch = int(ft_cfg["clmbr"].get("eval_batch_size", 16))
    use_bf16 = torch.cuda.is_bf16_supported()

    backbone = load_clmbr_transformer(model_dir, device)
    backbone.eval()
    tokens = torch.randint(0, 65536, (batch, seq), device=device)
    ages = torch.rand(batch, seq, device=device) * 30000
    lengths = torch.full((batch,), seq, dtype=torch.long, device=device)
    mask = causal_local_mask_padded(lengths, seq, ATTENTION_WIDTH, device)

    def frozen_infer():
        with torch.no_grad():
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else torch.autocast("cuda", enabled=False)
            )
            with ctx:
                backbone(tokens, ages, mask)

    frozen = timed(frozen_infer)
    frozen_pps = batch / frozen["sec_per_step_median"]

    classifier = ClmbrClassifier(backbone).to(device)
    classifier.eval()

    def ft_infer():
        with torch.no_grad():
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else torch.autocast("cuda", enabled=False)
            )
            with ctx:
                classifier(tokens, ages, mask, lengths)

    ft = timed(ft_infer)
    ft_pps = batch / ft["sec_per_step_median"]
    return {
        "clmbr_frozen": {
            "patients_per_s": frozen_pps,
            "latency_ms": latency_ms(frozen_pps),
            "infer_batch_size": batch,
            "peak_infer_gib": frozen["peak_allocated_gib"],
            "infer_detail": frozen,
            "infer_note": (
                f"Matched inference at batch {batch}, length {seq}, bf16 autocast. "
                "Stage-2 GPU-hours remain the unbatched embedding pass."
            ),
        },
        "clmbr_finetuned": {
            "patients_per_s": ft_pps,
            "latency_ms": latency_ms(ft_pps),
            "infer_batch_size": batch,
            "peak_infer_gib": ft["peak_allocated_gib"],
            "infer_detail": ft,
            "infer_note": (
                f"Matched inference at batch {batch}, length {seq}, bf16 autocast."
            ),
        },
    }


def bench_modernbert() -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda")
    conf = cfg("bioclinical_modernbert")
    model_name = conf["model"]["model_name"]
    seq = int(conf["model"]["max_length"])
    batch = int(conf["training"]["batch_size"])
    eval_batch = int(conf["training"]["eval_batch_size"])
    epochs = int(conf["training"]["epochs"])
    attn = conf["model"].get("attn_implementation", "flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {"num_labels": 2}
    if attn:
        load_kwargs["attn_implementation"] = attn
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:
        print(f"  flash attention load failed ({type(exc).__name__}: {exc}); retrying default", flush=True)
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)
    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)
    vocab = int(model.config.vocab_size)
    total, trainable = count_params(model)

    def make_batch(bs: int):
        ids = torch.randint(0, vocab, (bs, seq), device=device)
        mask = torch.ones(bs, seq, dtype=torch.long, device=device)
        y = torch.randint(0, 2, (bs,), device=device)
        return ids, mask, y

    def train_at(bs: int):
        ids, mask, y = make_batch(bs)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        model.train()

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=mask, labels=y)
                loss = out.loss
            loss.backward()
            opt.step()

        stats = timed(step)
        del opt
        return stats, ids, mask

    used = batch
    try:
        s2, ids, mask = train_at(batch)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        used = max(1, batch // 2)
        print(f"  ModernBERT OOM at batch {batch}; retrying batch {used}", flush=True)
        s2, ids, mask = train_at(used)

    model.eval()
    ebs = min(eval_batch, used)
    e_ids, e_mask, _ = make_batch(ebs)

    def infer():
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(input_ids=e_ids, attention_mask=e_mask)

    inf = timed(infer)
    pps = ebs / inf["sec_per_step_median"]
    return {
        "total_params": total,
        "trainable_params": trainable,
        "context": seq,
        "context_unit": "tokens",
        "stage1_gpu_h": None,
        "stage2_gpu_h": gpu_hours(N_TRAIN, epochs, used, s2["sec_per_step_median"]),
        "train_microbatch": used,
        "configured_microbatch": batch,
        "peak_vram_gib": max(s2["peak_allocated_gib"], inf["peak_allocated_gib"]),
        "patients_per_s": pps,
        "latency_ms": latency_ms(pps),
        "infer_batch_size": ebs,
        "notes": (
            f"{model_name}, context {seq}, {epochs} epoch, bf16 autocast, "
            f"gradient checkpointing. Train microbatch {used} "
            f"(configured {batch}). Inference batch {ebs}."
        ),
        "detail": {"stage2": s2, "inference": inf},
    }


def _load_unsloth(model_name: str, max_seq_length: int):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return FastLanguageModel, model, tokenizer


def _lora(FastLanguageModel, model, lora_cfg: dict, use_rslora: bool):
    kwargs = dict(
        r=int(lora_cfg.get("r", 16)),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing=True,
        random_state=42,
    )
    if use_rslora:
        kwargs["use_rslora"] = True
    return FastLanguageModel.get_peft_model(model, **kwargs)


def _enter_training(FastLanguageModel, model):
    """Leave Unsloth inference mode before LoRA wrapping.

    ``for_training`` deletes ``_flag_for_generation``. On a PeftModel,
    ``hasattr`` follows the base model, but ``del`` does not, so the call
    must run on the unwrapped model.
    """
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)
    else:
        model.train()


def bench_mistral(on_row=None) -> dict:
    import torch

    zero_cfg = cfg("zero_shot", "mistral")
    sft_cfg = cfg("sft", "mistral")
    s1_cfg = cfg("tehr_stage1", "mistral")
    s2_cfg = cfg("tehr_stage2", "mistral")

    zero_len = int(zero_cfg["model"]["max_length"])
    sft_len = int(sft_cfg["model"]["max_length"])
    s1_len = int(s1_cfg["model"]["max_length"])
    s2_len = int(s2_cfg["model"]["max_length"])
    s1_batch = int(s1_cfg["training"]["batch_size"])
    s2_batch = int(s2_cfg["training"]["batch_size"])
    s2_eval = int(s2_cfg["training"]["eval_batch_size"])
    sft_batch = int(sft_cfg["training"]["batch_size"])
    zero_new = int(zero_cfg.get("inference", {}).get("max_new_tokens", 8))

    out: dict = {}

    def emit(key, row):
        out[key] = row
        print(f"  saved {key}", flush=True)
        if on_row is not None:
            on_row(key, row)

    # Instruct model: zero-shot, then prompt SFT.
    FastLanguageModel, model, tokenizer = _load_unsloth(
        zero_cfg["model"]["model_name"],
        max(zero_len + zero_new, sft_len),
    )
    vocab = int(getattr(tokenizer, "vocab_size", 32768))
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    z_ids = torch.randint(0, vocab, (1, zero_len), device="cuda")
    z_mask = torch.ones(1, zero_len, dtype=torch.long, device="cuda")

    def generate():
        with torch.no_grad():
            model.generate(
                input_ids=z_ids,
                attention_mask=z_mask,
                max_new_tokens=zero_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    zstats = timed(generate)
    z_total, z_train = count_params(model)
    zpps = 1.0 / zstats["sec_per_step_median"]
    emit("zero_shot", {
        "total_params": z_total,
        "trainable_params": z_train,
        "context": zero_len,
        "context_unit": "tokens",
        "stage1_gpu_h": None,
        "stage2_gpu_h": None,
        "peak_vram_gib": zstats["peak_allocated_gib"],
        "patients_per_s": zpps,
        "latency_ms": latency_ms(zpps),
        "infer_batch_size": 1,
        "notes": (
            f"No training. generate max_new_tokens={zero_new}, batch 1, "
            f"context {zero_len} (resolved zero_shot config)."
        ),
        "detail": zstats,
    })
    del z_ids, z_mask

    _enter_training(FastLanguageModel, model)
    model = _lora(FastLanguageModel, model, sft_cfg.get("lora", {}), use_rslora=False)
    model.train()
    sft_ids = torch.randint(0, vocab, (sft_batch, sft_len), device="cuda")
    sft_mask = torch.ones(sft_batch, sft_len, dtype=torch.long, device="cuda")
    sft_labels = torch.full_like(sft_ids, -100)
    sft_labels[:, -16:] = sft_ids[:, -16:]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    model.train()

    def sft_step():
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=sft_ids, attention_mask=sft_mask, labels=sft_labels).loss
        loss.backward()
        opt.step()

    sft_stats = timed(sft_step)
    sft_total, sft_trainable = count_params(model)
    del opt
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    model.eval()

    def sft_infer():
        with torch.no_grad():
            model(input_ids=sft_ids, attention_mask=sft_mask)

    sft_inf = timed(sft_infer)
    sft_pps = sft_batch / sft_inf["sec_per_step_median"]
    emit("prompt_sft", {
        "total_params": sft_total,
        "trainable_params": sft_trainable,
        "context": sft_len,
        "context_unit": "tokens",
        "stage1_gpu_h": None,
        "stage2_gpu_h": gpu_hours(N_TRAIN, 1, sft_batch, sft_stats["sec_per_step_median"]),
        "peak_vram_gib": max(sft_stats["peak_allocated_gib"], sft_inf["peak_allocated_gib"]),
        "patients_per_s": sft_pps,
        "latency_ms": latency_ms(sft_pps),
        "infer_batch_size": sft_batch,
        "notes": (
            f"Instruct LoRA, batch {sft_batch}, context {sft_len}, 1 epoch on {N_TRAIN}. "
            "Loss is on the last 16 tokens so the backward matches a short completion. "
            "This estimates a finished epoch; the on-disk SFT run stopped early."
        ),
        "detail": {"stage2": sft_stats, "inference": sft_inf},
    })
    del model, tokenizer, sft_ids, sft_mask, sft_labels, FastLanguageModel
    gc.collect()
    torch.cuda.empty_cache()

    # Base model: stage-1 continued pretraining, then stage-2 classifier.
    load_len = max(s1_len, s2_len)
    FastLanguageModel, model, tokenizer = _load_unsloth(
        s1_cfg["model"]["model_name"],
        load_len,
    )
    vocab = int(getattr(tokenizer, "vocab_size", 32768))
    _enter_training(FastLanguageModel, model)
    model = _lora(FastLanguageModel, model, s1_cfg.get("lora", {}), use_rslora=True)
    model.train()

    def train_lm(length: int, batch: int):
        ids = torch.randint(0, vocab, (batch, length), device="cuda")
        mask = torch.ones(batch, length, dtype=torch.long, device="cuda")
        opt_lm = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
        model.train()

        def step():
            opt_lm.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
            loss.backward()
            opt_lm.step()

        stats = timed(step)
        del opt_lm
        return stats

    used_s1 = s1_batch
    try:
        s1_stats = train_lm(s1_len, s1_batch)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        used_s1 = max(1, s1_batch // 2)
        print(f"  stage-1 OOM at batch {s1_batch}; retrying {used_s1}", flush=True)
        s1_stats = train_lm(s1_len, used_s1)
    sequences = TEHR_S1_SEQUENCES_AT_8192 * (TEHR_S1_CONTEXT_COMPLETED / s1_len)
    s1_total, s1_trainable = count_params(model)
    emit("tehr_stage1", {
        "total_params": s1_total,
        "trainable_params": s1_trainable,
        "context": s1_len,
        "context_unit": "tokens",
        "stage1_gpu_h": gpu_hours(sequences, 1, used_s1, s1_stats["sec_per_step_median"]),
        "stage2_gpu_h": None,
        "train_microbatch": used_s1,
        "configured_microbatch": s1_batch,
        "peak_vram_gib": s1_stats["peak_allocated_gib"],
        "sequences_extrapolated": sequences,
        "notes": (
            f"4-bit base + LoRA causal LM at context {s1_len}, microbatch {used_s1}. "
            f"Sequence count {sequences:.0f} scales the completed run "
            f"({TEHR_S1_SEQUENCES_AT_8192} sequences at {TEHR_S1_CONTEXT_COMPLETED}) "
            "by the token budget. Full-length upper bound."
        ),
        "detail": s1_stats,
    })

    from src.models.llm_classifier import LLMClassifier

    hidden = int(s2_cfg["model"]["hidden_size"])
    clf = LLMClassifier(
        base_model=model,
        hidden_size=hidden,
        num_labels=2,
        freeze_base=True,
        trainable_param_keywords=["lora_"],
        pooling=s2_cfg["model"].get("pooling", "last_token"),
    ).to("cuda")

    def train_cls(batch: int):
        ids = torch.randint(0, vocab, (batch, s2_len), device="cuda")
        mask = torch.ones(batch, s2_len, dtype=torch.long, device="cuda")
        y = torch.randint(0, 2, (batch,), device="cuda")
        opt_c = torch.optim.AdamW((p for p in clf.parameters() if p.requires_grad), lr=1e-6)
        clf.train()

        def step():
            opt_c.zero_grad(set_to_none=True)
            # Trainer bf16 autocast. The 4-bit backbone returns bf16 states and
            # the linear head is fp32; without autocast that matmul crashes.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = clf(input_ids=ids, attention_mask=mask, labels=y)["loss"]
            loss.backward()
            opt_c.step()

        stats = timed(step)
        del opt_c
        return stats, ids, mask

    used_s2 = s2_batch
    try:
        s2_stats, ids, mask = train_cls(s2_batch)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        used_s2 = max(1, s2_batch // 2)
        print(f"  stage-2 OOM at batch {s2_batch}; retrying {used_s2}", flush=True)
        s2_stats, ids, mask = train_cls(used_s2)
    ebs = min(s2_eval, used_s2)
    if ebs != ids.shape[0]:
        ids = torch.randint(0, vocab, (ebs, s2_len), device="cuda")
        mask = torch.ones(ebs, s2_len, dtype=torch.long, device="cuda")
    clf.eval()

    def infer():
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                clf(input_ids=ids, attention_mask=mask)

    inf = timed(infer)
    pps = ebs / inf["sec_per_step_median"]
    c_total, c_train = count_params(clf)
    emit("tehr_stage2", {
        "total_params": c_total,
        "trainable_params": c_train,
        "context": s2_len,
        "context_unit": "tokens",
        "stage1_gpu_h": None,
        "stage2_gpu_h": gpu_hours(N_TRAIN, 1, used_s2, s2_stats["sec_per_step_median"]),
        "train_microbatch": used_s2,
        "configured_microbatch": s2_batch,
        "peak_vram_gib": max(s2_stats["peak_allocated_gib"], inf["peak_allocated_gib"]),
        "patients_per_s": pps,
        "latency_ms": latency_ms(pps),
        "infer_batch_size": ebs,
        "notes": (
            f"4-bit + LoRA + linear head, context {s2_len}, microbatch {used_s2} "
            f"(configured {s2_batch}), 1 epoch on {N_TRAIN}. "
            f"Inference batch {ebs}. Fresh LoRA, not the checkpoint-14000 weights; "
            "step cost matches. Full-length upper bound."
        ),
        "detail": {"stage2": s2_stats, "inference": inf},
    })
    return out


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def bench_text_embedding_mistral() -> dict:
    """Frozen Mistral-7B phrase embeddings, then a 2048-event transformer."""
    import torch
    import torch.nn.functional as F

    from src.baselines.text_embedding import TextEmbeddingTransformer

    device = torch.device("cuda")
    conf = cfg("text_embedding")
    model_cfg = conf["model"]
    mistral = cfg("tehr_stage1", "mistral")
    seq = 2048
    epochs = int(conf["training"]["epochs"])
    phrase_len = 128
    n_phrases = N_PHRASES

    FastLanguageModel, model, tokenizer = _load_unsloth(
        mistral["model"]["model_name"],
        phrase_len,
    )
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    model.eval()
    vocab = int(getattr(tokenizer, "vocab_size", 32768))
    cursor = model
    layers = None
    for _ in range(4):
        if hasattr(cursor, "layers"):
            layers = cursor.layers
            break
        if hasattr(cursor, "model"):
            cursor = cursor.model
        else:
            break
    if layers is None:
        raise RuntimeError("Could not find Mistral decoder layers for phrase embedding")

    def embed_batch(bs: int):
        ids = torch.randint(0, vocab, (bs, phrase_len), device=device)
        mask = torch.ones(bs, phrase_len, dtype=torch.long, device=device)
        captured = {}

        def hook(_module, _inputs, output):
            captured["h"] = output[0] if isinstance(output, tuple) else output

        handle = layers[-1].register_forward_hook(hook)

        def step():
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
                captured["h"][:, -1, :]

        try:
            stats = timed(step)
        finally:
            handle.remove()
        return stats

    phrase_batch = 128
    s1 = None
    for bs in (128, 64, 32, 16):
        try:
            s1 = embed_batch(bs)
            phrase_batch = bs
            break
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            print(f"  Mistral phrase embed OOM at batch {bs}", flush=True)
            torch.cuda.empty_cache()
    if s1 is None:
        raise RuntimeError("Mistral phrase embedding OOM at batch 16")
    n_embed_steps = (n_phrases + phrase_batch - 1) // phrase_batch
    stage1_h = n_embed_steps * s1["sec_per_step_median"] / 3600.0
    peak_s1 = s1["peak_allocated_gib"]
    del model, tokenizer, FastLanguageModel, layers
    gc.collect()
    torch.cuda.empty_cache()

    embed_dim = 4096
    net = TextEmbeddingTransformer(
        embed_dim=embed_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        max_seq_length=seq,
        num_labels=2,
        head_type=model_cfg.get("head_type", "linear"),
        head_hidden_size=int(model_cfg.get("head_hidden_size", 256)),
        pooling=model_cfg.get("pooling", "mean"),
        pos_init_std=float(model_cfg.get("pos_init_std", 0.02)),
    ).to(device)

    def train_at(bs: int):
        emb = torch.randn(bs, seq, embed_dim, device=device)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        mask = torch.ones(bs, seq, dtype=torch.long, device=device)
        y = torch.randint(0, 2, (bs,), device=device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
        net.train()

        def step():
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(emb, mask), y)
            loss.backward()
            opt.step()

        stats = timed(step)
        del opt, emb
        return stats, mask

    s2 = None
    used = None
    mask = None
    for bs in (128, 64, 32, 16, 8):
        try:
            s2, mask = train_at(bs)
            used = bs
            break
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            print(f"  embedding transformer OOM at batch {bs}", flush=True)
            torch.cuda.empty_cache()
    if s2 is None:
        raise RuntimeError("2048-event embedding transformer OOM at batch 8")
    total, trainable = count_params(net)
    net.eval()
    # Inference at the largest batch that fitted, matching the training microbatch.
    emb = torch.randn(used, seq, embed_dim, device=device)
    emb = torch.nn.functional.normalize(emb, dim=-1)

    def infer():
        with torch.no_grad():
            net(emb, mask)

    inf = timed(infer)
    pps = used / inf["sec_per_step_median"]
    # Logical Mistral-7B count. The 4-bit module's numel is the packed storage.
    mistral_params = 7_248_023_552
    return {
        "total_params": mistral_params + total,
        "embedder_params": mistral_params,
        "transformer_params": total,
        "trainable_params": trainable,
        "context": seq,
        "context_unit": "events",
        "stage1_gpu_h": stage1_h,
        "stage1_note": "frozen Mistral-7B phrase embedding",
        "stage2_gpu_h": gpu_hours(N_TRAIN, epochs, used, s2["sec_per_step_median"]),
        "peak_vram_gib": max(peak_s1, s2["peak_allocated_gib"], inf["peak_allocated_gib"]),
        "patients_per_s": pps,
        "latency_ms": latency_ms(pps),
        "train_batch_size_stage1": phrase_batch,
        "train_batch_size": used,
        "infer_batch_size": used,
        "phrase_length": phrase_len,
        "notes": (
            f"Stage-1 embeds {n_phrases} phrases with 4-bit Mistral-7B, "
            f"batch {phrase_batch}, length {phrase_len}, last-layer last token. "
            f"Stage-2 is a {model_cfg['num_layers']}-layer transformer on frozen "
            f"4096-d vectors, context {seq} events, batch {used}, {epochs} epoch."
        ),
        "detail": {"stage1": s1, "stage2": s2, "inference": inf},
    }


def bench_tehr_at_context(context: int, on_row=None) -> dict:
    """tEHR stage-1 and stage-2 at one context length for both stages."""
    import torch

    from src.models.llm_classifier import LLMClassifier

    s1_cfg = cfg("tehr_stage1", "mistral")
    s2_cfg = cfg("tehr_stage2", "mistral")
    s1_batch = int(s1_cfg["training"]["batch_size"])
    s2_batch = int(s2_cfg["training"]["batch_size"])
    s2_eval = int(s2_cfg["training"]["eval_batch_size"])
    out: dict = {}

    def emit(key, row):
        out[key] = row
        print(f"  saved {key}", flush=True)
        if on_row is not None:
            on_row(key, row)

    FastLanguageModel, model, tokenizer = _load_unsloth(
        s1_cfg["model"]["model_name"],
        context,
    )
    vocab = int(getattr(tokenizer, "vocab_size", 32768))
    _enter_training(FastLanguageModel, model)
    model = _lora(FastLanguageModel, model, s1_cfg.get("lora", {}), use_rslora=True)
    model.train()

    def train_lm(batch: int):
        ids = torch.randint(0, vocab, (batch, context), device="cuda")
        mask = torch.ones(batch, context, dtype=torch.long, device="cuda")
        opt_lm = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
        model.train()

        def step():
            opt_lm.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
            loss.backward()
            opt_lm.step()

        stats = timed(step)
        del opt_lm
        return stats

    used_s1 = s1_batch
    try:
        s1_stats = train_lm(s1_batch)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        used_s1 = max(1, s1_batch // 2)
        print(f"  stage-1 OOM at batch {s1_batch}; retrying {used_s1}", flush=True)
        s1_stats = train_lm(used_s1)
    sequences = TEHR_S1_SEQUENCES_AT_8192 * (TEHR_S1_CONTEXT_COMPLETED / context)
    s1_total, s1_trainable = count_params(model)
    tag = f"ctx{context}"
    emit(f"tehr_stage1_{tag}", {
        "total_params": s1_total,
        "trainable_params": s1_trainable,
        "logical_total_params": 7_248_023_552 + 41_943_040,
        "logical_trainable_params": 41_943_040,
        "context": context,
        "context_unit": "tokens",
        "stage1_gpu_h": gpu_hours(sequences, 1, used_s1, s1_stats["sec_per_step_median"]),
        "stage2_gpu_h": None,
        "train_microbatch": used_s1,
        "configured_microbatch": s1_batch,
        "peak_vram_gib": s1_stats["peak_allocated_gib"],
        "sequences_extrapolated": sequences,
        "notes": (
            f"4-bit base + LoRA causal LM at context {context}, microbatch {used_s1}. "
            f"Sequence count {sequences:.0f} keeps the completed run's token budget "
            f"({TEHR_S1_SEQUENCES_AT_8192} sequences at {TEHR_S1_CONTEXT_COMPLETED})."
        ),
        "detail": s1_stats,
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

    def train_cls(batch: int):
        ids = torch.randint(0, vocab, (batch, context), device="cuda")
        mask = torch.ones(batch, context, dtype=torch.long, device="cuda")
        y = torch.randint(0, 2, (batch,), device="cuda")
        opt_c = torch.optim.AdamW((p for p in clf.parameters() if p.requires_grad), lr=1e-6)
        clf.train()

        def step():
            opt_c.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = clf(input_ids=ids, attention_mask=mask, labels=y)["loss"]
            loss.backward()
            opt_c.step()

        stats = timed(step)
        del opt_c
        return stats, ids, mask

    used_s2 = s2_batch
    try:
        s2_stats, ids, mask = train_cls(s2_batch)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        used_s2 = max(1, s2_batch // 2)
        print(f"  stage-2 OOM at batch {s2_batch}; retrying {used_s2}", flush=True)
        s2_stats, ids, mask = train_cls(used_s2)
    ebs = min(s2_eval, used_s2)
    if ebs != ids.shape[0]:
        ids = torch.randint(0, vocab, (ebs, context), device="cuda")
        mask = torch.ones(ebs, context, dtype=torch.long, device="cuda")
    clf.eval()

    def infer():
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                clf(input_ids=ids, attention_mask=mask)

    inf = timed(infer)
    pps = ebs / inf["sec_per_step_median"]
    c_total, c_train = count_params(clf)
    emit(f"tehr_stage2_{tag}", {
        "total_params": c_total,
        "trainable_params": c_train,
        "logical_total_params": 7_248_023_552 + 41_943_040 + 8194,
        "logical_trainable_params": 41_943_040 + 8194,
        "context": context,
        "context_unit": "tokens",
        "stage1_gpu_h": None,
        "stage2_gpu_h": gpu_hours(N_TRAIN, 1, used_s2, s2_stats["sec_per_step_median"]),
        "train_microbatch": used_s2,
        "configured_microbatch": s2_batch,
        "peak_vram_gib": max(s2_stats["peak_allocated_gib"], inf["peak_allocated_gib"]),
        "patients_per_s": pps,
        "latency_ms": latency_ms(pps),
        "infer_batch_size": ebs,
        "notes": (
            f"4-bit + LoRA + linear head, context {context}, microbatch {used_s2}, "
            f"1 epoch on {N_TRAIN}. Inference batch {ebs}."
        ),
        "detail": {"stage2": s2_stats, "inference": inf},
    })
    return out


def bench_ctx16000(on_row=None) -> dict:
    """Zero-shot and prompt SFT at 16,000 tokens, plus tEHR stage-2 at 16,000."""
    import torch

    from src.models.llm_classifier import LLMClassifier

    context = 16_000
    zero_cfg = cfg("zero_shot", "mistral")
    sft_cfg = cfg("sft", "mistral")
    s1_cfg = cfg("tehr_stage1", "mistral")
    s2_cfg = cfg("tehr_stage2", "mistral")
    zero_new = int(zero_cfg.get("inference", {}).get("max_new_tokens", 8))
    sft_batch = int(sft_cfg["training"]["batch_size"])
    s2_batch = int(s2_cfg["training"]["batch_size"])
    s2_eval = int(s2_cfg["training"]["eval_batch_size"])
    out: dict = {}

    def emit(key, row):
        out[key] = row
        print(f"  saved {key}", flush=True)
        if on_row is not None:
            on_row(key, row)

    FastLanguageModel, model, tokenizer = _load_unsloth(
        zero_cfg["model"]["model_name"],
        context + zero_new,
    )
    vocab = int(getattr(tokenizer, "vocab_size", 32768))
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    z_ids = torch.randint(0, vocab, (1, context), device="cuda")
    z_mask = torch.ones(1, context, dtype=torch.long, device="cuda")

    def generate():
        with torch.no_grad():
            model.generate(
                input_ids=z_ids,
                attention_mask=z_mask,
                max_new_tokens=zero_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    zstats = timed(generate)
    zpps = 1.0 / zstats["sec_per_step_median"]
    emit("zero_shot_ctx16000", {
        "logical_total_params": 7_248_023_552,
        "logical_trainable_params": 0,
        "context": context,
        "context_unit": "tokens",
        "stage1_gpu_h": None,
        "stage2_gpu_h": N_TEST / zpps / 3600.0,
        "stage2_note": "inference GPU-hours on the test cohort",
        "peak_vram_gib": zstats["peak_allocated_gib"],
        "patients_per_s": zpps,
        "latency_ms": latency_ms(zpps),
        "infer_batch_size": 1,
        "n_test": N_TEST,
        "notes": (
            f"No training. generate max_new_tokens={zero_new}, batch 1, "
            f"context {context}. Stage-2 GPU-hours score the {N_TEST}-patient test cohort."
        ),
        "detail": zstats,
    })
    del z_ids, z_mask

    _enter_training(FastLanguageModel, model)
    model = _lora(FastLanguageModel, model, sft_cfg.get("lora", {}), use_rslora=False)
    model.train()
    sft_ids = torch.randint(0, vocab, (sft_batch, context), device="cuda")
    sft_mask = torch.ones(sft_batch, context, dtype=torch.long, device="cuda")
    sft_labels = torch.full_like(sft_ids, -100)
    sft_labels[:, -16:] = sft_ids[:, -16:]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)

    def sft_step():
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=sft_ids, attention_mask=sft_mask, labels=sft_labels).loss
        loss.backward()
        opt.step()

    sft_stats = timed(sft_step)
    del opt
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    model.eval()

    def sft_infer():
        with torch.no_grad():
            model(input_ids=sft_ids, attention_mask=sft_mask)

    sft_inf = timed(sft_infer)
    sft_pps = sft_batch / sft_inf["sec_per_step_median"]
    emit("prompt_sft_ctx16000", {
        "logical_total_params": 7_248_023_552 + 41_943_040,
        "logical_trainable_params": 41_943_040,
        "context": context,
        "context_unit": "tokens",
        "stage1_gpu_h": None,
        "stage2_gpu_h": gpu_hours(N_TRAIN, 1, sft_batch, sft_stats["sec_per_step_median"]),
        "peak_vram_gib": max(sft_stats["peak_allocated_gib"], sft_inf["peak_allocated_gib"]),
        "patients_per_s": sft_pps,
        "latency_ms": latency_ms(sft_pps),
        "train_batch_size": sft_batch,
        "infer_batch_size": sft_batch,
        "notes": (
            f"Instruct LoRA, batch {sft_batch}, context {context}, 1 epoch on {N_TRAIN}. "
            "Loss is on the last 16 tokens."
        ),
        "detail": {"stage2": sft_stats, "inference": sft_inf},
    })
    del model, tokenizer, sft_ids, sft_mask, sft_labels, FastLanguageModel
    gc.collect()
    torch.cuda.empty_cache()

    FastLanguageModel, model, tokenizer = _load_unsloth(
        s1_cfg["model"]["model_name"],
        context,
    )
    vocab = int(getattr(tokenizer, "vocab_size", 32768))
    _enter_training(FastLanguageModel, model)
    model = _lora(FastLanguageModel, model, s1_cfg.get("lora", {}), use_rslora=True)
    hidden = int(s2_cfg["model"]["hidden_size"])
    clf = LLMClassifier(
        base_model=model,
        hidden_size=hidden,
        num_labels=2,
        freeze_base=True,
        trainable_param_keywords=["lora_"],
        pooling=s2_cfg["model"].get("pooling", "last_token"),
    ).to("cuda")

    def train_cls(batch: int):
        ids = torch.randint(0, vocab, (batch, context), device="cuda")
        mask = torch.ones(batch, context, dtype=torch.long, device="cuda")
        y = torch.randint(0, 2, (batch,), device="cuda")
        opt_c = torch.optim.AdamW((p for p in clf.parameters() if p.requires_grad), lr=1e-6)
        clf.train()

        def step():
            opt_c.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = clf(input_ids=ids, attention_mask=mask, labels=y)["loss"]
            loss.backward()
            opt_c.step()

        stats = timed(step)
        del opt_c
        return stats, ids, mask

    used_s2 = s2_batch
    try:
        s2_stats, ids, mask = train_cls(s2_batch)
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()
        used_s2 = max(1, s2_batch // 2)
        print(f"  stage-2 OOM at batch {s2_batch}; retrying {used_s2}", flush=True)
        s2_stats, ids, mask = train_cls(used_s2)
    ebs = min(s2_eval, used_s2)
    if ebs != ids.shape[0]:
        ids = torch.randint(0, vocab, (ebs, context), device="cuda")
        mask = torch.ones(ebs, context, dtype=torch.long, device="cuda")
    clf.eval()

    def infer():
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                clf(input_ids=ids, attention_mask=mask)

    inf = timed(infer)
    pps = ebs / inf["sec_per_step_median"]
    emit("tehr_stage2_ctx16000", {
        "logical_total_params": 7_248_023_552 + 41_943_040 + 8194,
        "logical_trainable_params": 41_943_040 + 8194,
        "context": context,
        "context_unit": "tokens",
        "stage2_gpu_h": gpu_hours(N_TRAIN, 1, used_s2, s2_stats["sec_per_step_median"]),
        "train_microbatch": used_s2,
        "configured_microbatch": s2_batch,
        "peak_vram_gib": max(s2_stats["peak_allocated_gib"], inf["peak_allocated_gib"]),
        "patients_per_s": pps,
        "latency_ms": latency_ms(pps),
        "infer_batch_size": ebs,
        "notes": (
            f"4-bit + LoRA + linear head, context {context}, microbatch {used_s2}, "
            f"1 epoch on {N_TRAIN}. Inference batch {ebs}. "
            "Stage-1 GPU-hours stay the earlier 16,000-token measurement."
        ),
        "detail": {"stage2": s2_stats, "inference": inf},
    })
    return out


def bench_table_gaps(on_row=None) -> dict:
    """Batch-1 latency for the table, plus the missing tEHR context lengths."""
    import numpy as np
    import torch
    from scipy import sparse
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from xgboost import XGBClassifier

    from src.baselines.encoder import EHRTransformerEncoder
    from src.baselines.text_embedding import TextEmbeddingTransformer
    from src.clmbr.embed import ATTENTION_WIDTH, causal_local_mask_padded
    from src.clmbr.model import ClmbrClassifier, load_clmbr_transformer
    from src.models.llm_classifier import LLMClassifier

    out: dict = {}

    def emit(key, row):
        out[key] = row
        print(f"  saved {key}", flush=True)
        if on_row is not None:
            on_row(key, row)

    def batch1_latency(step_fn):
        stats = timed(step_fn)
        sec = stats["sec_per_step_median"]
        return {
            "latency_ms": sec * 1000.0,
            "sec_per_example": sec,
            "infer_batch_size": 1,
            "detail": stats,
        }

    # XGBoost, one row after a 100-tree fit on a few thousand synthetic patients.
    rng = np.random.default_rng(0)
    n_features, nnz, n_fit = 40_000, 200, 4_000
    rows = np.repeat(np.arange(n_fit), nnz)
    cols = rng.integers(0, n_features, size=rows.shape[0], dtype=np.int32)
    data = rng.random(rows.shape[0]).astype(np.float32)
    x_fit = sparse.csr_matrix((data, (rows, cols)), shape=(n_fit, n_features))
    y_fit = rng.integers(0, 2, size=n_fit)
    x_one = x_fit[:1]
    booster = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=42,
    )
    booster.fit(x_fit, y_fit)
    booster.predict_proba(x_one)
    emit("xgboost_latency_bs1", batch1_latency(lambda: booster.predict_proba(x_one)))

    # ModernBERT, 8192 tokens, batch 1.
    device = torch.device("cuda")
    mb_name = "thomas-sounack/BioClinical-ModernBERT-base"
    try:
        mb = AutoModelForSequenceClassification.from_pretrained(
            mb_name, num_labels=2, attn_implementation="flash_attention_2"
        )
    except Exception:
        mb = AutoModelForSequenceClassification.from_pretrained(mb_name, num_labels=2)
    if hasattr(mb.config, "reference_compile"):
        mb.config.reference_compile = False
    if hasattr(mb.config, "use_cache"):
        mb.config.use_cache = False
    mb.to(device).eval()
    vocab_mb = int(mb.config.vocab_size)
    mb_ids = torch.randint(0, vocab_mb, (1, 8192), device=device)
    mb_mask = torch.ones(1, 8192, dtype=torch.long, device=device)

    def mb_infer():
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mb(input_ids=mb_ids, attention_mask=mb_mask)

    emit("modernbert_latency_bs1", batch1_latency(mb_infer))
    del mb, mb_ids, mb_mask
    gc.collect()
    torch.cuda.empty_cache()

    # 12-layer encoder, 2048 events, batch 1.
    encoder = EHRTransformerEncoder(
        vocab_size=ENCODER_VOCAB, hidden_dim=512, num_layers=12, num_heads=8,
        ff_dim=2048, dropout=0.1, max_seq_length=2048, num_labels=2,
        head_type="linear", pooling="mean",
    ).to(device).eval()
    enc_ids = torch.randint(0, ENCODER_VOCAB, (1, 2048), device=device)
    enc_mask = torch.ones(1, 2048, dtype=torch.long, device=device)

    def enc_infer():
        with torch.no_grad():
            encoder(enc_ids, enc_mask)

    emit("encoder12_latency_bs1", batch1_latency(enc_infer))
    del encoder, enc_ids, enc_mask
    gc.collect()
    torch.cuda.empty_cache()

    # Event-embedding transformer, 2048 events, batch 1. Embeddings are precomputed.
    net = TextEmbeddingTransformer(
        embed_dim=4096, hidden_dim=512, num_layers=6, num_heads=8, ff_dim=2048,
        dropout=0.1, max_seq_length=2048, num_labels=2, head_type="linear",
        pooling="mean",
    ).to(device).eval()
    emb = torch.nn.functional.normalize(torch.randn(1, 2048, 4096, device=device), dim=-1)
    emb_mask = torch.ones(1, 2048, dtype=torch.long, device=device)

    def emb_infer():
        with torch.no_grad():
            net(emb, emb_mask)

    emit("embedding_transformer_latency_bs1", batch1_latency(emb_infer))
    del net, emb, emb_mask
    gc.collect()
    torch.cuda.empty_cache()

    # CLMBR-T fine-tuned path, batch 1.
    ft_cfg = cfg("clmbr_ft")
    model_dir = cfg("clmbr")["clmbr"]["model_dir"]
    seq = int(ft_cfg["clmbr"]["max_seq_len"])
    backbone = load_clmbr_transformer(model_dir, device)
    clf_c = ClmbrClassifier(backbone).to(device).eval()
    c_tokens = torch.randint(0, 65536, (1, seq), device=device)
    c_ages = torch.rand(1, seq, device=device) * 30000
    c_lengths = torch.ones(1, dtype=torch.long, device=device) * seq
    c_mask = causal_local_mask_padded(c_lengths, seq, ATTENTION_WIDTH, device)
    use_bf16 = torch.cuda.is_bf16_supported()

    def clmbr_infer():
        with torch.no_grad():
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 else torch.autocast("cuda", enabled=False)
            )
            with ctx:
                clf_c(c_tokens, c_ages, c_mask, c_lengths)

    emit("clmbr_ft_latency_bs1", batch1_latency(clmbr_infer))
    del clf_c, backbone, c_tokens, c_ages, c_mask, c_lengths
    gc.collect()
    torch.cuda.empty_cache()

    # tEHR: missing training lengths, then batch-1 latency at four contexts.
    s1_cfg = cfg("tehr_stage1", "mistral")
    s2_cfg = cfg("tehr_stage2", "mistral")
    FastLanguageModel, model, tokenizer = _load_unsloth(s1_cfg["model"]["model_name"], 16_000)
    vocab = int(getattr(tokenizer, "vocab_size", 32768))
    _enter_training(FastLanguageModel, model)
    model = _lora(FastLanguageModel, model, s1_cfg.get("lora", {}), use_rslora=True)
    model.train()

    def train_lm(length, batch):
        ids = torch.randint(0, vocab, (batch, length), device="cuda")
        mask = torch.ones(batch, length, dtype=torch.long, device="cuda")
        opt_lm = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)

        def step():
            opt_lm.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
            loss.backward()
            opt_lm.step()

        stats = timed(step)
        del opt_lm
        return stats

    for length in (8192, 12_000):
        stats = train_lm(length, 4)
        emit(f"tehr_stage1_ctx{length}", {
            "context": length,
            "train_microbatch": 4,
            "sec_per_step_median": stats["sec_per_step_median"],
            "gpu_s_per_example": stats["sec_per_step_median"] / 4,
            "peak_vram_gib": stats["peak_allocated_gib"],
            "detail": stats,
        })

    hidden = int(s2_cfg["model"]["hidden_size"])
    clf = LLMClassifier(
        base_model=model, hidden_size=hidden, num_labels=2, freeze_base=True,
        trainable_param_keywords=["lora_"], pooling="last_token",
    ).to("cuda")

    def train_cls(length, batch):
        ids = torch.randint(0, vocab, (batch, length), device="cuda")
        mask = torch.ones(batch, length, dtype=torch.long, device="cuda")
        y = torch.randint(0, 2, (batch,), device="cuda")
        opt_c = torch.optim.AdamW((p for p in clf.parameters() if p.requires_grad), lr=1e-6)
        clf.train()

        def step():
            opt_c.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = clf(input_ids=ids, attention_mask=mask, labels=y)["loss"]
            loss.backward()
            opt_c.step()

        stats = timed(step)
        del opt_c
        return stats

    stats = train_cls(8192, 8)
    emit("tehr_stage2_ctx8192", {
        "context": 8192,
        "train_microbatch": 8,
        "sec_per_step_median": stats["sec_per_step_median"],
        "gpu_s_per_example": stats["sec_per_step_median"] / 8,
        "peak_vram_gib": stats["peak_allocated_gib"],
        "detail": stats,
    })
    clf.eval()

    for length in (4096, 8192, 12_000, 16_000):
        ids = torch.randint(0, vocab, (1, length), device="cuda")
        mask = torch.ones(1, length, dtype=torch.long, device="cuda")

        def infer(ids=ids, mask=mask):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    clf(input_ids=ids, attention_mask=mask)

        row = batch1_latency(infer)
        row["context"] = length
        emit(f"tehr_latency_bs1_ctx{length}", row)

    return out


def bench_embedding_bs16() -> dict:
    """Stage-2 event-embedding transformer at batch 16, context 2048."""
    import torch
    import torch.nn.functional as F

    from src.baselines.text_embedding import TextEmbeddingTransformer

    device = torch.device("cuda")
    conf = cfg("text_embedding")
    model_cfg = conf["model"]
    seq = 2048
    batch = 16
    net = TextEmbeddingTransformer(
        embed_dim=4096,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        max_seq_length=seq,
        num_labels=2,
        head_type=model_cfg.get("head_type", "linear"),
        head_hidden_size=int(model_cfg.get("head_hidden_size", 256)),
        pooling=model_cfg.get("pooling", "mean"),
        pos_init_std=float(model_cfg.get("pos_init_std", 0.02)),
    ).to(device)
    emb = torch.nn.functional.normalize(torch.randn(batch, seq, 4096, device=device), dim=-1)
    mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    y = torch.randint(0, 2, (batch,), device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    net.train()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(emb, mask), y)
        loss.backward()
        opt.step()

    stats = timed(step)
    return {
        "context": seq,
        "train_batch_size": batch,
        "sec_per_step_median": stats["sec_per_step_median"],
        "gpu_s_per_example": stats["sec_per_step_median"] / batch,
        "peak_vram_gib": stats["peak_allocated_gib"],
        "detail": stats,
    }


def bench_embedding_12l() -> dict:
    """12-layer event-embedding transformer, same width as the 12-layer encoder."""
    import torch
    import torch.nn.functional as F

    from src.baselines.text_embedding import TextEmbeddingTransformer

    device = torch.device("cuda")
    conf = cfg("encoder_ctx2048")
    model_cfg = conf["model"]
    seq = 2048
    batch = 16
    layers = 12
    net = TextEmbeddingTransformer(
        embed_dim=4096,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=layers,
        num_heads=int(model_cfg["num_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        max_seq_length=seq,
        num_labels=2,
        head_type=model_cfg.get("head_type", "linear"),
        head_hidden_size=int(model_cfg.get("head_hidden_size", 256)),
        pooling=model_cfg.get("pooling", "mean"),
        pos_init_std=0.02,
    ).to(device)
    total, trainable = count_params(net)
    emb = torch.nn.functional.normalize(torch.randn(batch, seq, 4096, device=device), dim=-1)
    mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    y = torch.randint(0, 2, (batch,), device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    net.train()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(emb, mask), y)
        loss.backward()
        opt.step()

    stats = timed(step)
    del opt, emb, y
    gc.collect()
    torch.cuda.empty_cache()

    net.eval()
    emb1 = torch.nn.functional.normalize(torch.randn(1, seq, 4096, device=device), dim=-1)
    mask1 = torch.ones(1, seq, dtype=torch.long, device=device)

    def infer():
        with torch.no_grad():
            net(emb1, mask1)

    inf = timed(infer)
    mistral_params = 7_248_023_552
    return {
        "context": seq,
        "num_layers": layers,
        "num_heads": int(model_cfg["num_heads"]),
        "hidden_dim": int(model_cfg["hidden_dim"]),
        "ff_dim": int(model_cfg["ff_dim"]),
        "embedder_params": mistral_params,
        "transformer_params": total,
        "total_params": mistral_params + total,
        "trainable_params": trainable,
        "train_batch_size": batch,
        "sec_per_step_median": stats["sec_per_step_median"],
        "gpu_s_per_example": stats["sec_per_step_median"] / batch,
        "peak_vram_gib": stats["peak_allocated_gib"],
        "latency_ms": inf["sec_per_step_median"] * 1000,
        "detail": {"stage2": stats, "latency_bs1": inf},
    }


def bench_new_method_latency(on_row=None) -> dict:
    """Compute for the two new experiments only.

    The code transformer is timed at 12 layers and 2048 events, matching the
    paper compute setting, with this experiment's 30 MLM epochs and 10
    supervised epochs. Phrase initialisation does not change the step cost.
    tEHR and the frozen-embedding transformer are not re-timed.

    llm_phrase inference calls the inner decoder with embeddings only.
    Unsloth's outer Mistral eval forward reads input_ids.shape and rejects
    input_ids together with inputs_embeds.
    """
    import torch
    import torch.nn as nn

    from src.baselines.encoder import EHRTransformerEncoder, EHRTransformerMLMHead
    from src.baselines.llm_phrase import PhraseLLMClassifier

    out: dict = {}

    def emit(key, row):
        out[key] = row
        print(f"  saved {key}", flush=True)
        if on_row is not None:
            on_row(key, row)

    device = torch.device("cuda")
    layers, seq, batch = 12, 2048, 16
    epochs_s1, epochs_s2 = 30, 10
    vocab = ENCODER_VOCAB
    encoder = EHRTransformerEncoder(
        vocab_size=vocab, hidden_dim=512, num_layers=layers, num_heads=8,
        ff_dim=2048, dropout=0.1, max_seq_length=seq, num_labels=2,
        head_type="linear", pooling="mean",
    ).to(device)
    mlm = EHRTransformerMLMHead(encoder, vocab).to(device)
    ids = torch.randint(0, vocab, (batch, seq), device=device)
    mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    mlm_labels = torch.full((batch, seq), -100, dtype=torch.long, device=device)
    keep = torch.rand(batch, seq, device=device) < 0.15
    mlm_labels = torch.where(keep, ids, mlm_labels)
    opt = torch.optim.AdamW(mlm.parameters(), lr=1e-4, weight_decay=0.01)
    mlm.train()

    def train_mlm():
        opt.zero_grad(set_to_none=True)
        loss, _ = mlm(ids, mask, mlm_labels)
        loss.backward()
        opt.step()

    s1 = timed(train_mlm)
    del opt, mlm, mlm_labels, keep
    gc.collect()
    torch.cuda.empty_cache()

    y = torch.randint(0, 2, (batch,), device=device)
    opt2 = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=0.01)
    encoder.train()

    def train_cls():
        opt2.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(encoder(ids, mask), y)
        loss.backward()
        opt2.step()

    s2 = timed(train_cls)
    del opt2
    one_ids = ids[:1]
    one_mask = mask[:1]
    encoder.eval()

    def enc_infer():
        with torch.no_grad():
            encoder(one_ids, one_mask)

    enc_stats = timed(enc_infer)
    emit("encoder_phrase_init_ctx2048_12l", {
        "context": seq,
        "num_layers": layers,
        "stage1_gpu_h": gpu_hours(N_MLM, epochs_s1, batch, s1["sec_per_step_median"]),
        "stage2_gpu_h": gpu_hours(N_TRAIN, epochs_s2, batch, s2["sec_per_step_median"]),
        "peak_vram_gib": max(s1["peak_allocated_gib"], s2["peak_allocated_gib"]),
        "infer_batch_size": 1,
        "latency_ms": enc_stats["sec_per_step_median"] * 1000.0,
        "peak_infer_gib": enc_stats["peak_allocated_gib"],
        "notes": (
            "12-layer code transformer, context 2048 events. Stage 1 is 30 MLM "
            "epochs on 364627 trajectories, batch 16. Stage 2 is 10 supervised "
            "epochs on 75050 ICU-mortality examples. Batch-1 inference. "
            "Full-length upper bound. Phrase init does not change the step cost."
        ),
        "detail": {"stage1": s1, "stage2": s2, "latency_bs1": enc_stats},
    })
    del encoder, ids, mask, y, one_ids, one_mask
    release()

    s2_cfg = cfg("tehr_stage2", "mistral")
    FastLanguageModel, model, _tokenizer = _load_unsloth(
        s2_cfg["model"]["model_name"], 2048,
    )
    _enter_training(FastLanguageModel, model)
    model = _lora(FastLanguageModel, model, s2_cfg.get("lora", {}), use_rslora=True)
    hidden = int(s2_cfg["model"]["hidden_size"])
    clf = PhraseLLMClassifier(
        base_model=model, hidden_size=hidden, embed_dim=1024, num_labels=2,
        freeze_base=True, trainable_param_keywords=["lora_"], pooling="last_token",
    ).to(device)
    seq = 2048
    train_bs = 8
    train_emb = torch.randn(train_bs, seq, 1024, device=device)
    train_mask = torch.ones(train_bs, seq, dtype=torch.long, device=device)
    train_y = torch.randint(0, 2, (train_bs,), device=device)
    opt = torch.optim.AdamW((p for p in clf.parameters() if p.requires_grad), lr=1e-6)
    clf.train()

    def train_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = clf(train_emb, train_mask, labels=train_y)["loss"]
        loss.backward()
        opt.step()

    train_stats = timed(train_step)
    stage2_h = gpu_hours(N_TRAIN, 1, train_bs, train_stats["sec_per_step_median"])
    emit("llm_phrase_stage2", {
        "context": seq,
        "embed_dim": 1024,
        "hidden_size": hidden,
        "stage1_gpu_h": None,
        "stage1_note": "none; loads the existing tEHR stage-1 checkpoint",
        "stage2_gpu_h": stage2_h,
        "train_microbatch": train_bs,
        "peak_vram_gib": train_stats["peak_allocated_gib"],
        "notes": (
            "4-bit Mistral-7B + LoRA + 1024→4096 projection, context 2048 events, "
            "microbatch 8, 1 epoch on 75050. Full-length upper bound. Fresh LoRA."
        ),
        "detail": train_stats,
    })

    one_emb = train_emb[:1]
    one_mask = train_mask[:1]
    clf.eval()

    def infer_step():
        # Eval on the outer Unsloth CausalLM reads input_ids.shape and also
        # rejects input_ids together with inputs_embeds. The training path
        # calls the inner decoder with embeddings only; time that same call.
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden_in = one_emb.to(dtype=clf.input_norm.weight.dtype)
                projected = clf.input_proj(clf.input_norm(hidden_in)).to(dtype=clf._embed_dtype())
                backbone = getattr(clf.base_model, "model", clf.base_model)
                inner = getattr(backbone, "model", None)
                decoder = inner if inner is not None else backbone
                decoder._has_no_labels = False
                outputs = decoder(
                    inputs_embeds=projected,
                    attention_mask=one_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                states = getattr(outputs, "last_hidden_state", None)
                if states is None:
                    states = outputs[0]
                clf.classifier(clf._pool(states, one_mask))

    infer_stats = timed(infer_step)
    emit("llm_phrase_latency_bs1", {
        "context": seq,
        "infer_batch_size": 1,
        "latency_ms": infer_stats["sec_per_step_median"] * 1000.0,
        "peak_vram_gib": infer_stats["peak_allocated_gib"],
        "notes": (
            "Batch-1 forward of projection + inner Mistral decoder + head at "
            "2048 events. The outer Unsloth eval wrapper is not used."
        ),
        "detail": infer_stats,
    })
    del clf, model, FastLanguageModel, opt, train_emb, train_mask, train_y, one_emb, one_mask
    release()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "Subset: xgboost encoder encoder12 text_embedding embedding_mistral "
            "clmbr modernbert mistral tehr4096 ctx16000 new_methods"
        ),
    )
    args = parser.parse_args()
    # Unsloth must patch transformers before any later Hugging Face import.
    import unsloth  # noqa: F401
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "task": "icu_mortality",
        "method": "10 full-context steps on synthetic batches; no MIMIC load",
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_train": N_TRAIN,
        "n_tune": N_TUNE,
        "n_test": N_TEST,
        "n_mlm_trajectories": N_MLM,
        "steps": STEPS,
        "models": {},
    }
    if (OUT_DIR / "results.json").exists() and args.only:
        previous = json.loads((OUT_DIR / "results.json").read_text())
        results["models"] = previous.get("models", {})
    save(results)

    wanted = set(args.only) if args.only else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    if want("xgboost"):
        run_one(results, "xgboost", bench_xgboost)
    if want("encoder"):
        run_one(results, "encoder_ctx2048", bench_encoder)
    if want("encoder12"):
        run_one(results, "encoder_12l_ctx2048", lambda: bench_encoder(num_layers=12))
    if want("embedding_mistral"):
        run_one(results, "text_embedding_mistral2048", bench_text_embedding_mistral)
    if want("tehr4096"):
        def on_tehr(key, row):
            results["models"][key] = row
            save(results)

        run_one(
            results,
            "tehr4096",
            lambda: bench_tehr_at_context(4096, on_tehr),
            split=True,
        )
    if want("text_embedding"):
        run_one(results, "text_embedding", bench_text_embedding)
    if want("clmbr"):
        run_one(results, "clmbr", bench_clmbr, split=True)
    if want("clmbr_infer"):
        print("\n===== clmbr matched inference =====", flush=True)
        measured = bench_clmbr_matched_infer()
        for key, update in measured.items():
            row = results["models"].get(key, {})
            row.update(update)
            results["models"][key] = row
            print(key, json.dumps({
                "infer_batch_size": update["infer_batch_size"],
                "patients_per_s": update["patients_per_s"],
                "latency_ms": update["latency_ms"],
            }), flush=True)
        save(results)
    if want("modernbert"):
        run_one(results, "bioclinical_modernbert", bench_modernbert)
    if want("embedding_bs16"):
        run_one(results, "text_embedding_mistral2048_bs16", bench_embedding_bs16)
    if want("embedding_12l"):
        run_one(results, "text_embedding_mistral2048_12l", bench_embedding_12l)
    if want("gaps"):
        def on_gap(key, row):
            results["models"][key] = row
            save(results)

        run_one(results, "gaps", lambda: bench_table_gaps(on_gap), split=True)
    if want("ctx16000"):
        def on_ctx(key, row):
            results["models"][key] = row
            save(results)

        run_one(results, "ctx16000", lambda: bench_ctx16000(on_ctx), split=True)
    if want("new_methods"):
        def on_new(key, row):
            results["models"][key] = row
            save(results)

        run_one(
            results,
            "new_methods",
            lambda: bench_new_method_latency(on_new),
            split=True,
        )
    if want("mistral"):
        results["models"].pop("mistral", None)

        def on_row(key, row):
            results["models"][key] = row
            save(results)

        run_one(results, "mistral", lambda: bench_mistral(on_row), split=True)
    print("\nBenchmark finished.", flush=True)


if __name__ == "__main__":
    main()
