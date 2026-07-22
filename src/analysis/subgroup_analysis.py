#!/usr/bin/env python3
"""
Paired subgroup analysis for paper comparisons.

Stratifies held-out predictions by trajectory features (rare codes, sequence
length) and reports AUROC / AUPR with **positive prevalence (%)** in every row.

Designed for **fair paired comparisons** at matched adaptation levels:

  | Setting | Text path | Code path |
  |---|---|---|
  | Full adaptation | ``classifier`` (EHR CPT + LoRA + head) | ``encoder`` (MLM + encoder+head FT) |
  | Frozen backbone | ``ablation_frozen_pretrained`` (5b) | ``encoder_frozen_head`` |

Examples:
  python -m src.analysis.subgroup_analysis \\
    --config_filepath src/configs/mimic/_resolved_isambard/classifier-mistral-icu_mortality.yaml \\
    --primary_json .../mimic-classifier-mistral-icu_mortality/seed_42/results/test_results.json \\
    --primary_name text_llm \\
    --compare_json .../mimic-encoder-transformer-icu_mortality/results/test_results.json \\
    --compare_name encoder \\
    --out_dir analysis_out/subgroup_icu_full

  # Frozen pair (once encoder_frozen_head finishes):
  python -m src.analysis.subgroup_analysis \\
    --primary_json .../mimic-ablation-frozen-pretrained-mistral-icu_mortality/seed_42/results/test_results.json \\
    --compare_json .../mimic-encoder-frozen-head-icu_mortality/results/test_results.json \\
    --primary_name llm_frozen --compare_name encoder_frozen ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.training.config_utils import load_config
from src.data.dataset_factory import get_dataset


CLINICAL_PREFIXES = (
    "LAB//",
    "DIAGNOSIS//",
    "PROCEDURE//",
    "MEDICATION//",
    "DRG//",
)


def _is_clinical(token: str) -> bool:
    return str(token).startswith(CLINICAL_PREFIXES)


def _is_diagnosis(token: str) -> bool:
    return str(token).startswith("DIAGNOSIS//")


def _safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def _safe_aupr(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(labels, probs))
    except ValueError:
        return float("nan")


def _load_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _tokens_for_example(dataset, example_idx: int) -> List[str]:
    example = dataset.examples[example_idx]
    subject_id = example["subject_id"]
    records = dataset.subject_to_records.get(subject_id)
    if not records:
        return []
    record = records[0]
    tokens = list(record["tokens"])
    timestamps = list(record["timestamps"])
    tokens = dataset._apply_prediction_time_cutoff(
        tokens, timestamps, example.get("prediction_time")
    )
    return [dataset.id_to_token.get(tid, "") for tid in tokens]


def _count_train_frequencies(dataset) -> Counter:
    counts: Counter = Counter()
    n = len(dataset)
    print(f"Counting clinical token frequencies over {n:,} train examples...")
    for i in range(n):
        for tok in _tokens_for_example(dataset, i):
            if _is_clinical(tok):
                counts[tok] += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,}/{n:,}")
    print(f"  unique clinical tokens: {len(counts):,}")
    return counts


@dataclass
class ExampleFeatures:
    n_tokens: int
    n_clinical: int
    min_freq: float
    mean_freq: float
    min_diag_freq: float
    has_rare_q10: bool
    has_rare_diag_q10: bool


def _example_features(tokens: Sequence[str], freq: Counter, q10: float) -> ExampleFeatures:
    clinical = [t for t in tokens if _is_clinical(t)]
    diags = [t for t in tokens if _is_diagnosis(t)]
    if not clinical:
        return ExampleFeatures(
            n_tokens=len(tokens),
            n_clinical=0,
            min_freq=float("inf"),
            mean_freq=float("nan"),
            min_diag_freq=float("inf"),
            has_rare_q10=False,
            has_rare_diag_q10=False,
        )
    freqs = [freq.get(t, 0) for t in clinical]
    diag_freqs = [freq.get(t, 0) for t in diags] if diags else []
    min_f = float(min(freqs))
    min_df = float(min(diag_freqs)) if diag_freqs else float("inf")
    return ExampleFeatures(
        n_tokens=len(tokens),
        n_clinical=len(clinical),
        min_freq=min_f,
        mean_freq=float(np.mean(freqs)),
        min_diag_freq=min_df,
        has_rare_q10=min_f <= q10,
        has_rare_diag_q10=min_df <= q10 if diags else False,
    )


def _resolve_example_indices(
    results: Dict[str, Any],
    dataset,
    cache_dir: Optional[str],
    split: str,
    reference_indices: Optional[List[int]] = None,
) -> List[int]:
    n = len(results["labels"])
    if results.get("example_indices") and len(results["example_indices"]) == n:
        return [int(i) for i in results["example_indices"]]

    if reference_indices is not None and len(reference_indices) == n:
        print("Using example_indices from primary / reference run.")
        return reference_indices

    if cache_dir:
        for name in (f"MIMIC_{split}_sorted.pt", f"MIMIC_{split}_sorted_indices.pt"):
            path = os.path.join(cache_dir, name)
            if os.path.exists(path):
                import torch

                indices = [int(i) for i in torch.load(path, map_location="cpu")]
                if len(indices) == n:
                    print(f"Using length-sort cache: {path}")
                    return indices
                print(f"Cache length mismatch ({len(indices)} vs {n}): {path}")

    patient_ids = results.get("patient_ids")
    if not patient_ids or len(patient_ids) != n:
        raise ValueError(
            "Cannot align predictions: need example_indices, reference_indices, "
            "length-sort cache, or patient_ids."
        )
    print("WARNING: aligning via greedy subject_id match.")
    buckets: Dict[str, List[int]] = {}
    for i, ex in enumerate(dataset.examples):
        buckets.setdefault(str(ex["subject_id"]), []).append(i)
    out: List[int] = []
    for pid in patient_ids:
        key = str(pid)
        if not buckets.get(key):
            raise ValueError(f"No example for subject_id={key}")
        out.append(buckets[key].pop(0))
    return out


def _stratify_report(
    name: str,
    labels: np.ndarray,
    probs: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    lab = labels[mask]
    pr = probs[mask]
    n = int(mask.sum())
    n_pos = int(lab.sum()) if len(lab) else 0
    prev = float(lab.mean()) if len(lab) else float("nan")
    return {
        "stratum": name,
        "n": n,
        "n_pos": n_pos,
        "prevalence": prev,
        "prevalence_pct": round(prev * 100.0, 2) if np.isfinite(prev) else float("nan"),
        "auroc": _safe_auroc(lab, pr),
        "aupr": _safe_aupr(lab, pr),
    }


def _tertile_edges(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.quantile(finite, [1 / 3, 2 / 3]))


@dataclass
class StratumDef:
    name: str
    description: str
    mask_fn: Callable[[List[ExampleFeatures], Dict[str, Any]], np.ndarray]


def _build_strata(
    features: List[ExampleFeatures],
    minfreq_t1: float,
    minfreq_t2: float,
    ntok_t1: float,
    ntok_t2: float,
    nclin_t1: float,
    nclin_t2: float,
) -> List[StratumDef]:
    n = len(features)

    def _arr(attr: str) -> np.ndarray:
        return np.array([getattr(f, attr) for f in features])

    min_freqs = _arr("min_freq")
    n_tokens = _arr("n_tokens").astype(float)
    n_clinical = _arr("n_clinical").astype(float)
    has_rare = _arr("has_rare_q10").astype(bool)
    has_rare_diag = _arr("has_rare_diag_q10").astype(bool)

    return [
        StratumDef("all", "All examples", lambda _f, _c: np.ones(n, dtype=bool)),
        StratumDef(
            "has_rare_q10",
            "Trajectory contains a clinical token with train freq <= q10",
            lambda _f, _c: has_rare,
        ),
        StratumDef(
            "no_rare_q10",
            "No rare clinical token (freq > q10)",
            lambda _f, _c: (~has_rare) & (n_clinical > 0),
        ),
        StratumDef(
            "rare_diag_q10",
            "Contains a rare diagnosis code (freq <= q10)",
            lambda _f, _c: has_rare_diag,
        ),
        StratumDef(
            "minfreq_lo",
            f"Min clinical token freq <= {minfreq_t1:.1f}",
            lambda _f, _c: np.isfinite(min_freqs) & (min_freqs <= minfreq_t1),
        ),
        StratumDef(
            "minfreq_mid",
            f"Min freq in ({minfreq_t1:.1f}, {minfreq_t2:.1f}]",
            lambda _f, _c: np.isfinite(min_freqs)
            & (min_freqs > minfreq_t1)
            & (min_freqs <= minfreq_t2),
        ),
        StratumDef(
            "minfreq_hi",
            f"Min freq > {minfreq_t2:.1f}",
            lambda _f, _c: np.isfinite(min_freqs) & (min_freqs > minfreq_t2),
        ),
        StratumDef(
            "ntokens_lo",
            f"Total tokens <= {ntok_t1:.0f} (tertile)",
            lambda _f, _c: n_tokens <= ntok_t1,
        ),
        StratumDef(
            "ntokens_mid",
            f"Total tokens in ({ntok_t1:.0f}, {ntok_t2:.0f}]",
            lambda _f, _c: (n_tokens > ntok_t1) & (n_tokens <= ntok_t2),
        ),
        StratumDef(
            "ntokens_hi",
            f"Total tokens > {ntok_t2:.0f}",
            lambda _f, _c: n_tokens > ntok_t2,
        ),
        StratumDef(
            "nclinical_lo",
            f"Clinical event count <= {nclin_t1:.0f} (tertile)",
            lambda _f, _c: n_clinical <= nclin_t1,
        ),
        StratumDef(
            "nclinical_mid",
            f"Clinical events in ({nclin_t1:.0f}, {nclin_t2:.0f}]",
            lambda _f, _c: (n_clinical > nclin_t1) & (n_clinical <= nclin_t2),
        ),
        StratumDef(
            "nclinical_hi",
            f"Clinical events > {nclin_t2:.0f}",
            lambda _f, _c: n_clinical > nclin_t2,
        ),
    ]


def _print_model_table(name: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n=== {name} ===")
    hdr = ("stratum", "n", "pos%", "auroc", "aupr")
    print(" | ".join(f"{h:>12}" for h in hdr))
    print("-" * 72)
    for r in rows:
        print(
            " | ".join(
                [
                    f"{r['stratum']:>12}",
                    f"{r['n']:>12}",
                    f"{r['prevalence_pct']:>12.1f}",
                    f"{r['auroc']:>12.3f}",
                    f"{r['aupr']:>12.3f}",
                ]
            )
        )


def _print_delta_table(
    primary_name: str,
    compare_name: str,
    primary_rows: List[Dict[str, Any]],
    compare_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print(f"\n=== Δ AUROC ({primary_name} − {compare_name}) ===")
    hdr = ("stratum", "n", "pos%", f"{primary_name}", compare_name, "ΔAUROC", "ΔAUPR")
    print(" | ".join(f"{h:>10}" for h in hdr))
    print("-" * 90)
    deltas = []
    for pr, cr in zip(primary_rows, compare_rows):
        assert pr["stratum"] == cr["stratum"]
        d_auroc = pr["auroc"] - cr["auroc"]
        d_aupr = pr["aupr"] - cr["aupr"]
        deltas.append(
            {
                "stratum": pr["stratum"],
                "n": pr["n"],
                "prevalence_pct": pr["prevalence_pct"],
                f"{primary_name}_auroc": pr["auroc"],
                f"{compare_name}_auroc": cr["auroc"],
                "delta_auroc": d_auroc,
                "delta_aupr": d_aupr,
                f"{primary_name}_aupr": pr["aupr"],
                f"{compare_name}_aupr": cr["aupr"],
            }
        )
        print(
            " | ".join(
                [
                    f"{pr['stratum']:>10}",
                    f"{pr['n']:>10}",
                    f"{pr['prevalence_pct']:>10.1f}",
                    f"{pr['auroc']:>10.3f}",
                    f"{cr['auroc']:>10.3f}",
                    f"{d_auroc:>+10.3f}",
                    f"{d_aupr:>+10.3f}",
                ]
            )
        )
    return deltas


def _evaluate_model(
    name: str,
    results_path: str,
    labels_ref: np.ndarray,
    strata: List[StratumDef],
    ctx: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    res = _load_results(results_path)
    labels = np.asarray(res["labels"], dtype=np.int64)
    probs = np.asarray(res["probs"], dtype=np.float64)
    if len(labels) != len(labels_ref):
        raise ValueError(f"{name}: length {len(labels)} != reference {len(labels_ref)}")
    if not np.array_equal(labels, labels_ref):
        print(f"WARNING: {name} labels differ from reference (using its own labels).")
    rows = []
    for s in strata:
        mask = s.mask_fn(ctx["features"], ctx)
        rows.append(_stratify_report(s.name, labels, probs, mask))
    return rows, labels, probs


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(os.path.abspath(args.config_filepath))
    data_cfg = dict(cfg["data"])
    data_cfg["surface_form"] = "text"

    print("Loading train split for frequency counts...")
    train_ds = get_dataset(data_cfg.get("dataset", "MIMIC"), data_cfg, "train")
    freq = _count_train_frequencies(train_ds)

    all_freqs = np.array(list(freq.values()), dtype=np.float64)
    q10 = float(np.quantile(all_freqs, 0.10)) if len(all_freqs) else 0.0
    print(f"Train clinical freq quantiles: q10={q10:.1f}")

    print(f"Loading eval split '{args.split}'...")
    eval_ds = get_dataset(data_cfg.get("dataset", "MIMIC"), data_cfg, args.split)
    cache_dir = args.cache_dir or (cfg.get("training") or {}).get("data_cache_dir")

    primary_res = _load_results(args.primary_json)
    labels_ref = np.asarray(primary_res["labels"], dtype=np.int64)
    example_indices = _resolve_example_indices(
        primary_res, eval_ds, cache_dir, args.split
    )

    print("Computing per-example features (rarity + length)...")
    features: List[ExampleFeatures] = []
    for i, ex_i in enumerate(example_indices):
        toks = _tokens_for_example(eval_ds, ex_i)
        features.append(_example_features(toks, freq, q10))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,}/{len(example_indices):,}")

    min_freqs = np.array(
        [f.min_freq if np.isfinite(f.min_freq) else np.nan for f in features]
    )
    n_tokens = np.array([f.n_tokens for f in features], dtype=float)
    n_clinical = np.array([f.n_clinical for f in features], dtype=float)

    minfreq_t1, minfreq_t2 = _tertile_edges(min_freqs)
    ntok_t1, ntok_t2 = _tertile_edges(n_tokens)
    nclin_t1, nclin_t2 = _tertile_edges(n_clinical)

    ctx: Dict[str, Any] = {
        "features": features,
        "freq_q10": q10,
        "minfreq_tertile_edges": [minfreq_t1, minfreq_t2],
        "ntokens_tertile_edges": [ntok_t1, ntok_t2],
        "nclinical_tertile_edges": [nclin_t1, nclin_t2],
    }

    strata = _build_strata(
        features, minfreq_t1, minfreq_t2, ntok_t1, ntok_t2, nclin_t1, nclin_t2
    )

    primary_rows, _, _ = _evaluate_model(
        args.primary_name, args.primary_json, labels_ref, strata, ctx
    )
    _print_model_table(args.primary_name, primary_rows)

    payload: Dict[str, Any] = {
        "comparison_pair": args.comparison_pair,
        "split": args.split,
        "primary_name": args.primary_name,
        "primary_json": os.path.abspath(args.primary_json),
        "freq_q10": q10,
        "tertile_edges": {
            "minfreq": [minfreq_t1, minfreq_t2],
            "n_tokens": [ntok_t1, ntok_t2],
            "n_clinical": [nclin_t1, nclin_t2],
        },
        "stratum_definitions": [
            {"name": s.name, "description": s.description} for s in strata
        ],
        "primary_strata": primary_rows,
    }

    if args.compare_json:
        # Reuse primary example_indices for compare if it lacks them
        compare_res = _load_results(args.compare_json)
        if not compare_res.get("example_indices"):
            compare_res["example_indices"] = example_indices
        _resolve_example_indices(
            compare_res, eval_ds, cache_dir, args.split, reference_indices=example_indices
        )
        compare_rows, _, _ = _evaluate_model(
            args.compare_name, args.compare_json, labels_ref, strata, ctx
        )
        _print_model_table(args.compare_name, compare_rows)
        deltas = _print_delta_table(
            args.primary_name, args.compare_name, primary_rows, compare_rows
        )
        payload["compare_name"] = args.compare_name
        payload["compare_json"] = os.path.abspath(args.compare_json)
        payload["compare_strata"] = compare_rows
        payload["delta"] = deltas

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "subgroup_analysis.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")

    if args.save_per_example:
        csv_path = os.path.join(args.out_dir, "per_example_features.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "row",
                    "example_index",
                    "label",
                    "n_tokens",
                    "n_clinical",
                    "min_freq",
                    "has_rare_q10",
                ],
            )
            w.writeheader()
            for i, (ex_i, feat) in enumerate(zip(example_indices, features)):
                w.writerow(
                    {
                        "row": i,
                        "example_index": ex_i,
                        "label": int(labels_ref[i]),
                        "n_tokens": feat.n_tokens,
                        "n_clinical": feat.n_clinical,
                        "min_freq": feat.min_freq,
                        "has_rare_q10": int(feat.has_rare_q10),
                    }
                )
        print(f"Wrote {csv_path}")

    return payload


def main():
    p = argparse.ArgumentParser(description="Paired subgroup analysis (rare code + length)")
    p.add_argument("--config_filepath", required=True)
    p.add_argument("--primary_json", required=True)
    p.add_argument("--primary_name", default="text_llm")
    p.add_argument("--compare_json", default=None)
    p.add_argument("--compare_name", default="encoder")
    p.add_argument(
        "--comparison_pair",
        default="custom",
        help="Label for this comparison (e.g. full_adaptation, frozen_backbone)",
    )
    p.add_argument("--split", default="held_out")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save_per_example", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
