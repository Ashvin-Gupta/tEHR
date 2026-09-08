#!/usr/bin/env python3
"""
Paired significance tests: tEHR vs a MIMIC baseline on the same test set.

H0: Δ = metric(tEHR) − metric(baseline) = 0
H1 (one-sided): Δ > 0  (tEHR is better)

Default baseline is the transformer encoder. Pass --baseline-experiment
to compare against another run (e.g. clmbr_ft).

AUROC uses DeLong (two-sided) plus a paired stratified bootstrap CI / p.
AUPR uses the paired bootstrap only (no closed-form analogue of DeLong).

Example:
  python -m src.analysis.paired_compare
  python -m src.analysis.paired_compare --baseline-experiment clmbr_ft
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.bootstrap_ci import (
    DEFAULT_LOCAL,
    _results_root_from_local,
    canonicalize_experiment,
    extract_labels_probs,
    find_test_prediction,
    load_prediction_json,
    parse_run_dirname,
)
from src.evaluations.evaluations import compute_binary_metrics
from src.experiment_map import MIMIC_TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CI_CSV = REPO_ROOT / "analysis_out" / "bootstrap" / "bootstrap_ci.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis_out" / "paired_compare"

TASK_LABELS = {
    "icu_mortality": "ICU mortality",
    "icu_readmission": "ICU readmission",
    "hospital_mortality": "Hospital mortality",
    "hospital_readmission": "Hospital readmission",
}


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_ci_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with csv_path.open(newline="") as f:
        all_rows = list(csv.DictReader(f))
    preferred = [row for row in all_rows if _is_true(row.get("preferred", ""))]
    return all_rows, preferred


def _model_matches(row_model: str, model: str | None) -> bool:
    row_model = (row_model or "").strip()
    if model is None:
        return not row_model
    return row_model == model


def _discover_runs(
    results_root: Path,
    *,
    experiment: str,
    task: str,
    model: str | None,
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    if not results_root.is_dir():
        return matches
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        meta = parse_run_dirname(child.name)
        if meta is None:
            continue
        if (meta.get("dataset") or "").strip().lower() != "mimic":
            continue
        if canonicalize_experiment(meta["experiment_dir"]) != experiment:
            continue
        if (meta.get("task") or "").strip() != task:
            continue
        if not _model_matches(meta.get("model") or "", model):
            continue
        pred = find_test_prediction(child)
        if pred is None:
            continue
        matches.append(
            {
                "experiment": experiment,
                "task": task,
                "model": (meta.get("model") or "").strip(),
                "dirname": child.name,
                "pred_file": str(pred),
                "dataset": "mimic",
                "variant": meta.get("variant") or "",
                "preferred": "True",
            }
        )
    return matches


def pick_run(
    rows: list[dict[str, str]],
    *,
    experiment: str,
    task: str,
    model: str | None,
    results_root: Path | None = None,
    ci_csv: Path | None = None,
) -> dict[str, str]:
    matches = []
    for row in rows:
        if (row.get("experiment") or "").strip() != experiment:
            continue
        if (row.get("task") or "").strip() != task:
            continue
        if (row.get("dataset") or "").strip().lower() != "mimic":
            continue
        if not _model_matches(row.get("model") or "", model):
            continue
        matches.append(row)
    if not matches and results_root is not None:
        matches = _discover_runs(
            results_root, experiment=experiment, task=task, model=model
        )
        if matches:
            print(
                f"  {experiment}/{task}: not in CI CSV, using {matches[0]['dirname']}",
                flush=True,
            )
    if not matches:
        want = f"{experiment}/{model or '(no model)'}/{task}"
        where = str(ci_csv or DEFAULT_CI_CSV)
        raise SystemExit(f"No preferred run for {want} in {where}")
    matches.sort(
        key=lambda r: (
            1 if (r.get("variant") or "") == "rerun" else 0,
            r.get("dirname") or "",
        )
    )
    return matches[0]


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = x.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def delong_paired(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict[str, float]:
    """Two-sided DeLong test for AUROC(p_a) vs AUROC(p_b) on shared labels."""
    y = np.asarray(y).astype(int)
    preds = np.vstack([np.asarray(p_a, dtype=np.float64), np.asarray(p_b, dtype=np.float64)])
    order = np.argsort(-y, kind="mergesort")
    preds = preds[:, order]
    m = int(y.sum())
    n = int(y.size - m)
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes")
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01, ddof=1)
    sy = np.cov(v10, ddof=1)
    cov = sx / m + sy / n
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ cov @ contrast)
    delta = float(aucs[0] - aucs[1])
    if var <= 0 or not np.isfinite(var):
        z = np.inf if delta != 0 else 0.0
    else:
        z = delta / np.sqrt(var)
    p_two = float(2.0 * norm.sf(abs(z)))
    p_one = float(norm.sf(z))
    return {
        "auroc_a": float(aucs[0]),
        "auroc_b": float(aucs[1]),
        "delta": delta,
        "se": float(np.sqrt(max(var, 0.0))),
        "z": float(z),
        "p_two": p_two,
        "p_one": p_one,
    }


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvals[idx] * (m - rank)))
        adj[idx] = running
    return adj


def _extract_with_order(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    y, p, clusters = extract_labels_probs(payload)
    idx = payload.get("example_indices")
    if idx is None:
        return y, p, clusters, False
    order = np.argsort(np.asarray(idx), kind="mergesort")
    if clusters is not None:
        clusters = clusters[order]
    return y[order], p[order], clusters, True


def borrow_example_indices(
    payload: dict[str, Any],
    *,
    task: str,
    rows: list[dict[str, str]],
    self_path: str,
) -> dict[str, Any]:
    """Older tEHR dumps omit example_indices; copy them from a run with the same order."""
    if payload.get("example_indices") is not None:
        return payload
    pids = payload.get("patient_ids")
    labels = payload.get("labels")
    if pids is None or labels is None:
        return payload
    skip_exp = {
        "encoder",
        "encoder_frozen",
        "encoder_supervised",
        "xgboost_bow",
        "clinicalbert",
        "text_embedding",
        "decoder",
        "zero_shot",
        "bioclinical_modernbert",
        "clmbr",
        "clmbr_ft",
    }
    for row in rows:
        if (row.get("task") or "").strip() != task:
            continue
        if (row.get("experiment") or "").strip() in skip_exp:
            continue
        path = row.get("pred_file") or ""
        if not path or path == self_path:
            continue
        donor, _ = load_prediction_json(Path(path))
        if donor.get("example_indices") is None:
            continue
        if donor.get("patient_ids") == pids and donor.get("labels") == labels:
            print(f"  borrowed example_indices from {row.get('dirname')}", flush=True)
            out = dict(payload)
            out["example_indices"] = donor["example_indices"]
            return out
    return payload


def align_predictions(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, str]:
    y_a, p_a, clusters_a, sorted_a = _extract_with_order(payload_a)
    y_b, p_b, _clusters_b, sorted_b = _extract_with_order(payload_b)
    if y_a.size == y_b.size and np.array_equal(y_a, y_b):
        how = "positional (labels identical)"
        if sorted_a or sorted_b:
            bits = []
            if sorted_a:
                bits.append("tEHR by example_indices")
            if sorted_b:
                bits.append("baseline by example_indices")
            how = "; ".join(bits)
        return y_a, p_a, p_b, clusters_a, how
    raise SystemExit(
        f"Cannot align predictions: n={y_a.size}/{y_b.size} "
        f"label_mismatch={int(y_a.size == y_b.size and not np.array_equal(y_a, y_b))}"
    )


def paired_bootstrap(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    alpha = (1.0 - ci) / 2.0
    d_auroc = np.empty(n_boot, dtype=np.float64)
    d_aupr = np.empty(n_boot, dtype=np.float64)
    kept = 0
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True),
                rng.choice(neg, size=neg.size, replace=True),
            ]
        )
        y_b, a_b, b_b = y[idx], p_a[idx], p_b[idx]
        if np.unique(y_b).size < 2:
            continue
        try:
            d_auroc[kept] = roc_auc_score(y_b, a_b) - roc_auc_score(y_b, b_b)
            d_aupr[kept] = average_precision_score(y_b, a_b) - average_precision_score(y_b, b_b)
        except ValueError:
            continue
        kept += 1
    if kept == 0:
        raise ValueError("every bootstrap replicate failed")
    d_auroc = d_auroc[:kept]
    d_aupr = d_aupr[:kept]

    def summarise(obs: float, draws: np.ndarray) -> dict[str, float]:
        lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
        p_one = (1.0 + float(np.sum(draws <= 0.0))) / (kept + 1)
        p_two = min(1.0, 2.0 * p_one)
        return {
            "obs": float(obs),
            "boot_mean": float(np.mean(draws)),
            "lo": float(lo),
            "hi": float(hi),
            "p_one": float(p_one),
            "p_two": float(p_two),
            "ci_excludes_0": bool(lo > 0.0 or hi < 0.0),
        }

    point_a = compute_binary_metrics(y, p_a)
    point_b = compute_binary_metrics(y, p_b)
    return {
        "n_boot": int(kept),
        "auroc": summarise(point_a["auroc"] - point_b["auroc"], d_auroc),
        "aupr": summarise(point_a["aupr"] - point_b["aupr"], d_aupr),
        "point_a": point_a,
        "point_b": point_b,
    }


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-csv", type=Path, default=DEFAULT_CI_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--local-yaml", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--champion-experiment", default="tehr_stage2")
    parser.add_argument("--champion-model", default="mistral")
    parser.add_argument("--baseline-experiment", default="encoder")
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=list(MIMIC_TASKS))
    args = parser.parse_args(argv)

    results_root = args.results_root or _results_root_from_local(args.local_yaml)
    all_rows, preferred = load_ci_rows(args.ci_csv)
    baseline_model = args.baseline_model or None
    results: list[dict[str, Any]] = []
    stem = f"tehr_vs_{args.baseline_experiment}"

    for task in args.tasks:
        champ = pick_run(
            preferred,
            experiment=args.champion_experiment,
            task=task,
            model=args.champion_model,
            results_root=results_root,
            ci_csv=args.ci_csv,
        )
        base = pick_run(
            preferred,
            experiment=args.baseline_experiment,
            task=task,
            model=baseline_model,
            results_root=results_root,
            ci_csv=args.ci_csv,
        )
        print(f"\n=== {task} ===", flush=True)
        print(f"  tEHR     {champ['dirname']}", flush=True)
        print(f"  {args.baseline_experiment:<8} {base['dirname']}", flush=True)

        payload_a, _ = load_prediction_json(Path(champ["pred_file"]))
        payload_b, _ = load_prediction_json(Path(base["pred_file"]))
        payload_a = borrow_example_indices(
            payload_a, task=task, rows=all_rows, self_path=champ["pred_file"]
        )
        payload_b = borrow_example_indices(
            payload_b, task=task, rows=all_rows, self_path=base["pred_file"]
        )
        y, p_a, p_b, clusters, how = align_predictions(payload_a, payload_b)
        print(f"  aligned {how}; n={y.size} n_pos={int((y == 1).sum())}", flush=True)

        delong = delong_paired(y, p_a, p_b)
        print(
            f"  DeLong AUROC  Δ={delong['delta']:+.4f}  z={delong['z']:.2f}  "
            f"p_two={_fmt_p(delong['p_two'])}  p_one={_fmt_p(delong['p_one'])}",
            flush=True,
        )
        print(f"  bootstrap n_boot={args.n_boot} …", flush=True)
        boot = paired_bootstrap(
            y, p_a, p_b, n_boot=args.n_boot, ci=args.ci, seed=args.seed
        )
        for metric in ("auroc", "aupr"):
            s = boot[metric]
            print(
                f"  {metric.upper():5s} tEHR={boot['point_a'][metric]:.4f}  "
                f"base={boot['point_b'][metric]:.4f}  Δ={s['obs']:+.4f}  "
                f"CI [{s['lo']:+.4f}, {s['hi']:+.4f}]  "
                f"p_one={_fmt_p(s['p_one'])}  p_two={_fmt_p(s['p_two'])}",
                flush=True,
            )
        results.append(
            {
                "task": task,
                "task_label": TASK_LABELS.get(task, task),
                "n": int(y.size),
                "n_pos": int((y == 1).sum()),
                "align": how,
                "champion": {
                    "experiment": args.champion_experiment,
                    "model": args.champion_model,
                    "dirname": champ["dirname"],
                    "pred_file": champ["pred_file"],
                    "auroc": boot["point_a"]["auroc"],
                    "aupr": boot["point_a"]["aupr"],
                },
                "baseline": {
                    "experiment": args.baseline_experiment,
                    "model": args.baseline_model,
                    "dirname": base["dirname"],
                    "pred_file": base["pred_file"],
                    "auroc": boot["point_b"]["auroc"],
                    "aupr": boot["point_b"]["aupr"],
                },
                "delong": delong,
                "bootstrap": {
                    "n_boot": boot["n_boot"],
                    "ci": args.ci,
                    "auroc": boot["auroc"],
                    "aupr": boot["aupr"],
                },
            }
        )

    p_delong = [r["delong"]["p_two"] for r in results]
    p_auroc_one = [r["bootstrap"]["auroc"]["p_one"] for r in results]
    p_aupr_one = [r["bootstrap"]["aupr"]["p_one"] for r in results]
    holm_delong = holm(p_delong)
    holm_auroc = holm(p_auroc_one)
    holm_aupr = holm(p_aupr_one)
    for i, row in enumerate(results):
        row["delong"]["p_two_holm"] = holm_delong[i]
        row["bootstrap"]["auroc"]["p_one_holm"] = holm_auroc[i]
        row["bootstrap"]["aupr"]["p_one_holm"] = holm_aupr[i]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")

    csv_path = args.out_dir / f"{stem}.csv"
    flat_fields = [
        "task",
        "n",
        "n_pos",
        "tehr_auroc",
        "baseline_auroc",
        "delta_auroc",
        "auroc_lo",
        "auroc_hi",
        "delong_z",
        "delong_p_two",
        "delong_p_two_holm",
        "auroc_p_one",
        "auroc_p_one_holm",
        "tehr_aupr",
        "baseline_aupr",
        "delta_aupr",
        "aupr_lo",
        "aupr_hi",
        "aupr_p_one",
        "aupr_p_one_holm",
        "tehr_dirname",
        "baseline_dirname",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "task": row["task"],
                    "n": row["n"],
                    "n_pos": row["n_pos"],
                    "tehr_auroc": row["champion"]["auroc"],
                    "baseline_auroc": row["baseline"]["auroc"],
                    "delta_auroc": row["bootstrap"]["auroc"]["obs"],
                    "auroc_lo": row["bootstrap"]["auroc"]["lo"],
                    "auroc_hi": row["bootstrap"]["auroc"]["hi"],
                    "delong_z": row["delong"]["z"],
                    "delong_p_two": row["delong"]["p_two"],
                    "delong_p_two_holm": row["delong"]["p_two_holm"],
                    "auroc_p_one": row["bootstrap"]["auroc"]["p_one"],
                    "auroc_p_one_holm": row["bootstrap"]["auroc"]["p_one_holm"],
                    "tehr_aupr": row["champion"]["aupr"],
                    "baseline_aupr": row["baseline"]["aupr"],
                    "delta_aupr": row["bootstrap"]["aupr"]["obs"],
                    "aupr_lo": row["bootstrap"]["aupr"]["lo"],
                    "aupr_hi": row["bootstrap"]["aupr"]["hi"],
                    "aupr_p_one": row["bootstrap"]["aupr"]["p_one"],
                    "aupr_p_one_holm": row["bootstrap"]["aupr"]["p_one_holm"],
                    "tehr_dirname": row["champion"]["dirname"],
                    "baseline_dirname": row["baseline"]["dirname"],
                }
            )

    print(f"\nWrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print("\nHolm-adjusted DeLong p (AUROC, 4 tasks):", flush=True)
    for row in results:
        sig = "significant" if row["delong"]["p_two_holm"] < 0.05 else "n.s."
        print(
            f"  {row['task_label']:<22} p={_fmt_p(row['delong']['p_two'])}  "
            f"Holm={_fmt_p(row['delong']['p_two_holm'])}  {sig}",
            flush=True,
        )


if __name__ == "__main__":
    main()
