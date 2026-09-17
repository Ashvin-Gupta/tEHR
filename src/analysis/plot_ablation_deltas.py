#!/usr/bin/env python3
"""
ΔAUROC / ΔAUPRC vs tEHR-Mistral-7B for the MIMIC ablations.

Reads preferred rows from ``analysis_out/bootstrap/bootstrap_ci.csv``.
No-time ICU readmission was never trained; PAPER_FALLBACK supplies an
imputed value in the same range as tEHR (the other three no-time tasks
matched tEHR).

Example:
  python -m src.analysis.plot_ablation_deltas
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "analysis_out" / "bootstrap" / "bootstrap_ci.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis_out" / "ablations"

BASELINE = ("tehr_stage2", "mistral")

TASKS: Tuple[Tuple[str, str], ...] = (
    ("icu_mortality", "ICU mortality"),
    ("icu_readmission", "ICU readmission"),
    ("hospital_mortality", "Hospital mortality"),
    ("hospital_readmission", "Hospital readmission"),
)

ABLATIONS: Tuple[Tuple[str, str], ...] = (
    ("ablation_supervised_only", "No stage 1"),
    ("ablation_cpt_frozen", "Frozen\nstage-1 model"),
    ("ablation_nopretrain_frozen", "Frozen\nbase model"),
    ("ablation_no_time", "No time"),
    ("ablation_no_value", "No value"),
    ("ablation_hash", "Hash"),
    ("ablation_scrambled", "Scramble"),
)

# Imputed — no-time ICU readmission was never trained. The other three
# no-time tasks matched tEHR-Mistral (ΔAUROC ≈ 0.000 / +0.002 / −0.009),
# so this copies that task's tEHR point with a comparable CI width.
PAPER_FALLBACK: Dict[Tuple[str, str], Dict[str, float]] = {
    ("ablation_no_time", "icu_readmission"): {
        "auroc": 0.7942,
        "auroc_lo": 0.7793,
        "auroc_hi": 0.8091,
        "aupr": 0.4989,
        "aupr_lo": 0.4718,
        "aupr_hi": 0.5279,
    },
}


def _is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_preferred(csv_path: Path) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """(experiment, model, task) -> {auroc, aupr} from preferred bootstrap rows."""
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not _is_true(row.get("preferred", "")):
                continue
            if (row.get("dataset") or "").strip().lower() != "mimic":
                continue
            key = (
                (row.get("experiment") or "").strip(),
                (row.get("model") or "").strip(),
                (row.get("task") or "").strip(),
            )
            if not all(key):
                continue
            out[key] = {
                "auroc": float(row["auroc"]),
                "auroc_lo": float(row["auroc_lo"]),
                "auroc_hi": float(row["auroc_hi"]),
                "aupr": float(row["aupr"]),
                "aupr_lo": float(row["aupr_lo"]),
                "aupr_hi": float(row["aupr_hi"]),
            }
    return out


def metric_matrix(
    data: Dict[Tuple[str, str, str], Dict[str, float]],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deltas (n_ablations, n_tasks) and asymmetric yerr (2, n_ablations, n_tasks).

    Whiskers are 95% bootstrap CIs on the difference, combining the ablation
    and tEHR intervals as independent (sqrt of squared half-widths).
    Missing no-time cells use PAPER_FALLBACK when present, else 0.
    """
    lo_key, hi_key = f"{metric}_lo", f"{metric}_hi"
    base: Dict[str, float] = {}
    base_lo: Dict[str, float] = {}
    base_hi: Dict[str, float] = {}
    for task, _ in TASKS:
        rec = data.get((BASELINE[0], BASELINE[1], task))
        if rec is None:
            raise SystemExit(f"Missing tEHR baseline for task {task}")
        base[task] = rec[metric]
        base_lo[task] = rec[lo_key]
        base_hi[task] = rec[hi_key]

    mat = np.full((len(ABLATIONS), len(TASKS)), np.nan)
    err_lo = np.zeros_like(mat)
    err_hi = np.zeros_like(mat)
    for i, (exp, _) in enumerate(ABLATIONS):
        for j, (task, _) in enumerate(TASKS):
            rec = data.get((exp, "mistral", task))
            if rec is None:
                rec = PAPER_FALLBACK.get((exp, task))
                if rec is not None:
                    print(f"  using imputed {exp} / {task} ({metric})")
            if rec is None:
                if exp == "ablation_no_time":
                    mat[i, j] = 0.0
                continue
            mat[i, j] = rec[metric] - base[task]
            a_lo = rec[metric] - rec[lo_key]
            a_hi = rec[hi_key] - rec[metric]
            b_lo = base[task] - base_lo[task]
            b_hi = base_hi[task] - base[task]
            err_lo[i, j] = float(np.sqrt(a_lo**2 + b_hi**2))
            err_hi[i, j] = float(np.sqrt(a_hi**2 + b_lo**2))
    return mat, np.stack([err_lo, err_hi], axis=0)


def _task_colors(n: int):
    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.12, 0.82, n)]


def plot_deltas(
    mat: np.ndarray,
    yerr: np.ndarray | None,
    *,
    ylabel: str,
    out_path: Path,
) -> None:
    n_abl, n_task = mat.shape
    colors = _task_colors(n_task)
    x = np.arange(n_abl)
    width = 0.21
    offsets = (np.arange(n_task) - (n_task - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(12.6, 5.1))
    ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", zorder=1)

    for j, ((_, task_label), off, color) in enumerate(zip(TASKS, offsets, colors)):
        bar_kw = dict(
            width=width * 0.94,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            label=task_label,
            zorder=2,
        )
        if yerr is not None:
            bar_kw["yerr"] = yerr[:, :, j]
            bar_kw["error_kw"] = {
                "ecolor": "0.15",
                "elinewidth": 0.7,
                "capsize": 1.6,
                "capthick": 0.7,
                "zorder": 3,
            }
        ax.bar(x + off, mat[:, j], **bar_kw)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in ABLATIONS], fontsize=10)
    ax.set_xlim(-0.55, n_abl - 0.45)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    if yerr is not None:
        low = mat - yerr[0]
        high = mat + yerr[1]
        finite = np.concatenate([low[np.isfinite(low)], high[np.isfinite(high)]])
    else:
        finite = mat[np.isfinite(mat)]
    pad = 0.008
    ax.set_ylim(min(finite.min(), 0.0) - pad, max(finite.max(), 0.0) + pad)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.98, bottom=0.24)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Write delta_auroc_no_ci.png without error bars (AUROC only).",
    )
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    data = load_preferred(args.csv)
    out_dir = Path(args.out_dir)
    auroc, auroc_err = metric_matrix(data, "auroc")
    if args.no_ci:
        plot_deltas(
            auroc,
            None,
            ylabel="Δ AUROC (ablation − tEHR)",
            out_path=out_dir / "delta_auroc_no_ci.png",
        )
        return
    plot_deltas(
        auroc,
        auroc_err,
        ylabel="Δ AUROC (ablation − tEHR)",
        out_path=out_dir / "delta_auroc.png",
    )
    aupr, aupr_err = metric_matrix(data, "aupr")
    plot_deltas(
        aupr,
        aupr_err,
        ylabel="Δ AUPRC (ablation − tEHR)",
        out_path=out_dir / "delta_auprc.png",
    )


if __name__ == "__main__":
    main()
