#!/usr/bin/env python3
"""
Mean and Student-t 95% CIs across random-seed runs.

Looks under each experiment directory for ``seed_<n>/`` folders (the layout
``prepare_run_reproducibility`` already writes). Each seed's test
``labels``/``probs`` are scored, then AUROC / AUPRC / … are aggregated.

  SEED=43 sbatch scripts/site_b/run.sh mimic tehr_stage2 mistral icu_mortality
  python -m src.analysis.seed_ci
  sbatch scripts/site_b/run_cpu.sh seed_ci

This is training-luck uncertainty (several checkpoints). Test-set bootstrap
of a single checkpoint is ``python -m src.analysis.bootstrap_ci``.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.analysis.bootstrap_ci import (
    DEFAULT_LOCAL,
    DEFAULT_OUT_DIR,
    _results_root_from_local,
    _write_csv,
    canonicalize_experiment,
    extract_labels_probs,
    find_test_prediction,
    iter_run_dirs,
    load_prediction_json,
    parse_run_dirname,
)
from src.evaluations.evaluations import compute_binary_metrics, mean_t_ci

SEED_DIR_RE = re.compile(r"^seed_(\d+)$")
METRIC_KEYS = ("auroc", "aupr", "accuracy", "precision", "recall", "f1", "ece")


def iter_seed_predictions(run_dir: Path) -> list[tuple[int | None, Path]]:
    """``(seed, pred_json)`` for every ``seed_*`` subdir, else one unseeded file."""
    seeded: list[tuple[int | None, Path]] = []
    if not run_dir.is_dir():
        return seeded
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        match = SEED_DIR_RE.match(child.name)
        if match is None:
            continue
        pred = find_test_prediction(child)
        if pred is not None:
            seeded.append((int(match.group(1)), pred))
    if seeded:
        return seeded
    pred = find_test_prediction(run_dir)
    if pred is None:
        return []
    return [(None, pred)]


def score_pred_file(path: Path) -> dict[str, Any]:
    payload, recovered = load_prediction_json(path)
    y, p, _clusters = extract_labels_probs(payload)
    metrics = compute_binary_metrics(y, p)
    return {
        **metrics,
        "n": int(y.size),
        "n_pos": int((y == 1).sum()),
        "json_recovered": recovered,
    }


def collect_seed_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in iter_run_dirs(results_root):
        meta = parse_run_dirname(run_dir.name)
        assert meta is not None
        experiment = canonicalize_experiment(meta["experiment_dir"])
        found = iter_seed_predictions(run_dir)
        if not found:
            print(f"  SKIP  {run_dir.name}: no test/held_out results JSON")
            continue
        for seed, pred in found:
            try:
                scored = score_pred_file(pred)
            except Exception as exc:
                print(f"  FAIL  {run_dir.name} seed={seed}: {exc}")
                continue
            rows.append(
                {
                    "dirname": run_dir.name,
                    "dataset": meta["dataset"],
                    "legacy_name": meta["experiment_dir"],
                    "experiment": experiment,
                    "model": meta["model"] or "",
                    "task": meta["task"] or "",
                    "variant": meta["variant"] or "",
                    "run_seed": "" if seed is None else seed,
                    "pred_file": str(pred),
                    **scored,
                }
            )
            seed_lab = "unseeded" if seed is None else f"seed_{seed}"
            print(
                f"  OK    {run_dir.name}  {seed_lab}  "
                f"AUROC {scored['auroc']:.3f}  AUPR {scored['aupr']:.3f}"
            )
    return rows


def summarise_groups(
    seed_rows: list[dict[str, Any]],
    *,
    ci: float,
    merge_reruns: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        if merge_reruns:
            key = (
                row["dataset"],
                row["experiment"],
                row["model"],
                row["task"],
            )
        else:
            key = (
                row["dataset"],
                row["experiment"],
                row["model"],
                row["task"],
                row["dirname"],
            )
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        members = sorted(members, key=lambda r: (str(r["run_seed"]), r["dirname"]))
        first = members[0]
        seeds = [m["run_seed"] for m in members]
        summary: dict[str, Any] = {
            "dataset": first["dataset"],
            "experiment": first["experiment"],
            "model": first["model"],
            "task": first["task"],
            "dirnames": ";".join(sorted({m["dirname"] for m in members})),
            "n_seeds": len(members),
            "seeds": ",".join("na" if s == "" else str(s) for s in seeds),
            "ci": ci,
        }
        for metric in METRIC_KEYS:
            vals = [
                m[metric]
                for m in members
                if m.get(metric) not in ("", None)
            ]
            if not vals:
                summary[f"{metric}_mean"] = ""
                summary[f"{metric}_sd"] = ""
                summary[f"{metric}_min"] = ""
                summary[f"{metric}_max"] = ""
                summary[f"{metric}_lo"] = ""
                summary[f"{metric}_hi"] = ""
                continue
            stats = mean_t_ci(vals, ci=ci)
            summary[f"{metric}_mean"] = stats["mean"]
            summary[f"{metric}_sd"] = stats["sd"]
            summary[f"{metric}_min"] = stats["min"]
            summary[f"{metric}_max"] = stats["max"]
            summary[f"{metric}_lo"] = stats["lo"]
            summary[f"{metric}_hi"] = stats["hi"]
        summaries.append(summary)
        note = "" if len(members) >= 2 else "  (n=1, no t-interval)"
        print(
            f"  SUM   {first['experiment']} {first['model']} {first['task']}  "
            f"n={len(members)}  AUROC {summary['auroc_mean']:.3f}"
            + (
                f" [{summary['auroc_lo']:.3f}, {summary['auroc_hi']:.3f}]"
                if summary["auroc_lo"] is not None
                else ""
            )
            + note
        )
    return summaries


def _fmt3(value: Any) -> str:
    if value is None or value == "":
        return "---"
    return f"{float(value):.3f}"


def write_latex(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "% Mean across seeds; brackets = Student-t 95% CI on the mean.",
        "% n=1 groups have no interval.",
        r"\newcommand{\metric}[3]{\makecell[c]{#1\\[-0.15em]{\footnotesize[#2,\,#3]}}}",
        r"\newcommand{\metricna}[1]{\makecell[c]{#1\\[-0.15em]{\footnotesize$^{\dagger}$}}}",
        "",
    ]
    for row in summaries:
        label = " ".join(
            p for p in (row["experiment"], row["model"], row["task"]) if p
        )
        cells = []
        for metric in ("auroc", "aupr"):
            mean = _fmt3(row[f"{metric}_mean"])
            lo, hi = row[f"{metric}_lo"], row[f"{metric}_hi"]
            if lo is None or hi is None:
                cells.append(rf"\metricna{{{mean}}}")
            else:
                cells.append(rf"\metric{{{mean}}}{{{_fmt3(lo)}}}{{{_fmt3(hi)}}}")
        lines.append(f"{label} % n={row['n_seeds']} seeds={row['seeds']}")
        lines.append("  & " + " & ".join(cells) + r" \\")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--local-yaml", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--merge-reruns",
        action="store_true",
        help="Pool -rerun sibling directories with the same experiment/model/task.",
    )
    args = parser.parse_args(argv)

    results_root = (
        Path(args.results_root) if args.results_root else _results_root_from_local(args.local_yaml)
    )
    print(f"Results root: {results_root}")
    print(f"Across-seed Student-t CI, ci={args.ci}")

    seed_rows = collect_seed_rows(results_root)
    summaries = summarise_groups(
        seed_rows, ci=args.ci, merge_reruns=args.merge_reruns
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_fields = [
        "experiment",
        "model",
        "task",
        "run_seed",
        "variant",
        "dirname",
        "dataset",
        "n",
        "n_pos",
        *METRIC_KEYS,
        "pred_file",
    ]
    sum_fields = [
        "experiment",
        "model",
        "task",
        "dataset",
        "n_seeds",
        "seeds",
        "dirnames",
        "ci",
    ]
    for metric in METRIC_KEYS:
        sum_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_sd",
                f"{metric}_lo",
                f"{metric}_hi",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )

    runs_path = args.out_dir / "seed_runs.csv"
    sum_path = args.out_dir / "seed_summary.csv"
    tex_path = args.out_dir / "seed_summary.tex"
    _write_csv(runs_path, seed_rows, run_fields)
    _write_csv(sum_path, summaries, sum_fields)
    write_latex(tex_path, summaries)

    n_multi = sum(1 for s in summaries if s["n_seeds"] >= 2)
    print(f"\nWrote {len(seed_rows)} seed runs → {runs_path}")
    print(f"Wrote {len(summaries)} groups ({n_multi} with ≥2 seeds) → {sum_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
