#!/usr/bin/env python3
"""
Bootstrap 95% CIs on saved test-set predictions (no retraining).

This is test-set sampling uncertainty for one checkpoint. Across-seed
mean / Student-t CIs are ``python -m src.analysis.seed_ci``.

Example:

  python -m src.analysis.bootstrap_ci
  sbatch scripts/site_b/run_cpu.sh bootstrap_ci
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from src.evaluations.evaluations import bootstrap_binary_ci
from src.experiment_map import CPRD_TASKS, DATASETS, MIMIC_TASKS, MODELS, SPECS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL = REPO_ROOT / "configs" / "local.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis_out" / "bootstrap"

TEST_FILENAMES = ("test_results.json", "held_out_results.json")
SKIP_TOP_LEVEL = frozenset(
    {
        "mimic-baselines",
        "mimic-encoder",  # MLM-only, no classifier test split
        "mimic-decoder",  # NTP-only, no classifier test split
        "cprd-encoder",
        "cprd-decoder",
        "mimic-text_embedding",  # phrase embeddings, not a scored run
    }
)

# Pre-refactor directory stems → current experiment names in experiment_map.
# Longest-prefix match on the experiment token(s) between dataset and model/task.
LEGACY_EXPERIMENT_ALIASES: dict[str, str] = {
    "classifier": "tehr_stage2",
    "ablation-frozen-nopretrain": "ablation_nopretrain_frozen",
    "ablation-frozen-pretrained": "ablation_cpt_frozen",
    "ablation-no-pretrain-lora": "ablation_supervised_only",
    "surface-scramble-hash": "ablation_hash",
    "surface-scramble-shuffle": "ablation_scrambled",
    "surface-raw": "surface_raw",  # legacy raw-code titles; not a current experiment
    "encoder-transformer": "encoder",
    "encoder-frozen-head": "encoder_frozen",
    "encoder-supervised": "encoder_supervised",
    "text-embedding-transformer": "text_embedding",
    "xgboost-bow": "xgboost_bow",
    "prompt": "zero_shot",
}

_TASKS = tuple(sorted(set(MIMIC_TASKS) | set(CPRD_TASKS), key=len, reverse=True))
_MODELS = tuple(MODELS)
_DATASETS = tuple(DATASETS)


@dataclass
class RunRef:
    dirname: str
    dataset: str
    experiment_dir: str
    experiment: str
    model: str | None
    task: str | None
    variant: str | None
    pred_path: Path | None
    missing_reason: str | None = None


def _results_root_from_local(path: Path) -> Path:
    with open(path) as f:
        local = yaml.safe_load(f) or {}
    root = local.get("results_root")
    if not root:
        raise SystemExit(f"no results_root in {path}")
    return Path(root)


def parse_run_dirname(name: str) -> dict[str, Any] | None:
    """Parse ``{dataset}-{experiment}[-{model}]-{task}[-rerun]`` (old or new)."""
    variant = None
    stem = name
    if stem.endswith("-rerun"):
        variant = "rerun"
        stem = stem[: -len("-rerun")]

    dataset = None
    rest = stem
    for ds in _DATASETS:
        prefix = f"{ds}-"
        if stem.startswith(prefix):
            dataset = ds
            rest = stem[len(prefix) :]
            break
    if dataset is None:
        return None

    task = None
    for t in _TASKS:
        suffix = f"-{t}"
        if rest.endswith(suffix):
            task = t
            rest = rest[: -len(suffix)]
            break

    model = None
    for m in _MODELS:
        suffix = f"-{m}"
        if rest.endswith(suffix):
            model = m
            rest = rest[: -len(suffix)]
            break

    experiment_dir = rest
    if not experiment_dir:
        return None
    return {
        "dataset": dataset,
        "experiment_dir": experiment_dir,
        "model": model,
        "task": task,
        "variant": variant,
    }


def canonicalize_experiment(experiment_dir: str) -> str:
    if experiment_dir in SPECS:
        return experiment_dir
    return LEGACY_EXPERIMENT_ALIASES.get(experiment_dir, experiment_dir)


def load_prediction_json(path: Path) -> tuple[dict[str, Any], bool]:
    """Load a results JSON; if DDP concatenated two objects, take the first."""
    raw = path.read_text()
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        obj, _idx = json.JSONDecoder().raw_decode(raw)
        if not isinstance(obj, dict):
            raise
        return obj, True


def extract_labels_probs(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    labels = payload.get("labels")
    probs = payload.get("probs")
    if labels is None or probs is None:
        raise ValueError("missing labels/probs")
    y = np.asarray(labels)
    p = np.asarray(probs, dtype=np.float64)
    if y.ndim != 1:
        y = y.reshape(-1)
    if p.ndim != 1:
        p = p.reshape(-1)
    if y.size != p.size:
        raise ValueError(f"length mismatch labels={y.size} probs={p.size}")
    y = y.astype(int)
    pids = payload.get("patient_ids")
    clusters = np.asarray([str(x) for x in pids]) if pids is not None else None
    if clusters is not None and clusters.size != y.size:
        clusters = None
    return y, p, clusters


def find_test_prediction(run_dir: Path) -> Path | None:
    """Prefer ``test_results.json`` over ``held_out_results.json``; seed_42 if present."""
    found: list[Path] = []
    for fname in TEST_FILENAMES:
        found.extend(run_dir.rglob(fname))
    if not found:
        return None

    def rank(p: Path) -> tuple[int, int, int]:
        is_test = 0 if p.name == "test_results.json" else 1
        in_seed = 0 if "seed_" in p.parts else 1
        return (is_test, in_seed, -int(p.stat().st_mtime))

    found.sort(key=rank)
    return found[0]


def iter_run_dirs(results_root: Path) -> Iterable[Path]:
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in SKIP_TOP_LEVEL or child.name.startswith("Pretrain-"):
            continue
        parsed = parse_run_dirname(child.name)
        if parsed is None:
            continue
        yield child


def collect_runs(results_root: Path) -> list[RunRef]:
    runs: list[RunRef] = []
    for run_dir in iter_run_dirs(results_root):
        meta = parse_run_dirname(run_dir.name)
        assert meta is not None
        experiment = canonicalize_experiment(meta["experiment_dir"])
        pred = find_test_prediction(run_dir)
        runs.append(
            RunRef(
                dirname=run_dir.name,
                dataset=meta["dataset"],
                experiment_dir=meta["experiment_dir"],
                experiment=experiment,
                model=meta["model"],
                task=meta["task"],
                variant=meta["variant"],
                pred_path=pred,
                missing_reason=None if pred else "no test/held_out results JSON",
            )
        )
    return runs


def mark_preferred(rows: list[dict[str, Any]]) -> None:
    """One preferred row per (experiment, model, task): current name, then non-rerun."""
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        if row.get("status") != "ok":
            continue
        key = (row["experiment"], row.get("model") or "", row.get("task") or "")
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        def score(i: int) -> tuple:
            r = rows[i]
            current = 0 if r["experiment"] in r["dirname"] else 1
            rerun = 1 if r.get("variant") == "rerun" else 0
            return (current, rerun, -float(r.get("n") or 0))

        best = min(idxs, key=score)
        for i in idxs:
            rows[i]["preferred"] = i == best


def bootstrap_runs(
    runs: list[RunRef],
    *,
    n_boot: int,
    ci: float,
    seed: int,
    clustered: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ok_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for run in runs:
        base = {
            "dirname": run.dirname,
            "dataset": run.dataset,
            "legacy_name": run.experiment_dir,
            "experiment": run.experiment,
            "model": run.model or "",
            "task": run.task or "",
            "variant": run.variant or "",
        }
        if run.pred_path is None:
            missing_rows.append({**base, "reason": run.missing_reason or "missing"})
            print(f"  SKIP  {run.dirname}: {run.missing_reason}")
            continue
        rel = run.pred_path
        try:
            payload, recovered = load_prediction_json(run.pred_path)
            y, p, clusters = extract_labels_probs(payload)
            cluster_arg = clusters if clustered and clusters is not None else None
            stats = bootstrap_binary_ci(
                y,
                p,
                n_boot=n_boot,
                ci=ci,
                seed=seed,
                stratified=True,
                clusters=cluster_arg,
            )
        except Exception as exc:
            missing_rows.append(
                {
                    **base,
                    "pred_file": str(rel),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  FAIL  {run.dirname}: {exc}")
            continue
        row = {
            **base,
            "status": "ok",
            "pred_file": str(rel),
            "json_recovered": recovered,
            "n": stats["n"],
            "n_pos": stats["n_pos"],
            "n_boot": stats["n_boot"],
            "clustered": stats["clustered"],
            "auroc": stats["auroc"],
            "auroc_lo": stats["auroc_lo"],
            "auroc_hi": stats["auroc_hi"],
            "aupr": stats["aupr"],
            "aupr_lo": stats["aupr_lo"],
            "aupr_hi": stats["aupr_hi"],
            "preferred": False,
        }
        ok_rows.append(row)
        rec = " [recovered JSON]" if recovered else ""
        print(
            f"  OK    {run.dirname}  {run.experiment} {run.task}  "
            f"AUROC {stats['auroc']:.3f} [{stats['auroc_lo']:.3f}, {stats['auroc_hi']:.3f}]"
            f"{rec}"
        )
    mark_preferred(ok_rows)
    return ok_rows, missing_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=None,
        help="Experiment Results directory (default: configs/local.yaml results_root)",
    )
    parser.add_argument("--local-yaml", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="Resample patient_ids when present (more conservative than example-level).",
    )
    parser.add_argument(
        "--preferred-only",
        action="store_true",
        help="Write only the preferred run per experiment/model/task (still scans all).",
    )
    args = parser.parse_args(argv)

    results_root = (
        Path(args.results_root) if args.results_root else _results_root_from_local(args.local_yaml)
    )
    print(f"Results root: {results_root}")
    print(f"n_boot={args.n_boot}  ci={args.ci}  clustered={args.clustered}")

    runs = collect_runs(results_root)
    print(f"Found {len(runs)} run directories")
    ok_rows, missing_rows = bootstrap_runs(
        runs,
        n_boot=args.n_boot,
        ci=args.ci,
        seed=args.seed,
        clustered=args.clustered,
    )
    if args.preferred_only:
        out_rows = [r for r in ok_rows if r.get("preferred")]
    else:
        out_rows = ok_rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ci_fields = [
        "experiment",
        "model",
        "task",
        "variant",
        "preferred",
        "legacy_name",
        "dirname",
        "dataset",
        "n",
        "n_pos",
        "n_boot",
        "clustered",
        "json_recovered",
        "auroc",
        "auroc_lo",
        "auroc_hi",
        "aupr",
        "aupr_lo",
        "aupr_hi",
        "pred_file",
    ]
    miss_fields = [
        "experiment",
        "model",
        "task",
        "variant",
        "legacy_name",
        "dirname",
        "dataset",
        "reason",
        "pred_file",
    ]
    ci_path = args.out_dir / "bootstrap_ci.csv"
    miss_path = args.out_dir / "missing_test_predictions.csv"
    _write_csv(ci_path, out_rows, ci_fields)
    _write_csv(miss_path, missing_rows, miss_fields)

    summary = {
        "results_root": str(results_root),
        "n_boot": args.n_boot,
        "ci": args.ci,
        "clustered": args.clustered,
        "n_ok": len(ok_rows),
        "n_preferred": sum(1 for r in ok_rows if r.get("preferred")),
        "n_missing": len(missing_rows),
        "alias_map": LEGACY_EXPERIMENT_ALIASES,
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {len(out_rows)} CIs → {ci_path}")
    print(f"Wrote {len(missing_rows)} missing → {miss_path}")


if __name__ == "__main__":
    main()
