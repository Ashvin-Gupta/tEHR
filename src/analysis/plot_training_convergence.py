#!/usr/bin/env python3
"""
Plot validation AUROC / AUPR vs training progress for text (LLM classifier) vs code (encoder).

Preferred data sources (matched 1-epoch protocol):
  - LLM: ``trainer_state.json`` (eval_auroc / eval_aupr at epoch fractions)
  - Encoder: ``results/convergence.jsonl`` written during matched fraction evals

Legacy fallback:
  - Encoder: Slurm log epoch-end AUROC/AUPR lines

X-axis default is epoch fraction (examples_seen / N_train), not optimizer steps.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import yaml

DEFAULT_RESULTS_ROOT = Path("/projects/u6ps/TextCancEHR2/experiments/Results")
DEFAULT_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"


@dataclass
class EvalPoint:
    step: Optional[int]
    epoch: Optional[float]
    progress: Optional[float]
    auroc: Optional[float]
    aupr: Optional[float]
    source: str
    examples_seen: Optional[int] = None


def _find_latest_trainer_state(results_dir: Path) -> Optional[Path]:
    candidates = sorted(results_dir.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        candidates = sorted(results_dir.glob("**/checkpoint-*/trainer_state.json"))
    if not candidates:
        return None

    def _step(p: Path) -> int:
        m = re.search(r"checkpoint-(\d+)", str(p))
        return int(m.group(1)) if m else -1

    return max(candidates, key=_step)


def _load_training_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None or not config_path.is_file():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _effective_batch_size(training_cfg: Dict[str, Any]) -> int:
    batch = int(training_cfg.get("batch_size", 8))
    grad = int(training_cfg.get("gradient_accumulation_steps", 1))
    return max(1, batch * grad)


def parse_llm_trainer_state(
    trainer_state_path: Path,
    *,
    train_samples: Optional[int] = None,
    effective_batch: Optional[int] = None,
) -> Tuple[List[EvalPoint], Dict[str, Any]]:
    with open(trainer_state_path, "r") as f:
        state = json.load(f)

    max_steps = int(state.get("max_steps") or 0)
    if effective_batch is None:
        effective_batch = 8
    if train_samples is None and max_steps > 0:
        train_samples = max_steps * effective_batch

    points: List[EvalPoint] = []
    for row in state.get("log_history", []):
        if "eval_auroc" not in row:
            continue
        step = int(row["step"])
        epoch = float(row.get("epoch", float("nan")))
        progress = float(epoch) if not math.isnan(epoch) else ((step / max_steps) if max_steps else None)
        examples = int(step * effective_batch) if effective_batch else None
        points.append(
            EvalPoint(
                step=step,
                epoch=epoch,
                progress=progress,
                auroc=float(row["eval_auroc"]),
                aupr=float(row["eval_aupr"]) if "eval_aupr" in row else None,
                source="llm",
                examples_seen=examples,
            )
        )

    points.sort(key=lambda p: p.step or 0)
    meta = {
        "max_steps": max_steps,
        "train_samples": train_samples,
        "effective_batch": effective_batch,
        "trainer_state": str(trainer_state_path),
        "n_eval_points": len(points),
    }
    return points, meta


def parse_convergence_jsonl(path: Path) -> Tuple[List[EvalPoint], Dict[str, Any]]:
    points: List[EvalPoint] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            points.append(
                EvalPoint(
                    step=row.get("step"),
                    epoch=row.get("epoch_fraction"),
                    progress=row.get("epoch_fraction"),
                    auroc=row.get("auroc"),
                    aupr=row.get("aupr"),
                    source=str(row.get("source", "encoder")),
                    examples_seen=row.get("examples_seen"),
                )
            )
    points.sort(key=lambda p: (p.progress is None, p.progress or 0.0, p.step or 0))
    meta = {
        "convergence_jsonl": str(path),
        "n_eval_points": len(points),
    }
    return points, meta


_EPOCH_RE = re.compile(r"\[Classification\] Epoch (\d+)/(\d+)")
_AUROC_RE = re.compile(r"^\s*AUROC:\s*([0-9.]+)")
_AUPR_RE = re.compile(r"^\s*AUPR:\s*([0-9.]+)")


def parse_encoder_log(
    log_path: Path,
    *,
    steps_per_epoch: Optional[int] = None,
    batch_size: int = 128,
    train_samples: Optional[int] = None,
) -> Tuple[List[EvalPoint], Dict[str, Any]]:
    text = log_path.read_text(errors="replace").splitlines()

    if train_samples and steps_per_epoch is None:
        steps_per_epoch = max(1, math.ceil(train_samples / batch_size))

    points: List[EvalPoint] = []
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    pending_auroc: Optional[float] = None
    in_validation_block = False
    done = False

    for line in text:
        if done:
            break

        if "TEST Results:" in line:
            done = True
            continue

        m_ep = _EPOCH_RE.search(line)
        if m_ep:
            current_epoch = int(m_ep.group(1))
            total_epochs = int(m_ep.group(2))
            pending_auroc = None
            in_validation_block = False
            continue

        if "VALIDATION Results:" in line and current_epoch is not None:
            in_validation_block = True
            pending_auroc = None
            continue

        if not in_validation_block:
            continue

        m_au = _AUROC_RE.match(line)
        if m_au:
            pending_auroc = float(m_au.group(1))
            continue

        m_ap = _AUPR_RE.match(line)
        if m_ap and pending_auroc is not None and current_epoch is not None:
            aupr = float(m_ap.group(1))
            step = None
            progress = None
            if steps_per_epoch:
                step = current_epoch * steps_per_epoch
            if total_epochs and total_epochs > 0:
                progress = current_epoch / total_epochs
            points.append(
                EvalPoint(
                    step=step,
                    epoch=float(current_epoch),
                    progress=progress,
                    auroc=pending_auroc,
                    aupr=aupr,
                    source="encoder",
                )
            )
            pending_auroc = None
            in_validation_block = False
            if total_epochs is not None and current_epoch >= total_epochs:
                done = True

    meta = {
        "log_path": str(log_path),
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "total_epochs": total_epochs,
        "n_eval_points": len(points),
    }
    return points, meta


def _find_log_for_output_dir(logs_dir: Path, output_dir_substr: str) -> Optional[Path]:
    if not logs_dir.is_dir():
        return None
    matches: List[Path] = []
    for log in sorted(logs_dir.glob("isa-mimic-run-*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            if output_dir_substr in log.read_text(errors="replace"):
                matches.append(log)
        except OSError:
            continue
    return matches[0] if matches else None


def _default_paths(
    task: str,
    model: str,
    seed: int,
    results_root: Path,
) -> Tuple[Path, Path, str]:
    llm_dir = results_root / f"mimic-classifier-{model}-{task}" / f"seed_{seed}"
    encoder_dir = results_root / f"mimic-encoder-transformer-{task}"
    encoder_log_hint = f"mimic-encoder-transformer-{task}"
    return llm_dir, encoder_dir, encoder_log_hint


def _x_values(points: Sequence[EvalPoint], x_axis: str) -> List[float]:
    out: List[float] = []
    for p in points:
        if x_axis == "steps":
            if p.step is not None:
                out.append(float(p.step))
            elif p.progress is not None:
                out.append(p.progress * 100.0)
        elif x_axis == "progress":
            if p.progress is not None:
                out.append(p.progress * 100.0)
            elif p.step is not None:
                out.append(float(p.step))
        elif x_axis == "examples":
            if p.examples_seen is not None:
                out.append(float(p.examples_seen))
            elif p.progress is not None:
                out.append(p.progress * 100.0)
        else:
            raise ValueError(f"Unknown x_axis: {x_axis}")
    return out


def plot_convergence(
    llm_points: Sequence[EvalPoint],
    encoder_points: Sequence[EvalPoint],
    *,
    task: str,
    x_axis: str,
    out_path: Path,
    llm_label: str = "Text (LLM classifier)",
    encoder_label: str = "Code (encoder)",
    title: Optional[str] = None,
    footnote: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    metric_specs = [("auroc", "Validation AUROC"), ("aupr", "Validation AUPR")]

    colors = {"llm": "#2563eb", "encoder": "#dc2626"}
    if x_axis == "steps":
        x_label = "Training steps"
    elif x_axis == "examples":
        x_label = "Labeled examples seen"
    else:
        x_label = "Fraction of one labeled epoch (%)"

    for ax, (metric_key, metric_title) in zip(axes, metric_specs):
        for points, label, key in (
            (llm_points, llm_label, "llm"),
            (encoder_points, encoder_label, "encoder"),
        ):
            xs = _x_values(points, x_axis)
            ys = [getattr(p, metric_key) for p in points]
            pairs = [
                (x, y)
                for x, y in zip(xs, ys)
                if y is not None and not (isinstance(y, float) and math.isnan(y))
            ]
            if not pairs:
                continue
            px, py = zip(*pairs)
            ax.plot(px, py, marker="o", markersize=4, linewidth=1.8, label=label, color=colors[key])

        ax.set_title(metric_title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    if footnote:
        fig.text(0.5, -0.02, footnote, ha="center", va="top", fontsize=7.5, wrap=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_footnote(llm_meta: Dict[str, Any], enc_meta: Dict[str, Any], x_axis: str) -> str:
    parts = [
        "Validation metrics during matched supervised adaptation (not held-out test).",
        f"LLM: {llm_meta.get('n_eval_points', 0)} eval point(s).",
        f"Encoder: {enc_meta.get('n_eval_points', 0)} eval point(s).",
    ]
    if "convergence_jsonl" in enc_meta:
        parts.append("Encoder curve from convergence.jsonl (epoch-fraction schedule).")
    if x_axis == "progress":
        parts.append("X-axis = examples_seen / N_train.")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training convergence: text LLM vs code encoder.")
    parser.add_argument("--task", required=True, choices=["icu_mortality", "hospital_mortality", "hospital_readmission"])
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--llm-results-dir", type=Path, default=None)
    parser.add_argument("--encoder-results-dir", type=Path, default=None)
    parser.add_argument("--encoder-log", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--x-axis",
        choices=["progress", "examples", "steps"],
        default="progress",
        help="progress: %% of one labeled epoch (default); examples: raw examples seen; steps: optimizer steps",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    args = parser.parse_args()

    llm_dir_default, encoder_dir_default, encoder_log_hint = _default_paths(
        args.task, args.model, args.seed, args.results_root
    )
    llm_dir = args.llm_results_dir or llm_dir_default
    encoder_dir = args.encoder_results_dir or encoder_dir_default

    config_path = args.config
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "mimic"
            / "_resolved_isambard"
            / f"classifier-{args.model}-{args.task}.yaml"
        )
    training_cfg = _load_training_config(config_path).get("training", {})
    eff_batch = _effective_batch_size(training_cfg)

    trainer_state = _find_latest_trainer_state(llm_dir)
    if trainer_state is None:
        raise FileNotFoundError(f"No trainer_state.json under {llm_dir}")

    llm_points, llm_meta = parse_llm_trainer_state(trainer_state, effective_batch=eff_batch)
    train_samples = llm_meta.get("train_samples")

    enc_points: List[EvalPoint] = []
    enc_meta: Dict[str, Any] = {}
    jsonl = encoder_dir / "results" / "convergence.jsonl"
    if jsonl.is_file():
        enc_points, enc_meta = parse_convergence_jsonl(jsonl)
    else:
        encoder_log = args.encoder_log
        if encoder_log is None:
            encoder_log = _find_log_for_output_dir(args.logs_dir, encoder_log_hint)
        if encoder_log is None or not encoder_log.is_file():
            raise FileNotFoundError(
                f"No encoder convergence.jsonl at {jsonl} and no --encoder-log "
                f"(hint: logs containing '{encoder_log_hint}')."
            )
        enc_points, enc_meta = parse_encoder_log(
            encoder_log,
            train_samples=train_samples,
            batch_size=args.encoder_batch_size,
        )

    out_dir = args.out_dir or (
        Path(__file__).resolve().parents[2] / "analysis_out" / f"convergence-{args.model}-{args.task}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    footnote = build_footnote(llm_meta, enc_meta, args.x_axis)
    title = f"Training convergence — {args.task.replace('_', ' ')}"

    plot_path = out_dir / f"convergence_{args.x_axis}.png"
    plot_convergence(
        llm_points,
        enc_points,
        task=args.task,
        x_axis=args.x_axis,
        out_path=plot_path,
        title=title,
        footnote=footnote,
    )

    summary = {
        "task": args.task,
        "model": args.model,
        "seed": args.seed,
        "x_axis": args.x_axis,
        "llm": {"points": [asdict(p) for p in llm_points], "meta": llm_meta},
        "encoder": {"points": [asdict(p) for p in enc_points], "meta": enc_meta},
        "plots": {"primary": str(plot_path)},
        "caveats": footnote,
    }
    json_path = out_dir / "convergence_curves.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {plot_path}")
    print(f"Wrote {json_path}")
    print(f"\nLLM eval points: {len(llm_points)} | Encoder eval points: {len(enc_points)}")


if __name__ == "__main__":
    main()
