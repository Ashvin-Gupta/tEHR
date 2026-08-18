"""
Matched supervised evaluation schedule for LLM classifier vs code encoder.

Both models evaluate validation AUROC/AUPR at the same fractions of one
labeled epoch (examples seen / N_train), not raw optimizer steps.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

# Paper-default matched checkpoints within one supervised epoch.
DEFAULT_EVAL_EPOCH_FRACTIONS: List[float] = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]


def get_eval_epoch_fractions(training_config: Dict[str, Any]) -> List[float]:
    """Return configured epoch fractions, or the paper default."""
    raw = training_config.get("eval_epoch_fractions", DEFAULT_EVAL_EPOCH_FRACTIONS)
    if raw is None or raw == "default":
        return list(DEFAULT_EVAL_EPOCH_FRACTIONS)
    fracs = sorted({float(x) for x in raw if float(x) > 0.0})
    if not fracs:
        raise ValueError("training.eval_epoch_fractions must be a non-empty list of >0 values")
    return fracs


def world_size() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def effective_train_batch_size(
    per_device_batch: int,
    grad_accum: int = 1,
    n_devices: Optional[int] = None,
) -> int:
    devices = world_size() if n_devices is None else max(1, int(n_devices))
    return max(1, int(per_device_batch) * max(1, int(grad_accum)) * devices)


def steps_per_epoch(n_train: int, effective_batch: int) -> int:
    return max(1, math.ceil(int(n_train) / max(1, int(effective_batch))))


def eval_optimizer_steps_from_fractions(
    n_train: int,
    fractions: Sequence[float],
    *,
    per_device_batch: int,
    grad_accum: int = 1,
    n_devices: Optional[int] = None,
) -> List[int]:
    """
    Map epoch fractions → HuggingFace global optimizer steps for one epoch.

    examples_seen ≈ step * effective_batch
    progress = examples_seen / n_train ≈ step / steps_per_epoch
    """
    eff = effective_train_batch_size(per_device_batch, grad_accum, n_devices)
    spe = steps_per_epoch(n_train, eff)
    steps: List[int] = []
    for f in fractions:
        f = float(f)
        if f >= 1.0:
            step = spe
        else:
            step = max(1, int(round(f * spe)))
        steps.append(min(step, spe))
    # Unique, sorted; keep final step even if earlier fractions collide on tiny sets.
    return sorted(set(steps))


def next_fraction_targets(fractions: Sequence[float]) -> List[float]:
    return sorted({float(f) for f in fractions if float(f) > 0.0})


def EvalAtStepsCallback(*args, **kwargs):
    """Lazy wrapper so importing this module does not load transformers."""
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class _EvalAtStepsCallback(TrainerCallback):
        """
        Evaluate (and checkpoint) only at the given optimizer steps.

        Pair with eval_strategy/save_strategy = "steps" and a huge eval_steps/
        save_steps so the Trainer's built-in schedule never fires on its own.
        """

        def __init__(self, eval_steps: Sequence[int], fractions: Optional[Sequence[float]] = None):
            self.eval_steps = set(int(s) for s in eval_steps)
            self.fractions = list(fractions) if fractions is not None else None
            self._max_step = max(self.eval_steps) if self.eval_steps else 0

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            print(
                f"  - Matched eval schedule: {len(self.eval_steps)} checkpoints "
                f"at optimizer steps {sorted(self.eval_steps)}"
            )
            if self.fractions is not None:
                print(f"  - Epoch fractions: {self.fractions}")
            return control

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            if state.global_step in self.eval_steps:
                control.should_evaluate = True
                control.should_save = True
            return control

    return _EvalAtStepsCallback(*args, **kwargs)


def MaxRuntimeCallback(*args, **kwargs):
    """Lazy wrapper so importing this module does not load transformers."""
    import time
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class _MaxRuntimeCallback(TrainerCallback):
        """
        Stop training after ``max_runtime_hours`` wall-clock from train start.

        Leaves time before a SLURM walltime kill for the usual post-train
        val/test evaluation in ``run_full_pipeline`` / baseline ``run()``.
        """

        def __init__(self, max_runtime_hours: float):
            hours = float(max_runtime_hours)
            if hours <= 0:
                raise ValueError(f"max_runtime_hours must be > 0, got {hours}")
            self.max_runtime_sec = hours * 3600.0
            self._start: Optional[float] = None
            self._stopped = False

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            self._start = time.monotonic()
            print(
                f"  - Max runtime: {self.max_runtime_sec / 3600.0:.1f}h from train start "
                f"(early stop → final val/test eval before walltime kill)"
            )
            return control

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            if self._stopped or self._start is None:
                return control
            elapsed = time.monotonic() - self._start
            if elapsed >= self.max_runtime_sec:
                self._stopped = True
                print(
                    f"\n  - Max runtime reached ({elapsed / 3600.0:.2f}h at step "
                    f"{state.global_step}); stopping training for final evaluation.\n",
                    flush=True,
                )
                control.should_training_stop = True
                control.should_save = True
                control.should_evaluate = True
            return control

    return _MaxRuntimeCallback(*args, **kwargs)


def append_max_runtime_callback(
    callbacks: List[Any],
    training_config: Dict[str, Any],
) -> List[Any]:
    """Append ``MaxRuntimeCallback`` when ``training.max_runtime_hours`` is set."""
    raw = training_config.get("max_runtime_hours")
    if raw is None or raw is False:
        return callbacks
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"training.max_runtime_hours must be a number, got {raw!r}")
    if hours <= 0:
        return callbacks
    callbacks.append(MaxRuntimeCallback(hours))
    return callbacks


def save_convergence_point(
    path: str,
    *,
    source: str,
    epoch_fraction: float,
    examples_seen: int,
    n_train: int,
    step: Optional[int],
    metrics: Dict[str, Any],
) -> None:
    """Append one validation point to a JSONL convergence log."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {
        "source": source,
        "epoch_fraction": float(epoch_fraction),
        "examples_seen": int(examples_seen),
        "n_train": int(n_train),
        "step": int(step) if step is not None else None,
        "auroc": float(metrics["auroc"]) if metrics.get("auroc") is not None else None,
        "aupr": float(metrics["aupr"]) if metrics.get("aupr") is not None else None,
        "eval_loss": float(metrics["loss"]) if metrics.get("loss") is not None else None,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
