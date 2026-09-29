"""
Start here for every training / scoring job.

sbatch scripts/<site>/run.sh  →  resolved YAML  →  this file.

Routing lives in ``src/experiment_map.py``. Every runner implements ``run()``;
those with ``supports_eval`` also implement ``run_eval_only(checkpoint)``.

    torchrun ... src/pipelines/run.py --config_filepath configs/_resolved/...
    python -m src.pipelines.run --config_filepath configs/_resolved/...
    python -m src.pipelines.run --config_filepath ... --eval_only
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import pprint

from src.experiment_map import get_spec
from src.training.config_utils import apply_smoke_config, load_config
from src.training.training_utils import (
    check_clobber,
    find_eval_checkpoint,
    prepare_run_reproducibility,
    resolve_checkpoint_selection,
)

logger = logging.getLogger(__name__)


def _load_class(dotted: str):
    module_path, class_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _experiment(config: dict) -> str:
    exp = (config.get("_meta") or {}).get("experiment")
    if not exp:
        raise SystemExit(
            "ERROR: resolved config is missing _meta.experiment. "
            "Build it with: python -m src.training.config_utils build ..."
        )
    return str(exp)


def _sync_output_dir(config: dict) -> str:
    training = config.setdefault("training", {})
    output_dir = training.get("output_dir") or config.get("output_dir")
    if not output_dir:
        raise SystemExit("ERROR: config has no training.output_dir / output_dir")
    training["output_dir"] = output_dir
    config["output_dir"] = output_dir
    return output_dir


def _select_eval_checkpoint(
    args: argparse.Namespace, output_dir: str, ckpt: str | None
) -> str:
    if args.checkpoint_path:
        print(f"Evaluating  : {args.checkpoint_path}  (given explicitly)")
        return args.checkpoint_path

    found = ckpt or find_eval_checkpoint(output_dir)
    if not found:
        raise SystemExit(
            f"ERROR: no checkpoint under {output_dir}. "
            "Pass --checkpoint_path / CHECKPOINT= to a saved run."
        )
    if os.path.isdir(found) and os.path.basename(os.path.normpath(found)).startswith(
        "checkpoint-"
    ):
        selected = resolve_checkpoint_selection(
            found, output_dir, args.checkpoint_selection
        )
        print(f"Newest ckpt : {found}")
        print(f"Evaluating  : {selected}  (selection={args.checkpoint_selection})")
        return selected
    print(f"Evaluating  : {found}")
    return found


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tEHR experiment from a resolved config YAML"
    )
    parser.add_argument("--config_filepath", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--resume_from_last",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Score a checkpoint; do not train",
    )
    parser.add_argument(
        "--checkpoint_selection",
        choices=("best", "latest"),
        default="latest",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb", action="store_true", dest="wandb_enabled")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config_filepath)
    if os.environ.get("TEHR_SMOKE", "").strip().lower() in {"1", "true", "yes", "on"}:
        config = apply_smoke_config(config, eval_only=bool(args.eval_only))
        print(
            "TEHR_SMOKE=1: outputs under _smoke, "
            f"max_steps={config.get('training', {}).get('max_steps')} "
            f"skip_final_eval={config.get('training', {}).get('skip_final_eval')}"
        )
        if args.eval_only:
            # _smoke trees are disposable; a train smoke's result files must
            # not block the matching evaluate smoke.
            args.overwrite = True
    experiment = _experiment(config)
    spec = get_spec(experiment)

    logger.info("=" * 80)
    logger.info(f"Experiment: {experiment}")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config_filepath}")
    pprint.pprint(config)

    if args.eval_only and not spec.supports_eval:
        raise SystemExit(f"ERROR: {spec.eval_error or f'{experiment} does not support evaluate'}")

    resume_from_last = args.resume_from_last
    if args.eval_only and resume_from_last is None and args.checkpoint_path is None:
        resume_from_last = True

    seed, ckpt = prepare_run_reproducibility(
        config,
        cli_seed=args.seed,
        cli_checkpoint=args.checkpoint_path,
        cli_resume_from_last=resume_from_last,
        use_seed_subdir=True,
    )
    output_dir = _sync_output_dir(config)
    logger.info(f"Using seed: {seed}")
    print(f"Seed        : {seed}")
    print(f"Output dir  : {output_dir}")

    if args.eval_only:
        config.setdefault("wandb", {})["enabled"] = bool(args.wandb_enabled)
        ckpt = _select_eval_checkpoint(args, output_dir, ckpt)
        check_clobber(output_dir, args.overwrite, spec.eval_result_files)

    runner = _load_class(spec.runner)(config)
    if args.eval_only:
        print(f"evaluate: scoring {ckpt} (no training)")
        val_metrics, test_metrics = runner.run_eval_only(ckpt)
        logger.info(f"  Validation: {val_metrics}")
        logger.info(f"  Test:       {test_metrics}")
        logger.info("Evaluation completed successfully!")
        return

    if ckpt:
        logger.info(f"Resuming from: {ckpt}")
    result = runner.run(resume_from_checkpoint=ckpt)
    if result is None:
        val_metrics, test_metrics = None, None
    else:
        val_metrics, test_metrics = result
    if val_metrics is not None:
        print(f"Val: {val_metrics}")
    if test_metrics is not None:
        print(f"Test: {test_metrics}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
