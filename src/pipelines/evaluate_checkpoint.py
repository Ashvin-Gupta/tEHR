"""
Evaluate an already-trained classifier checkpoint, without training.

Recovers the test metrics for runs that were killed by the wall clock after
their last checkpoint but before the final evaluation — the state most of the
long ablation jobs end up in. The run directory is found the same way
``training.resume_from_last`` finds it (newest ``checkpoint-*`` under the run's
seed directory), so no path has to be typed out. That newest checkpoint is
scored by default.

Writes the same files a completed run writes, into the same place:

    <output_dir>/seed_<seed>/results/test_results.json
    <output_dir>/seed_<seed>/results/eval_provenance.json   (which checkpoint)
    <output_dir>/seed_<seed>/plots_test/

Validation is deliberately skipped — only the held-out test set is scored.
Existing results are never silently replaced: pass ``--overwrite`` to do that.

Supports the experiments that train through ``finetune_llm_classifier.py``:
classifier, ablation_no_pretrain_lora, ablation_frozen_pretrained, surface_raw,
surface_scramble_hash, surface_scramble_shuffle.

Also supports SFT (``finetune_llm_sft_classifier.py``): scores tuning (validation)
and held_out (test) via logit extraction.

``ablation_frozen_nopretrain`` uses the baseline path instead
(``run_baseline --eval_only``); the dispatcher routes it the same way via the
``evaluate`` flag.

Usage:
    torchrun --nproc_per_node=4 src/pipelines/evaluate_checkpoint.py \\
        --config_filepath src/configs/mimic/_resolved_isambard/classifier-mistral-hospital_readmission.yaml

    # SFT — same dispatcher, append ``evaluate``:
    sbatch src/submissions/isambard/run.sh sft mistral hospital_mortality evaluate

    # or through the dispatcher, which resolves the config for you:
    sbatch src/submissions/isambard/run.sh classifier mistral hospital_readmission evaluate
"""
import argparse
import json
import logging
import os
import pprint

from src.training.config_utils import load_config
from src.training.training_utils import prepare_run_reproducibility

logger = logging.getLogger(__name__)

# Only the terminal artifacts block a re-run. validation_results.json is written
# before the test split is scored, so a job killed partway through the final
# evaluation leaves one behind — that is the case this script exists to fix, and
# it must not be mistaken for a finished run.
FINAL_RESULT_FILES = ("test_results.json", "combined_results.json", "held_out_results.json")
SFT_FINAL_RESULT_FILES = ("held_out_results.json", "summary.json")


def _is_sft_experiment(config: dict) -> bool:
    meta = config.get("_meta") or {}
    if meta.get("experiment") == "sft":
        return True
    name = str(config.get("name", ""))
    return name.startswith("mimic_sft_")


def _resolve_selection(latest_ckpt: str, output_dir: str, selection: str) -> str:
    """
    Choose between the newest checkpoint and the best-scoring one.

    Default is ``latest``: the newest ``checkpoint-*`` on disk. Pass
    ``selection='best'`` to follow ``best_model_checkpoint`` in the newest
    checkpoint's ``trainer_state.json`` (what ``load_best_model_at_end`` would
    have restored at the end of a completed run).
    """
    if selection == "latest":
        return latest_ckpt

    state_path = os.path.join(latest_ckpt, "trainer_state.json")
    if not os.path.isfile(state_path):
        print(f"  ! No trainer_state.json in {latest_ckpt}; using the newest checkpoint.")
        return latest_ckpt

    try:
        with open(state_path) as f:
            state = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"  ! Could not read {state_path} ({exc}); using the newest checkpoint.")
        return latest_ckpt

    best = state.get("best_model_checkpoint")
    if not best:
        print("  ! trainer_state.json records no best checkpoint; using the newest.")
        return latest_ckpt

    if not os.path.isdir(best):
        # The recorded path is absolute and from another site/run directory.
        relocated = os.path.join(output_dir, os.path.basename(best.rstrip("/")))
        if os.path.isdir(relocated):
            best = relocated
        else:
            print(
                f"  ! Best checkpoint {best} no longer exists "
                "(likely pruned by save_total_limit); using the newest."
            )
            return latest_ckpt

    metric = state.get("best_metric")
    print(
        f"  - Best checkpoint by validation {state.get('metric_for_best_model', 'auroc')}"
        f"={metric}: {os.path.basename(best)}"
    )
    return best


def _check_clobber(output_dir: str, overwrite: bool, is_sft: bool = False):
    """Refuse to overwrite the results of an already-completed evaluation."""
    results_dir = os.path.join(output_dir, "results")
    result_files = SFT_FINAL_RESULT_FILES if is_sft else FINAL_RESULT_FILES
    existing = [
        f for f in result_files if os.path.isfile(os.path.join(results_dir, f))
    ]
    if is_sft and os.path.isfile(os.path.join(output_dir, "summary.json")):
        existing.append("summary.json")
    existing = sorted(set(existing))
    if not existing or overwrite:
        return
    raise SystemExit(
        f"ERROR: {results_dir} already contains {', '.join(existing)}.\n"
        "That run already has final metrics, which this would replace. Re-run\n"
        "with --overwrite (or OVERWRITE=true via the submission script) if that\n"
        "is what you want, or move the old files aside first."
    )


def main(
    config_path: str,
    seed: int | None = None,
    checkpoint_path: str | None = None,
    checkpoint_selection: str = "latest",
    overwrite: bool = False,
    wandb_enabled: bool = False,
):
    logger.info("=" * 80)
    logger.info("Checkpoint evaluation (test set only, no training)")
    logger.info("=" * 80)
    logger.info(f"\nLoading configuration from: {config_path}")

    config = load_config(config_path)
    logger.info("Loaded configuration:")
    pprint.pprint(config)

    # resume_from_last=True makes prepare_run_reproducibility apply the seed
    # subdirectory and hand back the newest checkpoint under it — the same
    # discovery training uses, so no path has to be supplied. An explicit
    # --checkpoint_path short-circuits all of this.
    seed, ckpt = prepare_run_reproducibility(
        config,
        cli_seed=seed,
        cli_checkpoint=checkpoint_path,
        cli_resume_from_last=True,
        use_seed_subdir=True,
    )

    output_dir = config["training"]["output_dir"]
    if not ckpt:
        raise SystemExit(
            f"ERROR: no checkpoint-* directory found under {output_dir}\n"
            "Nothing to evaluate. Check the seed (--seed) and that training got "
            "far enough to save a checkpoint."
        )

    latest_ckpt = None
    if checkpoint_path is None:
        latest_ckpt = ckpt
        ckpt = _resolve_selection(latest_ckpt, output_dir, checkpoint_selection)

    _check_clobber(output_dir, overwrite, is_sft=_is_sft_experiment(config))

    # Eval-only runs should not open a new training run in wandb by default.
    config.setdefault("wandb", {})["enabled"] = bool(wandb_enabled)

    print(f"Seed        : {seed}")
    print(f"Output dir  : {output_dir}")
    if latest_ckpt is None:
        print(f"Evaluating  : {ckpt}  (given explicitly)")
    else:
        print(f"Newest ckpt : {latest_ckpt}")
        print(f"Evaluating  : {ckpt}  (selection={checkpoint_selection})")

    # Imported here, not at module scope: it pulls in Unsloth, which raises on
    # import without a visible GPU. Keeping it local lets --help and the
    # checkpoint/clobber checks above run anywhere.
    if _is_sft_experiment(config):
        from src.training.sft_classification_trainer import SFTClassificationTrainer

        trainer = SFTClassificationTrainer(config)
        val_metrics, test_metrics = trainer.run_eval_only(ckpt)
        logger.info("\n" + "=" * 80)
        logger.info("SFT evaluation completed successfully!")
        logger.info("=" * 80)
        logger.info(f"  Validation: {val_metrics}")
        logger.info(f"  Test:       {test_metrics}")
        return val_metrics, test_metrics

    from src.training.classification_trainer import EHRClassificationTrainer

    trainer = EHRClassificationTrainer(config)
    _, test_results = trainer.run_eval_only(ckpt)

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 80)

    return None, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier checkpoint on the test set (no training)"
    )
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed subdirectory to evaluate (default: config seed, usually 42)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit checkpoint to score (default: auto-resolved, see --checkpoint_selection)",
    )
    parser.add_argument(
        "--checkpoint_selection",
        choices=("best", "latest"),
        default="latest",
        help=(
            "Which auto-resolved checkpoint to score. 'latest' (default) uses "
            "the newest checkpoint-*; 'best' uses best_model_checkpoint from "
            "trainer_state.json (the load_best_model_at_end pick)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing results/*.json instead of refusing to run",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        dest="wandb_enabled",
        help="Log this evaluation to wandb (off by default)",
    )
    args = parser.parse_args()

    main(
        args.config_filepath,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        checkpoint_selection=args.checkpoint_selection,
        overwrite=args.overwrite,
        wandb_enabled=args.wandb_enabled,
    )
