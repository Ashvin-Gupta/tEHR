import argparse
import importlib
import os

from src.training.config_utils import load_config
from src.training.training_utils import prepare_run_reproducibility

BASELINE_REGISTRY = {
    "frozen_llm_linear":     "src.baselines.baseline_frozen_llm_linear.FrozenLLMLinearBaseline",
    "pretrained_llm_linear": "src.baselines.baseline_pretrained_llm_linear.PretrainedLLMLinearBaseline",
    "lora_llm_linear":       "src.baselines.baseline_lora_llm_linear.LoRALLMLinearBaseline",
    "pure_llm":              "src.baselines.baseline_pure_llm.PureLLMBaseline",
    "clinicalbert":          "src.baselines.baseline_clinicalbert.ClinicalBERTBaseline",
    "xgboost":               "src.baselines.baseline_xgboost.XGBoostBaseline",
    "xgboost_bow":           "src.baselines.baseline_xgboost_bow.XGBoostBOWBaseline",
    "encoder_transformer":   "src.baselines.baseline_encoder_transformer.EncoderTransformerBaseline",
    "encoder_transformer_hparam_search": "src.baselines.baseline_encoder_transformer_hparam_search.EncoderTransformerHyperparamSearchBaseline",
    "qwen3_embedding":       "src.baselines.baseline_qwen3_embedding.Qwen3EmbeddingBaseline",
    "qwen_llm_linear":       "src.baselines.baseline_qwen_llm_linear.QwenLLMLinearBaseline",
    "logreg_demo_events":    "src.baselines.baseline_logreg_demo_events.LogRegDemoEventsBaseline",
    "text_embedding_transformer": "src.baselines.baseline_text_embedding_transformer.TextEmbeddingTransformerBaseline",
}

# Baselines that implement run_eval_only (LLMClassifier Trainer checkpoints).
EVAL_ONLY_BASELINES = {
    "frozen_llm_linear",
    "pretrained_llm_linear",
    "lora_llm_linear",
    "clinicalbert",
}

FINAL_RESULT_FILES = ("held_out_results.json", "test_results.json", "combined_results.json")


def load_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _check_clobber(output_dir: str, overwrite: bool):
    results_dir = os.path.join(output_dir, "results")
    existing = [
        f for f in FINAL_RESULT_FILES if os.path.isfile(os.path.join(results_dir, f))
    ]
    if not existing or overwrite:
        return
    raise SystemExit(
        f"ERROR: {results_dir} already contains {', '.join(existing)}.\n"
        "That run already has final metrics, which this would replace. Re-run\n"
        "with --overwrite (or OVERWRITE=true via the submission script) if that\n"
        "is what you want, or move the old files aside first."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=BASELINE_REGISTRY.keys())
    parser.add_argument("--config_filepath", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit Trainer checkpoint to resume/eval from (overrides resume_from_last)",
    )
    parser.add_argument(
        "--resume_from_last",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override training.resume_from_last (auto-resume newest checkpoint-*)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help=(
            "Score the newest (or --checkpoint_path) checkpoint on held_out only; "
            "do not train. Supported for LLM-classifier baselines."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="With --eval_only: replace existing results/*.json instead of refusing",
    )
    args = parser.parse_args()

    config = load_config(args.config_filepath)

    # Align top-level output_dir with training.output_dir when only the latter is set.
    training = config.setdefault("training", {})
    if not config.get("output_dir") and training.get("output_dir"):
        config["output_dir"] = training["output_dir"]
    if training.get("output_dir") is None and config.get("output_dir"):
        training["output_dir"] = config["output_dir"]

    # Eval-only always resolves the newest checkpoint under the run dir
    # (output_dir/checkpoints for these baselines). An explicit path still wins.
    resume_from_last = args.resume_from_last
    if args.eval_only and resume_from_last is None and args.checkpoint_path is None:
        resume_from_last = True

    seed, resume_ckpt = prepare_run_reproducibility(
        config,
        cli_seed=args.seed,
        cli_checkpoint=args.checkpoint_path,
        cli_resume_from_last=resume_from_last,
        use_seed_subdir=False,
    )
    # Keep top-level output_dir in sync after any path rewrites.
    if training.get("output_dir"):
        config["output_dir"] = training["output_dir"]
    if resume_ckpt:
        training["resume_from_checkpoint"] = resume_ckpt
        print(f"Using seed={seed}; resume_from_checkpoint={resume_ckpt}")
    else:
        print(f"Using seed={seed}")

    baseline_class = load_class(BASELINE_REGISTRY[args.baseline])
    baseline = baseline_class(config)

    if args.eval_only:
        if args.baseline not in EVAL_ONLY_BASELINES:
            raise SystemExit(
                f"ERROR: --eval_only is not supported for baseline '{args.baseline}'.\n"
                f"  Supported: {sorted(EVAL_ONLY_BASELINES)}"
            )
        if not hasattr(baseline, "run_eval_only"):
            raise SystemExit(
                f"ERROR: {baseline_class.__name__} has no run_eval_only method."
            )
        ckpt = args.checkpoint_path or resume_ckpt
        if not ckpt:
            raise SystemExit(
                f"ERROR: no checkpoint-* directory found under "
                f"{training.get('output_dir')}\n"
                "Nothing to evaluate."
            )
        _check_clobber(baseline.output_dir, args.overwrite)
        print(f"evaluate: scoring {ckpt} on held_out (no training)")
        val_metrics, test_metrics = baseline.run_eval_only(ckpt)
    else:
        val_metrics, test_metrics = baseline.run()

    print(f"Val: {val_metrics}")
    print(f"Test: {test_metrics}")

if __name__ == "__main__":
    main()
