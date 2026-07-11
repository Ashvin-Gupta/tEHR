import argparse
import importlib

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

def load_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=BASELINE_REGISTRY.keys())
    parser.add_argument("--config_filepath", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit Trainer checkpoint to resume from (overrides resume_from_last)",
    )
    args = parser.parse_args()

    config = load_config(args.config_filepath)

    # Align top-level output_dir with training.output_dir when only the latter is set.
    training = config.setdefault("training", {})
    if not config.get("output_dir") and training.get("output_dir"):
        config["output_dir"] = training["output_dir"]
    if training.get("output_dir") is None and config.get("output_dir"):
        training["output_dir"] = config["output_dir"]

    seed, resume_ckpt = prepare_run_reproducibility(
        config,
        cli_seed=args.seed,
        cli_checkpoint=args.checkpoint_path,
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
    val_metrics, test_metrics = baseline.run()
    print(f"Val: {val_metrics}")
    print(f"Test: {test_metrics}")

if __name__ == "__main__":
    main()
