"""
Single source of truth for experiment names and how they run.

Add a new experiment by:
  1. creating ``configs/experiments/<name>.yaml``
  2. adding a ``Spec`` here (runner class, DDP / one-GPU, eval, …)
  3. implementing ``run()`` and, if ``supports_eval``, ``run_eval_only()`` on that class

Shell arrays are emitted by ``python -m src.experiment_map emit_bash``.
"""
from __future__ import annotations

from dataclasses import dataclass


DATASETS = ("mimic", "cprd")
MODELS = ("mistral", "qwen0.6", "qwen8", "medgemma")
# Checkpoints that are not interchangeable with the causal-LM training stack.
MODEL_EXPERIMENTS = {
    "medgemma": ("zero_shot",),
}
MIMIC_TASKS = (
    "icu_mortality",
    "hospital_mortality",
    "hospital_readmission",
    "icu_readmission",
)
CPRD_TASKS = ("1m", "3m", "6m", "12m")
DEFAULT_TASK = {
    "mimic": "icu_mortality",
    "cprd": "6m",
}

CLASSIFIER_RESULTS = ("test_results.json", "combined_results.json", "held_out_results.json")
SPLIT_RESULTS = ("held_out_results.json", "tuning_results.json", "test_results.json", "summary.json")
SFT_RESULTS = ("held_out_results.json", "summary.json")
ENCODER_RESULTS = ("test_results.json", "validation_results.json", "summary.json")


@dataclass(frozen=True)
class Spec:
    runner: str
    ddp: bool = False
    one_gpu: bool = False
    needs_model: bool = False
    mimic_only: bool = False
    allowed_datasets: tuple[str, ...] | None = None
    supports_eval: bool = True
    task_optional: bool = False
    instruct: bool = False
    clear_pretrained: bool = False
    uses_stage1_ckpt: bool = False
    stage1_experiment: str | None = None
    allowed_models: tuple[str, ...] | None = None
    eval_result_files: tuple[str, ...] = SPLIT_RESULTS
    eval_error: str | None = None


_TEHR_CLF = "src.training.classification_trainer.TehrClassifierTrainer"
_ENCODER = "src.baselines.encoder.EncoderBaseline"

SPECS: dict[str, Spec] = {
    "tehr_stage1": Spec(
        runner="src.training.pretrainer_trainer.TehrStage1Trainer",
        ddp=True,
        needs_model=True,
        supports_eval=False,
        task_optional=True,
        clear_pretrained=True,
        eval_error=(
            "tehr_stage1 is continued pretraining; it has no classification "
            "checkpoint to score. Train tehr_stage2 and evaluate that."
        ),
    ),
    "tehr_stage1_v2": Spec(
        runner="src.training.pretrainer_trainer.TehrStage1Trainer",
        ddp=True,
        needs_model=True,
        supports_eval=False,
        task_optional=True,
        clear_pretrained=True,
        eval_error=(
            "tehr_stage1_v2 is continued pretraining; it has no classification "
            "checkpoint to score. Point stage 2 at this run's final_model "
            "explicitly when you want to score it."
        ),
    ),
    "tehr_stage2": Spec(
        runner=_TEHR_CLF,
        ddp=True,
        needs_model=True,
        uses_stage1_ckpt=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "sft": Spec(
        runner="src.baselines.sft.SFTBaseline",
        ddp=True,
        needs_model=True,
        instruct=True,
        eval_result_files=SFT_RESULTS,
    ),
    "zero_shot": Spec(
        runner="src.baselines.zero_shot.ZeroShotBaseline",
        needs_model=True,
        instruct=True,
        supports_eval=False,
        eval_error=(
            "zero_shot is inference-only and does not save a checkpoint. "
            "Run it without 'evaluate'."
        ),
    ),
    "xgboost_bow": Spec(
        runner="src.baselines.xgboost_bow.XGBoostBOWBaseline",
    ),
    "clmbr": Spec(
        runner="src.baselines.clmbr.CLMBRBaseline",
        one_gpu=True,
        mimic_only=True,
    ),
    "clmbr_ft": Spec(
        runner="src.baselines.clmbr_ft.CLMBRFineTuneBaseline",
        one_gpu=True,
        mimic_only=True,
    ),
    "clinicalbert": Spec(
        runner="src.baselines.clinicalbert.ClinicalBERTBaseline",
        one_gpu=True,
    ),
    "bioclinical_modernbert": Spec(
        runner="src.baselines.bioclinical_modernbert.BioClinicalModernBERTBaseline",
        ddp=True,
    ),
    "encoder": Spec(
        runner=_ENCODER,
        eval_result_files=ENCODER_RESULTS,
    ),
    "encoder_ctx2048": Spec(
        runner=_ENCODER,
        eval_result_files=ENCODER_RESULTS,
    ),
    "decoder": Spec(
        runner="src.baselines.decoder.DecoderBaseline",
        ddp=True,
        eval_result_files=ENCODER_RESULTS,
    ),
    "encoder_frozen": Spec(
        runner=_ENCODER,
        eval_result_files=ENCODER_RESULTS,
    ),
    "encoder_supervised": Spec(
        runner=_ENCODER,
        eval_result_files=ENCODER_RESULTS,
    ),
    "encoder_phrase_init": Spec(
        runner=_ENCODER,
        eval_result_files=ENCODER_RESULTS,
    ),
    "text_embedding": Spec(
        runner="src.baselines.text_embedding.TextEmbeddingBaseline",
    ),
    "llm_phrase": Spec(
        runner="src.baselines.llm_phrase.LLMPhraseTrainer",
        ddp=True,
        needs_model=True,
        uses_stage1_ckpt=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "ablation_supervised_only": Spec(
        runner=_TEHR_CLF,
        ddp=True,
        needs_model=True,
        allowed_datasets=("mimic",),
        clear_pretrained=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "ablation_cpt_frozen": Spec(
        runner=_TEHR_CLF,
        ddp=True,
        needs_model=True,
        allowed_datasets=("mimic",),
        uses_stage1_ckpt=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "ablation_nopretrain_frozen": Spec(
        runner="src.baselines.ablation_nopretrain_frozen.AblationNopretrainFrozenBaseline",
        one_gpu=True,
        needs_model=True,
        allowed_datasets=("mimic",),
        clear_pretrained=True,
    ),
    "ablation_hash": Spec(
        runner=_TEHR_CLF,
        ddp=True,
        needs_model=True,
        allowed_datasets=("mimic",),
        clear_pretrained=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "ablation_scrambled": Spec(
        runner=_TEHR_CLF,
        ddp=True,
        needs_model=True,
        allowed_datasets=("mimic",),
        clear_pretrained=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "ablation_no_time": Spec(
        runner=_TEHR_CLF,
        ddp=True,
        needs_model=True,
        uses_stage1_ckpt=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "ablation_no_value": Spec(
        runner=_TEHR_CLF,
        ddp=True,
        needs_model=True,
        uses_stage1_ckpt=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "ablation_max_events": Spec(
        runner=_TEHR_CLF,
        ddp=True,
        needs_model=True,
        uses_stage1_ckpt=True,
        eval_result_files=CLASSIFIER_RESULTS,
    ),
    "event_token_stage1": Spec(
        runner="src.training.event_token_trainer.EventTokenStage1Trainer",
        ddp=True,
        needs_model=True,
        supports_eval=False,
        task_optional=True,
        clear_pretrained=True,
        allowed_models=("mistral",),
        eval_error=(
            "event_token_stage1 is continued pretraining; it has no "
            "classification checkpoint to score. Train event_token_stage2 "
            "and evaluate that."
        ),
    ),
    "event_token_stage2": Spec(
        runner="src.training.event_token_trainer.EventTokenStage2Trainer",
        ddp=True,
        needs_model=True,
        uses_stage1_ckpt=True,
        stage1_experiment="event_token_stage1",
        allowed_models=("mistral",),
        eval_result_files=CLASSIFIER_RESULTS,
    ),
}


def get_spec(name: str) -> Spec:
    try:
        return SPECS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown experiment '{name}'. Known: {', '.join(sorted(SPECS))}"
        ) from exc


def dataset_allowed(experiment: str, dataset: str) -> bool:
    spec = get_spec(experiment)
    if spec.allowed_datasets is not None:
        return dataset in spec.allowed_datasets
    if spec.mimic_only:
        return dataset == "mimic"
    return True


def names() -> tuple[str, ...]:
    return tuple(SPECS)


def _bash_array(name: str, values: tuple[str, ...] | list[str]) -> str:
    return f"{name}=({' '.join(values)})"


def emit_bash() -> str:
    """Shell arrays for ``scripts/lib/dispatch.sh`` / parse scripts."""
    lines = [
        _bash_array("KNOWN_DATASETS", DATASETS),
        _bash_array("KNOWN_MODELS", MODELS),
        _bash_array("MIMIC_TASKS", MIMIC_TASKS),
        _bash_array("CPRD_TASKS", CPRD_TASKS),
        _bash_array("ALL_EXPERIMENTS", list(SPECS)),
        _bash_array("LLM_DDP_EXPERIMENTS", [n for n, s in SPECS.items() if s.ddp]),
        _bash_array("ONE_GPU_EXPERIMENTS", [n for n, s in SPECS.items() if s.one_gpu]),
        _bash_array("NEEDS_MODEL", [n for n, s in SPECS.items() if s.needs_model]),
        _bash_array("MIMIC_ONLY_EXPERIMENTS", [n for n, s in SPECS.items() if s.mimic_only]),
        _bash_array("EVAL_SUPPORTED", [n for n, s in SPECS.items() if s.supports_eval]),
        _bash_array("TASK_OPTIONAL_EXPERIMENTS", [n for n, s in SPECS.items() if s.task_optional]),
    ]
    return "\n".join(lines) + "\n"


def _cli() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "emit_bash":
        sys.stdout.write(emit_bash())
        return
    raise SystemExit("Usage: python -m src.experiment_map emit_bash")


if __name__ == "__main__":
    _cli()
