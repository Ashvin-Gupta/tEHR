#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=mimic-run
#
# =============================================================================
# MIMIC experiment dispatcher
# =============================================================================
#
# Usage:
#   sbatch src/submissions/mimic/run.sh <experiment> [model] [task]
#
# Args after <experiment> can be in any order. Known values:
#   models : mistral | medgemma | qwen0.6 | qwen8
#   tasks  : icu_mortality | hospital_mortality | hospital_readmission | icu_readmission
#            (default: icu_mortality for prediction experiments)
#   flags  : evaluate   # score an existing checkpoint instead of training
#
# Experiments (LLM — require model):
#   pretrain                    # no task needed
#   classifier                  # Exp 1
#   prompt                      # Exp 2
#   sft                         # Exp 3
#   ablation_no_pretrain_lora   # 5a, no pretraining and just doing sft with LORA and head
#   ablation_frozen_pretrained  # 5b, pretrained LLM but frozen and only doing sft on head
#   ablation_frozen_nopretrain  # 5c, no pretraining and frozen LLM, only doing sft on head
#   surface_raw                 # lexical ablation: raw codes (no CPT, like 5a)
#   surface_scramble_hash       # lexical ablation: Concept_<hash> titles
#   surface_scramble_shuffle    # lexical ablation: shuffled real titles
#   explain                     # post-hoc grounded explanations, base LM (needs GPU)
#
# Experiments (no model):
#   xgboost | clinicalbert | encoder | encoder_frozen_head | encoder_supervised | text_embedding
#
# Examples:
#   sbatch src/submissions/mimic/run.sh classifier mistral icu_mortality
#   sbatch src/submissions/mimic/run.sh surface_raw mistral icu_mortality
#   sbatch src/submissions/mimic/run.sh sft medgemma hospital_readmission
#   sbatch src/submissions/mimic/run.sh xgboost hospital_mortality
#   sbatch src/submissions/mimic/run.sh prompt mistral hospital_readmission
#   sbatch --gres=gpu:2 src/submissions/mimic/run.sh pretrain mistral
#   sbatch --gres=gpu:1 src/submissions/mimic/run.sh explain mistral icu_mortality
#   # NPROC defaults to the number of GPUs SLURM allocated; override with NPROC=2
#
# Recovering metrics from a job that ran out of wall clock: append `evaluate` to
# the exact command you used to train. It scores the run's newest checkpoint on
# the test set only and writes the usual results/test_results.json.
#   sbatch src/submissions/mimic/run.sh classifier mistral hospital_readmission evaluate
#   sbatch src/submissions/mimic/run.sh sft mistral hospital_mortality evaluate
#   sbatch src/submissions/mimic/run.sh ablation_frozen_nopretrain mistral hospital_mortality evaluate
# Supported for: classifier, ablation_no_pretrain_lora, ablation_frozen_pretrained,
# ablation_frozen_nopretrain, surface_raw, surface_scramble_hash, surface_scramble_shuffle,
# sft.
#
# Optional env:
#   HF_TOKEN=...   (required for medgemma)
#   NPROC=2        (torchrun GPU count)
#   RESUME_FROM_LAST=true|false  (override training.resume_from_last)
#   For explain: RESULTS_JSON CHECKPOINT OUT_DIR SEED SELECT_ONLY=true
#   For evaluate: SEED CHECKPOINT CKPT_SELECT=latest|best OVERWRITE=true EVAL_WANDB=true
# =============================================================================

set -euo pipefail

EXPERIMENT="${1:-}"
shift || true

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Usage: sbatch $0 <experiment> [model] [task]"
    echo "  models: mistral | medgemma | qwen0.6 | qwen8"
    echo "  tasks:  icu_mortality | hospital_mortality | hospital_readmission | icu_readmission"
    exit 1
fi

# Optional CLI overrides forwarded to pipeline scripts.
PIPELINE_EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_LAST:-}" ]]; then
    case "${RESUME_FROM_LAST,,}" in
        1|true|yes|on)  PIPELINE_EXTRA_ARGS+=(--resume_from_last) ;;
        0|false|no|off) PIPELINE_EXTRA_ARGS+=(--no-resume_from_last) ;;
        *)
            echo "ERROR: RESUME_FROM_LAST must be true/false (got '${RESUME_FROM_LAST}')"
            exit 1
            ;;
    esac
    echo "RESUME_FROM_LAST=${RESUME_FROM_LAST} → ${PIPELINE_EXTRA_ARGS[*]}"
fi

KNOWN_MODELS=("mistral" "medgemma" "qwen0.6" "qwen8")
KNOWN_TASKS=("icu_mortality" "hospital_mortality" "hospital_readmission" "icu_readmission")
MODEL=""
TASK=""
EVAL_ONLY=""

is_in() {
    local needle="$1"; shift
    local x
    for x in "$@"; do
        [[ "$x" == "$needle" ]] && return 0
    done
    return 1
}

for arg in "$@"; do
    if is_in "$arg" "${KNOWN_MODELS[@]}"; then
        MODEL="$arg"
    elif is_in "$arg" "${KNOWN_TASKS[@]}"; then
        TASK="$arg"
    elif [[ "$arg" == "evaluate" || "$arg" == "eval" ]]; then
        EVAL_ONLY=1
    else
        echo "ERROR: unknown argument '$arg'"
        echo "  models: ${KNOWN_MODELS[*]}"
        echo "  tasks:  ${KNOWN_TASKS[*]}"
        echo "  flags:  evaluate"
        exit 1
    fi
done

# EVAL_ONLY=true also works, for callers that pass env instead of args.
case "${EVAL_ONLY:-}" in
    1|true|yes|on) EVAL_ONLY=1 ;;
    *)             EVAL_ONLY="" ;;
esac

# Defaults
if [[ -z "${TASK}" && "${EXPERIMENT}" != "pretrain" ]]; then
    TASK="icu_mortality"
fi

# SLURM copies this script to /var/spool/.../slurm_script, so BASH_SOURCE is
# unreliable — always source common.sh from the repo path.
BASE_DIR="/home/ag619/TextCancEHR2"
# shellcheck source=common.sh
source "${BASE_DIR}/src/submissions/mimic/common.sh"

echo "============================================================"
echo " MIMIC run"
echo "   experiment : ${EXPERIMENT}"
if [[ -n "${EVAL_ONLY}" ]]; then MODE_LABEL="evaluate (no training)"; else MODE_LABEL="train"; fi
echo "   mode       : ${MODE_LABEL}"
echo "   model      : ${MODEL:-<none>}"
echo "   task       : ${TASK:-<none>}"
echo "   host       : $(hostname)"
echo "   MASTER_PORT: ${MASTER_PORT}"
echo "============================================================"


RESOLVED_DIR="${BASE_DIR}/src/configs/mimic/_resolved"
mkdir -p "${RESOLVED_DIR}"
TAG="${EXPERIMENT}"
[[ -n "${MODEL}" ]] && TAG="${TAG}-${MODEL}"
[[ -n "${TASK}" ]] && TAG="${TAG}-${TASK}"
RESOLVED_CFG="${RESOLVED_DIR}/${TAG}.yaml"

BUILD_ARGS=(--experiment "${EXPERIMENT}" --out "${RESOLVED_CFG}")
[[ -n "${MODEL}" ]] && BUILD_ARGS+=(--model "${MODEL}")
[[ -n "${TASK}" ]] && BUILD_ARGS+=(--task "${TASK}")

python -m src.training.config_utils build "${BUILD_ARGS[@]}"
echo "Resolved config → ${RESOLVED_CFG}"

if [[ -z "${NPROC:-}" ]]; then
    # Match torchrun workers to GPUs actually allocated by SLURM.
    # (Old default of 2 for pretrain broke when --gres=gpu:1.)
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        # Forms: "2" or "gpu:2" or "a100:2"
        NPROC="${SLURM_GPUS_ON_NODE##*:}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
        NPROC="${#_gpu_arr[@]}"
    else
        NPROC=1
    fi
fi
# Guard against empty/non-numeric
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARNING: could not parse GPU count ('${NPROC}'); defaulting to 1"
    NPROC=1
fi
echo "Using nproc_per_node=${NPROC} (SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"

run_torch() {
    local script="$1"
    shift
    torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
        "${script}" --config_filepath "${RESOLVED_CFG}" \
        "${PIPELINE_EXTRA_ARGS[@]}" "$@"
}

# Score an existing checkpoint instead of training. Deliberately does not forward
# PIPELINE_EXTRA_ARGS: --resume_from_last has no meaning here.
run_torch_eval() {
    local eval_args=()
    [[ -n "${SEED:-}" ]] && eval_args+=(--seed "${SEED}")
    [[ -n "${CHECKPOINT:-}" ]] && eval_args+=(--checkpoint_path "${CHECKPOINT}")
    [[ -n "${CKPT_SELECT:-}" ]] && eval_args+=(--checkpoint_selection "${CKPT_SELECT}")
    case "${OVERWRITE:-}" in
        1|true|yes|on) eval_args+=(--overwrite) ;;
    esac
    case "${EVAL_WANDB:-}" in
        1|true|yes|on) eval_args+=(--wandb) ;;
    esac

    echo "evaluate: scoring an existing checkpoint (no training)"
    echo "  extra args: ${eval_args[*]:-<none>}"
    torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
        src/pipelines/evaluate_checkpoint.py \
        --config_filepath "${RESOLVED_CFG}" "${eval_args[@]}"
}

# Experiments whose checkpoints evaluate understands.
EVAL_SUPPORTED=(
    classifier ablation_no_pretrain_lora ablation_frozen_pretrained
    ablation_frozen_nopretrain
    surface_raw surface_scramble_hash surface_scramble_shuffle
    sft
)
if [[ -n "${EVAL_ONLY}" ]] && ! is_in "${EXPERIMENT}" "${EVAL_SUPPORTED[@]}"; then
    echo "ERROR: 'evaluate' is not supported for experiment '${EXPERIMENT}'."
    echo "  Supported: ${EVAL_SUPPORTED[*]}"
    echo "  classifier / LoRA ablations / surface_* / sft use evaluate_checkpoint.py;"
    echo "  ablation_frozen_nopretrain uses the frozen_llm_linear baseline path."
    echo "  prompt / the other baselines save a different format."
    exit 1
fi

require_model() {
    if [[ -z "${MODEL}" ]]; then
        echo "ERROR: ${EXPERIMENT} requires a model (${KNOWN_MODELS[*]})"
        exit 1
    fi
}

case "${EXPERIMENT}" in
    pretrain)
        require_model
        run_torch src/pipelines/llm_pretrain.py
        ;;
    classifier|ablation_no_pretrain_lora|ablation_frozen_pretrained|surface_raw|surface_scramble_hash|surface_scramble_shuffle)
        require_model
        if [[ -n "${EVAL_ONLY}" ]]; then
            run_torch_eval
        else
            run_torch src/pipelines/finetune_llm_classifier.py
        fi
        ;;
    ablation_frozen_nopretrain)
        require_model
        # Plain python (not torchrun). Pin to one GPU so HF Trainer does not
        # wrap the 4-bit model in DataParallel across all allocated GPUs.
        _first_gpu="${CUDA_VISIBLE_DEVICES:-}"
        _first_gpu="${_first_gpu%%,*}"
        export CUDA_VISIBLE_DEVICES="${_first_gpu:-0}"
        echo "ablation_frozen_nopretrain: pinning CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        BASELINE_ARGS=(
            --baseline frozen_llm_linear
            --config_filepath "${RESOLVED_CFG}"
        )
        if [[ -n "${EVAL_ONLY}" ]]; then
            BASELINE_ARGS+=(--eval_only)
            [[ -n "${SEED:-}" ]] && BASELINE_ARGS+=(--seed "${SEED}")
            [[ -n "${CHECKPOINT:-}" ]] && BASELINE_ARGS+=(--checkpoint_path "${CHECKPOINT}")
            case "${OVERWRITE:-}" in
                1|true|yes|on) BASELINE_ARGS+=(--overwrite) ;;
            esac
            echo "evaluate: scoring newest checkpoint on held_out (no training)"
            echo "  args: ${BASELINE_ARGS[*]}"
            python -m src.pipelines.run_baseline "${BASELINE_ARGS[@]}"
        else
            python -m src.pipelines.run_baseline \
                "${BASELINE_ARGS[@]}" \
                "${PIPELINE_EXTRA_ARGS[@]}"
        fi
        ;;
    sft)
        require_model
        if [[ -n "${EVAL_ONLY}" ]]; then
            run_torch_eval
        else
            run_torch src/pipelines/finetune_llm_sft_classifier.py
        fi
        ;;
    prompt)
        require_model
        python -m src.pipelines.run_baseline \
            --baseline pure_llm \
            --config_filepath "${RESOLVED_CFG}" \
            "${PIPELINE_EXTRA_ARGS[@]}"
        ;;
    xgboost)
        python -m src.pipelines.run_baseline \
            --baseline xgboost_bow \
            --config_filepath "${RESOLVED_CFG}" \
            "${PIPELINE_EXTRA_ARGS[@]}"
        ;;
    clinicalbert)
        # Plain python (not torchrun). Pin to one GPU on multi-GPU allocations.
        _first_gpu="${CUDA_VISIBLE_DEVICES:-}"
        _first_gpu="${_first_gpu%%,*}"
        export CUDA_VISIBLE_DEVICES="${_first_gpu:-0}"
        echo "clinicalbert: pinning CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        python -m src.pipelines.run_baseline \
            --baseline clinicalbert \
            --config_filepath "${RESOLVED_CFG}" \
            "${PIPELINE_EXTRA_ARGS[@]}"
        ;;
    encoder|encoder_frozen_head|encoder_supervised)
        python -m src.pipelines.run_baseline \
            --baseline encoder_transformer \
            --config_filepath "${RESOLVED_CFG}" \
            "${PIPELINE_EXTRA_ARGS[@]}"
        ;;
    text_embedding)
        # Phrase-level cache (last-token pool). Old token_embeddings_*.pt is incompatible.
        EMBEDDINGS_PATH="/home/ag619/experiments/Results/mimic-baselines/text_embedding_transformer/phrase_embeddings_qwen3_0.6B.pt"
        if [[ ! -f "${EMBEDDINGS_PATH}" ]]; then
            echo "Precomputing phrase embeddings (last-token pool)..."
            python -m src.pipelines.precompute_token_embeddings \
                --vocab_filepath  "/home/ag619/MIMIC-IV-tokenisation/vocab.csv" \
                --lookup_dir      "/home/ag619/MIMIC_data/hosp" \
                --data_dir        "/home/ag619/MIMIC-IV-tokenisation" \
                --model_name      "Qwen/Qwen3-Embedding-0.6B" \
                --output_path     "${EMBEDDINGS_PATH}" \
                --batch_size      256 \
                --max_length      128 \
                --device          cuda
        else
            echo "Phrase embeddings already exist at ${EMBEDDINGS_PATH}"
        fi
        python -m src.pipelines.run_baseline \
            --baseline text_embedding_transformer \
            --config_filepath "${RESOLVED_CFG}" \
            "${PIPELINE_EXTRA_ARGS[@]}"
        ;;
    explain)
        require_model
        _first_gpu="${CUDA_VISIBLE_DEVICES:-}"
        _first_gpu="${_first_gpu%%,*}"
        export CUDA_VISIBLE_DEVICES="${_first_gpu:-0}"
        echo "explain: pinning CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

        SEED="${SEED:-42}"
        RESULTS_ROOT="/home/ag619/experiments/Results"
        DEFAULT_RESULTS="${RESULTS_ROOT}/mimic-classifier-${MODEL}-${TASK}-rerun/seed_${SEED}/results/test_results.json"
        if [[ ! -f "${DEFAULT_RESULTS}" ]]; then
            DEFAULT_RESULTS="${RESULTS_ROOT}/mimic-classifier-${MODEL}-${TASK}/seed_${SEED}/results/test_results.json"
        fi
        RESULTS_JSON="${RESULTS_JSON:-${DEFAULT_RESULTS}}"
        DEFAULT_CKPT="${RESULTS_ROOT}/mimic-classifier-${MODEL}-${TASK}-rerun/seed_${SEED}/final_model"
        if [[ ! -d "${DEFAULT_CKPT}" ]]; then
            DEFAULT_CKPT="${RESULTS_ROOT}/mimic-classifier-${MODEL}-${TASK}/seed_${SEED}/final_model"
        fi
        CHECKPOINT="${CHECKPOINT:-${DEFAULT_CKPT}}"
        OUT_DIR="${OUT_DIR:-${BASE_DIR}/analysis_out/explain-${MODEL}-${TASK}}"
        mkdir -p "${OUT_DIR}"

        EXPLAIN_ARGS=(
            --config_filepath "${RESOLVED_CFG}"
            --out_dir "${OUT_DIR}"
            --seed "${SEED}"
            --split held_out
            --n_per_cell 5
        )
        if [[ -n "${RESULTS_JSON}" ]]; then
            EXPLAIN_ARGS+=(--results_json "${RESULTS_JSON}")
        fi
        if [[ -n "${CHECKPOINT}" && -e "${CHECKPOINT}" ]]; then
            EXPLAIN_ARGS+=(--checkpoint "${CHECKPOINT}")
        fi
        if [[ "${SELECT_ONLY:-}" == "true" || "${SELECT_ONLY:-}" == "1" ]]; then
            EXPLAIN_ARGS+=(--select_only)
        fi

        echo "RESULTS_JSON → ${RESULTS_JSON}"
        echo "CHECKPOINT   → ${CHECKPOINT}"
        echo "OUT_DIR      → ${OUT_DIR}"
        python -m src.analysis.prediction_explain "${EXPLAIN_ARGS[@]}"
        ;;
    *)
        echo "ERROR: unknown experiment '${EXPERIMENT}'"
        echo "Known: pretrain classifier prompt sft ablation_no_pretrain_lora"
        echo "       ablation_frozen_pretrained ablation_frozen_nopretrain"
        echo "       surface_raw surface_scramble_hash surface_scramble_shuffle"
        echo "       explain"
        echo "       xgboost clinicalbert encoder encoder_frozen_head encoder_supervised text_embedding"
        exit 1
        ;;
esac

echo "Pipeline finished."
conda deactivate
