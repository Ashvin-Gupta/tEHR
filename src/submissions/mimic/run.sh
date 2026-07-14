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
#   models : mistral | medgemma
#   tasks  : icu_mortality | hospital_mortality | hospital_readmission
#            (default: icu_mortality for prediction experiments)
#
# Experiments (LLM — require model):
#   pretrain                    # no task needed
#   classifier                  # Exp 1
#   prompt                      # Exp 2
#   sft                         # Exp 3
#   ablation_no_pretrain_lora   # 5a, no pretraining and just doing sft with LORA and head
#   ablation_frozen_pretrained  # 5b, pretrained LLM but frozen and only doing sft on head
#   ablation_frozen_nopretrain  # 5c, no pretraining and frozen LLM, only doing sft on head
#
# Experiments (no model):
#   xgboost | encoder | text_embedding
#
# Examples:
#   sbatch src/submissions/mimic/run.sh classifier mistral icu_mortality
#   sbatch src/submissions/mimic/run.sh sft medgemma hospital_readmission
#   sbatch src/submissions/mimic/run.sh xgboost hospital_mortality
#   sbatch src/submissions/mimic/run.sh prompt mistral hospital_readmission
#   sbatch --gres=gpu:2 src/submissions/mimic/run.sh pretrain mistral
#   # NPROC defaults to the number of GPUs SLURM allocated; override with NPROC=2
#
# Optional env:
#   HF_TOKEN=...   (required for medgemma)
#   NPROC=2        (torchrun GPU count)
# =============================================================================

set -euo pipefail

EXPERIMENT="${1:-}"
shift || true

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Usage: sbatch $0 <experiment> [model] [task]"
    echo "  models: mistral | medgemma"
    echo "  tasks:  icu_mortality | hospital_mortality | hospital_readmission"
    exit 1
fi

KNOWN_MODELS=("mistral" "medgemma")
KNOWN_TASKS=("icu_mortality" "hospital_mortality" "hospital_readmission")

MODEL=""
TASK=""

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
    else
        echo "ERROR: unknown argument '$arg'"
        echo "  models: ${KNOWN_MODELS[*]}"
        echo "  tasks:  ${KNOWN_TASKS[*]}"
        exit 1
    fi
done

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
        "${script}" --config_filepath "${RESOLVED_CFG}" "$@"
}

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
    classifier|ablation_no_pretrain_lora|ablation_frozen_pretrained)
        require_model
        run_torch src/pipelines/finetune_llm_classifier.py
        ;;
    ablation_frozen_nopretrain)
        require_model
        python -m src.pipelines.run_baseline \
            --baseline frozen_llm_linear \
            --config_filepath "${RESOLVED_CFG}"
        ;;
    sft)
        require_model
        run_torch src/pipelines/finetune_llm_sft_classifier.py
        ;;
    prompt)
        require_model
        python -m src.pipelines.run_baseline \
            --baseline pure_llm \
            --config_filepath "${RESOLVED_CFG}"
        ;;
    xgboost)
        python -m src.pipelines.run_baseline \
            --baseline xgboost_bow \
            --config_filepath "${RESOLVED_CFG}"
        ;;
    encoder)
        python -m src.pipelines.run_baseline \
            --baseline encoder_transformer \
            --config_filepath "${RESOLVED_CFG}"
        ;;
    text_embedding)
        EMBEDDINGS_PATH="/home/ag619/experiments/Results/mimic-baselines/text_embedding_transformer/token_embeddings_qwen3_0.6B.pt"
        if [[ ! -f "${EMBEDDINGS_PATH}" ]]; then
            echo "Precomputing token embeddings..."
            python -m src.pipelines.precompute_token_embeddings \
                --vocab_filepath  "/home/ag619/MIMIC-IV-tokenisation/vocab.csv" \
                --lookup_dir      "/home/ag619/MIMIC_data/hosp" \
                --model_name      "Qwen/Qwen3-Embedding-0.6B" \
                --output_path     "${EMBEDDINGS_PATH}" \
                --batch_size      256 \
                --max_length      64 \
                --device          cuda
        else
            echo "Embeddings already exist at ${EMBEDDINGS_PATH}"
        fi
        python -m src.pipelines.run_baseline \
            --baseline text_embedding_transformer \
            --config_filepath "${RESOLVED_CFG}"
        ;;
    *)
        echo "ERROR: unknown experiment '${EXPERIMENT}'"
        echo "Known: pretrain classifier prompt sft ablation_no_pretrain_lora"
        echo "       ablation_frozen_pretrained ablation_frozen_nopretrain"
        echo "       xgboost encoder text_embedding"
        exit 1
        ;;
esac

echo "Pipeline finished."
conda deactivate
