#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=isa-mimic-run
#
# =============================================================================
# Isambard MIMIC experiment dispatcher
# =============================================================================
#
# Same experiment matrix as src/submissions/mimic/run.sh, with Isambard paths
# remapped via:  python -m src.training.config_utils build --site isambard ...
#
# Default allocation: 4 GPUs. torchrun nproc matches allocated GPUs.
# Override GPUs at submit time (must use sbatch):
#   sbatch --gpus=1 src/submissions/isambard/run.sh ...
#   sbatch --gpus=4 src/submissions/isambard/run.sh ...
#
# Usage:
#   sbatch src/submissions/isambard/run.sh <experiment> [model] [task]
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
#   ablation_no_pretrain_lora   # 5a
#   ablation_frozen_pretrained  # 5b
#   ablation_frozen_nopretrain  # 5c
#
# Experiments (no model):
#   xgboost | encoder | text_embedding
#
# Examples:
#   sbatch src/submissions/isambard/run.sh classifier mistral icu_mortality
#   sbatch src/submissions/isambard/run.sh sft mistral hospital_readmission
#   sbatch src/submissions/isambard/run.sh pretrain mistral
#   sbatch --gpus=1 src/submissions/isambard/run.sh classifier mistral icu_mortality
#
# Optional env:
#   HF_TOKEN=...   (required for medgemma)
#   NPROC=2        (override torchrun GPU count; default = allocated GPUs)
# =============================================================================

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch, e.g.:"
    echo "  sbatch src/submissions/isambard/run.sh classifier mistral icu_mortality"
    echo "  sbatch --gpus=1 src/submissions/isambard/run.sh classifier mistral icu_mortality"
    echo "  sbatch --gpus=4 src/submissions/isambard/run.sh pretrain mistral"
    exit 1
fi

EXPERIMENT="${1:-}"
shift || true

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Usage: sbatch $0 <experiment> [model] [task]"
    echo "  models: mistral | medgemma"
    echo "  tasks:  icu_mortality | hospital_mortality | hospital_readmission"
    echo "  GPUs:   default 4; override with: sbatch --gpus=1|4 ..."
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
BASE_DIR="${HOME}/TextCancEHR2"
# shellcheck source=common.sh
source "${BASE_DIR}/src/submissions/isambard/common.sh"

echo "============================================================"
echo " Isambard MIMIC run"
echo "   experiment : ${EXPERIMENT}"
echo "   model      : ${MODEL:-<none>}"
echo "   task       : ${TASK:-<none>}"
echo "   site       : isambard"
echo "   host       : $(hostname)"
echo "   MASTER_PORT: ${MASTER_PORT}"
echo "============================================================"

RESOLVED_DIR="${BASE_DIR}/src/configs/mimic/_resolved_isambard"
mkdir -p "${RESOLVED_DIR}"
TAG="${EXPERIMENT}"
[[ -n "${MODEL}" ]] && TAG="${TAG}-${MODEL}"
[[ -n "${TASK}" ]] && TAG="${TAG}-${TASK}"
RESOLVED_CFG="${RESOLVED_DIR}/${TAG}.yaml"

BUILD_ARGS=(--experiment "${EXPERIMENT}" --site isambard --out "${RESOLVED_CFG}")
[[ -n "${MODEL}" ]] && BUILD_ARGS+=(--model "${MODEL}")
[[ -n "${TASK}" ]] && BUILD_ARGS+=(--task "${TASK}")

python -m src.training.config_utils build "${BUILD_ARGS[@]}"
echo "Resolved config → ${RESOLVED_CFG}"

# Match torchrun workers to GPUs SLURM actually allocated (--gpus=N / #SBATCH --gpus=N).
if [[ -z "${NPROC:-}" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        # Forms: "4" or "gpu:4" or "a100:4"
        NPROC="${SLURM_GPUS_ON_NODE##*:}"
    elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
        # Comma-separated GPU IDs, e.g. "0,1,2,3"
        IFS=',' read -r -a _gpu_arr <<< "${SLURM_JOB_GPUS}"
        NPROC="${#_gpu_arr[@]}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
        NPROC="${#_gpu_arr[@]}"
    else
        NPROC=4
    fi
fi
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARNING: could not parse GPU count ('${NPROC}'); defaulting to 4"
    NPROC=4
fi
echo "Using nproc_per_node=${NPROC} (SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}, SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"

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
        EMBEDDINGS_PATH="/projects/u6ps/TextCancEHR2/experiments/Results/mimic-baselines/text_embedding_transformer/token_embeddings_qwen3_0.6B.pt"
        if [[ ! -f "${EMBEDDINGS_PATH}" ]]; then
            echo "Precomputing token embeddings..."
            mkdir -p "$(dirname "${EMBEDDINGS_PATH}")"
            python -m src.pipelines.precompute_token_embeddings \
                --vocab_filepath  "${HOME}/MIMIC-IV_tokenisation/vocab.csv" \
                --lookup_dir      "${HOME}/MIMIC_data/hosp" \
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
