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
#   models : mistral | medgemma | qwen0.6 | qwen8
#   tasks  : icu_mortality | hospital_mortality | hospital_readmission | icu_readmission
#            (default: icu_mortality for prediction experiments)
#   flags  : evaluate   # score the newest checkpoint instead of training
#
# Experiments (LLM — require model):
#   pretrain                    # no task needed
#   classifier                  # Exp 1
#   prompt                      # Exp 2
#   sft                         # Exp 3
#   ablation_no_pretrain_lora   # 5a
#   ablation_frozen_pretrained  # 5b
#   ablation_frozen_nopretrain  # 5c
#   surface_raw                 # lexical ablation: raw codes (no CPT, like 5a)
#   surface_scramble_hash       # lexical ablation: Concept_<hash> titles
#   surface_scramble_shuffle    # lexical ablation: shuffled real titles
#   explain                     # post-hoc grounded explanations, base LM (needs GPU)
#
# Experiments (no model):
#   xgboost | clinicalbert | encoder | encoder_frozen_head | encoder_supervised | text_embedding
#
# Examples:
#   sbatch src/submissions/isambard/run.sh classifier mistral icu_mortality
#   sbatch src/submissions/isambard/run.sh surface_raw mistral icu_mortality
#   sbatch src/submissions/isambard/run.sh sft mistral hospital_readmission
#   sbatch src/submissions/isambard/run.sh pretrain mistral
#   sbatch --gpus=1 src/submissions/isambard/run.sh classifier mistral icu_mortality
#   sbatch --gpus=1 src/submissions/isambard/run.sh encoder_frozen_head icu_mortality
#   sbatch --gpus=1 src/submissions/isambard/run.sh explain mistral icu_mortality
#
# Recovering metrics from a job that ran out of wall clock: append `evaluate` to
# the exact command you used to train. It scores the newest checkpoint-* in the
# run directory (same choice resume_from_last makes) on the test set only, and
# writes the usual results/test_results.json — no checkpoint path needed.
#   sbatch src/submissions/isambard/run.sh sft mistral hospital_readmission evaluate
#   sbatch --gpus=4 src/submissions/isambard/run.sh classifier mistral hospital_readmission evaluate
#   sbatch --gpus=4 src/submissions/isambard/run.sh surface_raw mistral hospital_mortality evaluate
#   sbatch --gpus=1 src/submissions/isambard/run.sh ablation_frozen_nopretrain mistral hospital_mortality evaluate
# Supported for: classifier, ablation_no_pretrain_lora, ablation_frozen_pretrained,
# ablation_frozen_nopretrain, surface_raw, surface_scramble_hash, surface_scramble_shuffle,
# sft.
#
# Optional env:
#   HF_TOKEN=...   (required for medgemma)
#   NPROC=2        (override torchrun GPU count; default = allocated GPUs)
#   RESUME_FROM_LAST=true|false  (override training.resume_from_last),--export=ALL,RESUME_FROM_LAST=true
#   (or RESUME_FROM_LAST=true|false with sbatch run.sh)
#   For explain:
#     RESULTS_JSON=... CHECKPOINT=... OUT_DIR=... SEED=42
#     SELECT_ONLY=true   (pick 20 cases only, no generation)
#   For evaluate:
#     SEED=42            (which seed_<N> directory to score; classifier-family only)
#     CHECKPOINT=...     (score this exact checkpoint)
#     CKPT_SELECT=latest|best  (default latest: newest checkpoint-*)
#     OVERWRITE=true     (replace an existing test/held_out_results.json; refuses by default)
#     EVAL_WANDB=true    (log the evaluation to wandb; off by default)
# =============================================================================

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch, e.g.:"
    echo "  sbatch src/submissions/isambard/run.sh classifier mistral icu_mortality"
    echo "  sbatch --gpus=1 src/submissions/isambard/run.sh classifier mistral icu_mortality"
    echo "  sbatch --gpus=4 src/submissions/isambard/run.sh pretrain mistral"
    exit 1
fi

# `salloc` sets SLURM_JOB_ID but leaves the shell on the login node, where no
# GPUs are bound. Catch that here instead of failing deep inside Unsloth.
if [[ -z "${SLURM_GPUS_ON_NODE:-}${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: job ${SLURM_JOB_ID} has no GPUs visible to this shell (host $(hostname))."
    echo "Every experiment in this script needs a GPU. Options:"
    echo "  1. Step onto the node already allocated to this salloc session:"
    echo "       srun --pty bash --login"
    echo "  2. Request an interactive GPU node directly:"
    echo "       srun --gpus=1 --time=2:00:00 --mem=64G --pty bash --login"
    echo "  3. Submit as a batch job:"
    echo "       sbatch --gpus=1 src/submissions/isambard/run.sh explain mistral icu_mortality"
    echo "(CPU-only analyses live in src/submissions/isambard/run_cpu.sh)"
    exit 1
fi

EXPERIMENT="${1:-}"
shift || true

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Usage: sbatch $0 <experiment> [model] [task]"
    echo "  models: mistral | medgemma | qwen0.6 | qwen8"
    echo "  tasks:  icu_mortality | hospital_mortality | hospital_readmission | icu_readmission"
    echo "  GPUs:   default 4; override with: sbatch --gpus=1|4 ..."
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
BASE_DIR="${HOME}/TextCancEHR2"
# shellcheck source=common.sh
source "${BASE_DIR}/src/submissions/isambard/common.sh"

echo "============================================================"
echo " Isambard MIMIC run"
echo "   experiment : ${EXPERIMENT}"
if [[ -n "${EVAL_ONLY}" ]]; then MODE_LABEL="evaluate (no training)"; else MODE_LABEL="train"; fi
echo "   mode       : ${MODE_LABEL}"
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
        "${script}" --config_filepath "${RESOLVED_CFG}" \
        "${PIPELINE_EXTRA_ARGS[@]}" "$@"
}

# Score the newest checkpoint instead of training. Deliberately does not forward
# PIPELINE_EXTRA_ARGS: --resume_from_last has no meaning here, the newest
# checkpoint is always the one evaluated.
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

    echo "evaluate: scoring newest checkpoint (no training)"
    echo "  extra args: ${eval_args[*]:-<none>}"
    torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
        src/pipelines/evaluate_checkpoint.py \
        --config_filepath "${RESOLVED_CFG}" "${eval_args[@]}"
}

# Experiments whose checkpoints evaluate understands.
# classifier-family → evaluate_checkpoint.py (torchrun)
# ablation_frozen_nopretrain → run_baseline --eval_only (single GPU)
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
        EMBEDDINGS_PATH="/projects/u6ps/TextCancEHR2/experiments/Results/mimic-baselines/text_embedding_transformer/phrase_embeddings_qwen3_0.6B.pt"
        if [[ ! -f "${EMBEDDINGS_PATH}" ]]; then
            echo "Precomputing phrase embeddings (last-token pool)..."
            mkdir -p "$(dirname "${EMBEDDINGS_PATH}")"
            python -m src.pipelines.precompute_token_embeddings \
                --vocab_filepath  "${HOME}/MIMIC-IV_tokenisation/vocab.csv" \
                --lookup_dir      "${HOME}/MIMIC_data/hosp" \
                --data_dir        "${HOME}/MIMIC-IV_tokenisation" \
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
        # Single-GPU generation (no torchrun). Prefer 1 GPU at submit time:
        #   sbatch --gpus=1 ... run.sh explain mistral icu_mortality
        _first_gpu="${CUDA_VISIBLE_DEVICES:-}"
        _first_gpu="${_first_gpu%%,*}"
        export CUDA_VISIBLE_DEVICES="${_first_gpu:-0}"
        echo "explain: pinning CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

        SEED="${SEED:-42}"
        RESULTS_ROOT="/projects/u6ps/TextCancEHR2/experiments/Results"
        DEFAULT_RESULTS="${RESULTS_ROOT}/mimic-classifier-${MODEL}-${TASK}-rerun/seed_${SEED}/results/test_results.json"
        # Fall back to non-rerun path if present
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
