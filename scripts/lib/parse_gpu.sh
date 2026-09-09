# Parse sbatch args: <dataset> <experiment> [model] [task] [evaluate]
# Requires KNOWN_* arrays from ``python -m src.experiment_map emit_bash`` (common.sh).

DATASET="${1:-}"
EXPERIMENT="${2:-}"
shift 2 || true

if [[ -z "${DATASET}" || -z "${EXPERIMENT}" ]]; then
    echo "Usage: sbatch $0 <dataset> <experiment> [model] [task] [evaluate]"
    echo "  datasets: ${KNOWN_DATASETS[*]}"
    echo "  models:   ${KNOWN_MODELS[*]}"
    echo "  mimic tasks:    ${MIMIC_TASKS[*]}"
    echo "  cprd tasks:     ${CPRD_TASKS[*]}"
    echo "  ehrshot tasks:  ${EHRSHOT_TASKS[*]}"
    exit 1
fi

if ! is_in "${DATASET}" "${KNOWN_DATASETS[@]}"; then
    echo "ERROR: unknown dataset '${DATASET}'. Expected: ${KNOWN_DATASETS[*]}"
    exit 1
fi

if ! python -c "from src.experiment_map import dataset_allowed; import sys; sys.exit(0 if dataset_allowed('${EXPERIMENT}', '${DATASET}') else 1)"; then
    echo "ERROR: experiment '${EXPERIMENT}' cannot run on dataset '${DATASET}'."
    exit 1
fi

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
fi
if [[ -n "${SEED:-}" ]]; then
    PIPELINE_EXTRA_ARGS+=(--seed "${SEED}")
    echo "SEED=${SEED} → writes under seed_${SEED}/"
fi

MODEL=""
TASK=""
EVAL_ONLY=""
KNOWN_TASKS=("${MIMIC_TASKS[@]}")
if [[ "${DATASET}" == "cprd" ]]; then
    KNOWN_TASKS=("${CPRD_TASKS[@]}")
elif [[ "${DATASET}" == "ehrshot" ]]; then
    KNOWN_TASKS=("${EHRSHOT_TASKS[@]}")
fi

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

case "${EVAL_ONLY:-}" in
    1|true|yes|on) EVAL_ONLY=1 ;;
    *)             EVAL_ONLY="" ;;
esac

if [[ -z "${TASK}" ]] && ! is_in "${EXPERIMENT}" "${TASK_OPTIONAL_EXPERIMENTS[@]}"; then
    if [[ "${DATASET}" == "cprd" ]]; then
        TASK="6m"
    elif [[ "${DATASET}" == "ehrshot" ]]; then
        TASK="icu_admission"
    else
        TASK="icu_mortality"
    fi
fi

if [[ -z "${NPROC:-}" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        NPROC="${SLURM_GPUS_ON_NODE##*:}"
    elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
        IFS=',' read -r -a _gpu_arr <<< "${SLURM_JOB_GPUS}"
        NPROC="${#_gpu_arr[@]}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a _gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
        NPROC="${#_gpu_arr[@]}"
    else
        NPROC="${DEFAULT_NPROC:-1}"
    fi
fi
if ! [[ "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARNING: could not parse GPU count ('${NPROC}'); defaulting to ${DEFAULT_NPROC:-1}"
    NPROC="${DEFAULT_NPROC:-1}"
fi

TAG="${DATASET}-${EXPERIMENT}"
[[ -n "${MODEL}" ]] && TAG="${TAG}-${MODEL}"
[[ -n "${TASK}" ]] && TAG="${TAG}-${TASK}"
RESOLVED_CFG="${REPO_ROOT}/configs/_resolved/${TAG}.yaml"
export RESOLVED_CFG

BUILD_ARGS=(--dataset "${DATASET}" --experiment "${EXPERIMENT}" --out "${RESOLVED_CFG}")
[[ -n "${MODEL}" ]] && BUILD_ARGS+=(--model "${MODEL}")
[[ -n "${TASK}" ]] && BUILD_ARGS+=(--task "${TASK}")

python -m src.training.config_utils build "${BUILD_ARGS[@]}"
echo "Resolved config → ${RESOLVED_CFG}"
echo "Using nproc_per_node=${NPROC}"
if [[ -n "${TEHR_STAGE1_TAG:-}" ]]; then
    echo "TEHR_STAGE1_TAG=${TEHR_STAGE1_TAG}"
fi
