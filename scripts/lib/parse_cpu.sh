# CPU dispatcher body. Sourced by scripts/site_*/run_cpu.sh after common.sh.

CPU_ANALYSIS=(plot_ablation plot_ablation_deltas bootstrap_ci seed_ci paired_compare)
CPU_TRAIN=(xgboost_bow)

ARG1="${1:-}"
ARG2="${2:-}"
# `shift 2` is a no-op (and fails) when only one arg is passed, e.g.
# `sbatch ... run_cpu.sh bootstrap_ci`. Shift one at a time.
[[ $# -ge 1 ]] && shift
[[ $# -ge 1 ]] && shift

if [[ -z "${ARG1}" ]]; then
    echo "Usage: sbatch $0 <dataset> xgboost_bow [task] [evaluate]"
    echo "   or: sbatch $0 plot_ablation|plot_ablation_deltas|bootstrap_ci|seed_ci|paired_compare [args via env]"
    exit 1
fi

if is_in "${ARG1}" "${CPU_ANALYSIS[@]}"; then
    EXPERIMENT="${ARG1}"
    DATASET="${ARG2:-}"
else
    DATASET="${ARG1}"
    EXPERIMENT="${ARG2}"
    if [[ -z "${EXPERIMENT}" ]]; then
        echo "ERROR: expected experiment after dataset"
        exit 1
    fi
fi

TASK=""
EVAL_ONLY=""
KNOWN_TASKS=("${MIMIC_TASKS[@]}")
if [[ "${DATASET}" == "cprd" ]]; then
    KNOWN_TASKS=("${CPRD_TASKS[@]}")
fi
for arg in "$@"; do
    if is_in "$arg" "${KNOWN_TASKS[@]}"; then
        TASK="$arg"
    elif [[ "$arg" == "evaluate" || "$arg" == "eval" ]]; then
        EVAL_ONLY=1
    else
        echo "ERROR: unknown argument '$arg'"
        exit 1
    fi
done

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/analysis_out}"
mkdir -p "${OUT_DIR}"

case "${EXPERIMENT}" in
    xgboost_bow)
        if [[ -z "${DATASET}" ]]; then
            echo "ERROR: xgboost_bow requires a dataset"
            exit 1
        fi
        if [[ -z "${TASK}" ]]; then
            if [[ "${DATASET}" == "cprd" ]]; then
                TASK="6m"
            else
                TASK="icu_mortality"
            fi
        fi
        TAG="${DATASET}-xgboost_bow-${TASK}"
        RESOLVED_CFG="${REPO_ROOT}/configs/_resolved/${TAG}.yaml"
        python -m src.training.config_utils build \
            --dataset "${DATASET}" --experiment xgboost_bow --task "${TASK}" \
            --out "${RESOLVED_CFG}"
        extra=()
        [[ -n "${SEED:-}" ]] && extra+=(--seed "${SEED}")
        if [[ -n "${EVAL_ONLY}" ]]; then
            extra+=(--eval_only)
            [[ -n "${CHECKPOINT:-}" ]] && extra+=(--checkpoint_path "${CHECKPOINT}")
            case "${OVERWRITE:-}" in
                1|true|yes|on) extra+=(--overwrite) ;;
            esac
        fi
        python -m src.pipelines.run \
            --config_filepath "${RESOLVED_CFG}" "${extra[@]}"
        ;;
    plot_ablation)
        CSV="${CSV:-${REPO_ROOT}/analysis_out/ablations/results.csv}"
        ABLATION_OUT="${ABLATION_OUT:-${REPO_ROOT}/analysis_out/ablations}"
        python -m src.analysis.plot_ablation_auroc \
            --csv "${CSV}" --out-dir "${ABLATION_OUT}" --style "${STYLE:-dot_1x4}"
        ;;
    plot_ablation_deltas)
        python -m src.analysis.plot_ablation_deltas \
            --csv "${CSV:-${REPO_ROOT}/analysis_out/bootstrap/bootstrap_ci.csv}" \
            --out-dir "${ABLATION_OUT:-${REPO_ROOT}/analysis_out/ablations}"
        ;;
    bootstrap_ci)
        BOOT_OUT="${BOOT_OUT:-${OUT_DIR}/bootstrap}"
        BOOT_ARGS=(--out-dir "${BOOT_OUT}" --n-boot "${N_BOOT:-1000}")
        [[ -n "${RESULTS_ROOT:-}" ]] && BOOT_ARGS+=(--results-root "${RESULTS_ROOT}")
        case "${CLUSTERED:-}" in
            1|true|yes|on) BOOT_ARGS+=(--clustered) ;;
        esac
        python -m src.analysis.bootstrap_ci "${BOOT_ARGS[@]}"
        ;;
    seed_ci)
        SEED_OUT="${SEED_OUT:-${OUT_DIR}/bootstrap}"
        SEED_ARGS=(--out-dir "${SEED_OUT}" --ci "${SEED_CI:-0.95}")
        [[ -n "${RESULTS_ROOT:-}" ]] && SEED_ARGS+=(--results-root "${RESULTS_ROOT}")
        python -m src.analysis.seed_ci "${SEED_ARGS[@]}"
        ;;
    paired_compare)
        PAIR_OUT="${PAIR_OUT:-${OUT_DIR}/paired_compare}"
        PAIR_ARGS=(--out-dir "${PAIR_OUT}")
        [[ -n "${BASELINE_EXPERIMENT:-}" ]] && PAIR_ARGS+=(--baseline-experiment "${BASELINE_EXPERIMENT}")
        [[ -n "${BASELINE_MODEL:-}" ]] && PAIR_ARGS+=(--baseline-model "${BASELINE_MODEL}")
        python -m src.analysis.paired_compare "${PAIR_ARGS[@]}"
        ;;
    *)
        echo "ERROR: unknown CPU experiment '${EXPERIMENT}'"
        echo "Known: ${CPU_TRAIN[*]} ${CPU_ANALYSIS[*]}"
        exit 1
        ;;
esac

echo "CPU job finished."
