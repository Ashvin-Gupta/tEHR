# CPU dispatcher body. Sourced by scripts/site_*/run_cpu.sh after common.sh.

CPU_ANALYSIS=(plot_ablation hdbscan plot_labs plot_rr)
CPU_TRAIN=(xgboost_bow)

ARG1="${1:-}"
ARG2="${2:-}"
shift 2 || true

if [[ -z "${ARG1}" ]]; then
    echo "Usage: sbatch $0 <dataset> xgboost_bow [task] [evaluate]"
    echo "   or: sbatch $0 plot_ablation|hdbscan|plot_labs|plot_rr [args via env]"
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
if [[ "${DATASET}" == "cprd" ]]; then
    KNOWN_TASKS=("${CPRD_TASKS[@]}")
else
    KNOWN_TASKS=("${MIMIC_TASKS[@]}")
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
        [[ -z "${TASK}" && "${DATASET}" == "cprd" ]] && TASK="6m"
        [[ -z "${TASK}" ]] && TASK="icu_mortality"
        TAG="${DATASET}-xgboost_bow-${TASK}"
        RESOLVED_CFG="${REPO_ROOT}/configs/_resolved/${TAG}.yaml"
        python -m src.training.config_utils build \
            --dataset "${DATASET}" --experiment xgboost_bow --task "${TASK}" \
            --out "${RESOLVED_CFG}"
        extra=()
        if [[ -n "${EVAL_ONLY}" ]]; then
            extra+=(--eval_only)
            [[ -n "${SEED:-}" ]] && extra+=(--seed "${SEED}")
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
    hdbscan)
        : "${EMBEDDINGS_DIR:?set EMBEDDINGS_DIR to the directory with embeddings.npy}"
        : "${CONFIG_PATH:?set CONFIG_PATH to a resolved experiment YAML}"
        HDBSCAN_OUT="${HDBSCAN_OUT:-${OUT_DIR}/hdbscan}"
        python -m src.analysis.hdbscan \
            --config-path "${CONFIG_PATH}" \
            --embeddings-dir "${EMBEDDINGS_DIR}" \
            --output-dir "${HDBSCAN_OUT}" \
            ${PREDICTIONS_JSON:+--predictions-json "${PREDICTIONS_JSON}"}
        ;;
    plot_labs)
        : "${CLUSTERS_CSV:?set CLUSTERS_CSV}"
        : "${CONFIG_PATH:?set CONFIG_PATH to a resolved experiment YAML}"
        python -m src.analysis.plot_lab_violins \
            --clusters-csv "${CLUSTERS_CSV}" \
            --config-path "${CONFIG_PATH}" \
            --output-path "${LAB_OUT:-${OUT_DIR}/lab_violins.png}"
        ;;
    plot_rr)
        : "${SUMMARY_CSV:?set SUMMARY_CSV to full_differential_analysis.csv}"
        python -m src.analysis.clinical_heatmap_terms \
            --summary-csv "${SUMMARY_CSV}" \
            --output-dir "${RR_OUT:-${OUT_DIR}/rr}"
        ;;
    *)
        echo "ERROR: unknown CPU experiment '${EXPERIMENT}'"
        echo "Known: ${CPU_TRAIN[*]} ${CPU_ANALYSIS[*]}"
        exit 1
        ;;
esac

echo "CPU job finished."
