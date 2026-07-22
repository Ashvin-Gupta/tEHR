#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/isa-mimic-cpu-%x-%j.log
#SBATCH --job-name=isa-mimic-cpu
#
# =============================================================================
# Isambard MIMIC CPU experiment dispatcher (no GPU request)
# =============================================================================
#
# Usage:
#   sbatch src/submissions/isambard/run_cpu.sh <experiment> [task]
#
# CPU experiments:
#   xgboost | clinicalbert
#   subgroup_full      # classifier vs encoder (matched full adaptation)
#   subgroup_frozen    # LLM 5b vs encoder_frozen_head (matched frozen backbone)
#   subgroup_analysis  # custom pair (set PRIMARY_JSON, COMPARE_JSON, ...)
#   rare_code          # alias for subgroup_analysis
#   qualitative_dump | preview_surface | plot_convergence
#
# Examples:
#   sbatch src/submissions/isambard/run_cpu.sh subgroup_full icu_mortality
#   sbatch src/submissions/isambard/run_cpu.sh subgroup_frozen hospital_mortality
# =============================================================================

set -euo pipefail

EXPERIMENT="${1:-}"
shift || true

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Usage: sbatch $0 <experiment> [task]"
    echo "  CPU experiments: xgboost clinicalbert subgroup_full subgroup_frozen subgroup_analysis rare_code qualitative_dump preview_surface plot_convergence"
    echo "  tasks: icu_mortality | hospital_mortality | hospital_readmission"
    exit 1
fi

KNOWN_TASKS=("icu_mortality" "hospital_mortality" "hospital_readmission")
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
    if is_in "$arg" "${KNOWN_TASKS[@]}"; then
        TASK="$arg"
    else
        echo "ERROR: unknown argument '$arg'"
        echo "  tasks: ${KNOWN_TASKS[*]}"
        exit 1
    fi
done

[[ -z "${TASK}" ]] && TASK="icu_mortality"

BASE_DIR="${HOME}/TextCancEHR2"
# shellcheck source=common.sh
source "${BASE_DIR}/src/submissions/isambard/common.sh"

MODEL="${MODEL:-mistral}"
SEED="${SEED:-42}"
RESULTS_ROOT="/projects/u6ps/TextCancEHR2/experiments/Results"

# Default paths for fair comparison pairs
_DEFAULT_PRIMARY_FULL="${RESULTS_ROOT}/mimic-classifier-${MODEL}-${TASK}/seed_${SEED}/results/test_results.json"
_DEFAULT_COMPARE_FULL="${RESULTS_ROOT}/mimic-encoder-transformer-${TASK}/results/test_results.json"
_DEFAULT_PRIMARY_FROZEN="${RESULTS_ROOT}/mimic-ablation-frozen-pretrained-${MODEL}-${TASK}/seed_${SEED}/results/test_results.json"
_DEFAULT_COMPARE_FROZEN="${RESULTS_ROOT}/mimic-encoder-frozen-head-${TASK}/results/test_results.json"

PRIMARY_JSON="${PRIMARY_JSON:-}"
COMPARE_JSON="${COMPARE_JSON:-}"
PRIMARY_NAME="${PRIMARY_NAME:-text_llm}"
COMPARE_NAME="${COMPARE_NAME:-encoder}"
COMPARISON_PAIR="${COMPARISON_PAIR:-custom}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/analysis_out/${EXPERIMENT}-${MODEL}-${TASK}}"

case "${EXPERIMENT}" in
    subgroup_full)
        PRIMARY_JSON="${PRIMARY_JSON:-${_DEFAULT_PRIMARY_FULL}}"
        COMPARE_JSON="${COMPARE_JSON:-${_DEFAULT_COMPARE_FULL}}"
        PRIMARY_NAME="${PRIMARY_NAME:-text_llm}"
        COMPARE_NAME="${COMPARE_NAME:-encoder}"
        COMPARISON_PAIR="${COMPARISON_PAIR:-full_adaptation}"
        OUT_DIR="${OUT_DIR:-${BASE_DIR}/analysis_out/subgroup-full-${MODEL}-${TASK}}"
        ;;
    subgroup_frozen)
        PRIMARY_JSON="${PRIMARY_JSON:-${_DEFAULT_PRIMARY_FROZEN}}"
        COMPARE_JSON="${COMPARE_JSON:-${_DEFAULT_COMPARE_FROZEN}}"
        PRIMARY_NAME="${PRIMARY_NAME:-llm_frozen}"
        COMPARE_NAME="${COMPARE_NAME:-encoder_frozen}"
        COMPARISON_PAIR="${COMPARISON_PAIR:-frozen_backbone}"
        OUT_DIR="${OUT_DIR:-${BASE_DIR}/analysis_out/subgroup-frozen-${MODEL}-${TASK}}"
        ;;
esac

# Legacy alias
RESULTS_JSON="${RESULTS_JSON:-${PRIMARY_JSON}}"

echo "============================================================"
echo " Isambard MIMIC CPU run"
echo "   experiment : ${EXPERIMENT}"
echo "   task       : ${TASK}"
echo "   model      : ${MODEL}"
echo "   site       : isambard"
echo "   host       : $(hostname)"
echo "============================================================"

RESOLVED_DIR="${BASE_DIR}/src/configs/mimic/_resolved_isambard"
mkdir -p "${RESOLVED_DIR}"

case "${EXPERIMENT}" in
    xgboost|clinicalbert)
        RESOLVED_CFG="${RESOLVED_DIR}/${EXPERIMENT}-${TASK}.yaml"
        python -m src.training.config_utils build \
            --experiment "${EXPERIMENT}" \
            --task "${TASK}" \
            --site isambard \
            --out "${RESOLVED_CFG}"
        echo "Resolved config → ${RESOLVED_CFG}"
        ;;
    rare_code|qualitative_dump|preview_surface|plot_convergence|subgroup_full|subgroup_frozen|subgroup_analysis)
        RESOLVED_CFG="${RESOLVED_DIR}/classifier-${MODEL}-${TASK}.yaml"
        python -m src.training.config_utils build \
            --experiment classifier \
            --model "${MODEL}" \
            --task "${TASK}" \
            --site isambard \
            --out "${RESOLVED_CFG}"
        echo "Resolved classifier config → ${RESOLVED_CFG}"
        ;;
    *)
        echo "ERROR: '${EXPERIMENT}' is not a CPU experiment."
        echo "Use run_cpu.sh for: xgboost clinicalbert subgroup_full subgroup_frozen subgroup_analysis rare_code qualitative_dump preview_surface plot_convergence"
        echo "Use run.sh (GPU) for everything else."
        exit 1
        ;;
esac

case "${EXPERIMENT}" in
    xgboost)
        python -m src.pipelines.run_baseline \
            --baseline xgboost_bow \
            --config_filepath "${RESOLVED_CFG}"
        ;;
    clinicalbert)
        python -m src.pipelines.run_baseline \
            --baseline clinicalbert \
            --config_filepath "${RESOLVED_CFG}"
        ;;
    subgroup_full|subgroup_frozen|subgroup_analysis|rare_code)
        mkdir -p "${OUT_DIR}"
        if [[ -z "${PRIMARY_JSON}" ]]; then
            echo "ERROR: PRIMARY_JSON not set"
            exit 1
        fi
        if [[ ! -f "${PRIMARY_JSON}" ]]; then
            echo "ERROR: primary results not found: ${PRIMARY_JSON}"
            exit 1
        fi
        SUBGROUP_ARGS=(
            --config_filepath "${RESOLVED_CFG}"
            --primary_json "${PRIMARY_JSON}"
            --primary_name "${PRIMARY_NAME}"
            --split held_out
            --out_dir "${OUT_DIR}"
            --comparison_pair "${COMPARISON_PAIR}"
            --save_per_example
        )
        if [[ -n "${COMPARE_JSON}" ]]; then
            if [[ ! -f "${COMPARE_JSON}" ]]; then
                echo "ERROR: compare results not found: ${COMPARE_JSON}"
                exit 1
            fi
            SUBGROUP_ARGS+=(--compare_json "${COMPARE_JSON}")
            SUBGROUP_ARGS+=(--compare_name "${COMPARE_NAME}")
        fi
        echo "PRIMARY  → ${PRIMARY_JSON}"
        echo "COMPARE  → ${COMPARE_JSON:-<none>}"
        echo "OUT_DIR  → ${OUT_DIR}"
        python -m src.analysis.subgroup_analysis "${SUBGROUP_ARGS[@]}"
        ;;
    qualitative_dump)
        mkdir -p "${OUT_DIR}"
        echo "RESULTS_JSON → ${RESULTS_JSON}"
        echo "OUT_DIR      → ${OUT_DIR}"
        if [[ ! -f "${RESULTS_JSON}" ]]; then
            echo "ERROR: results file not found: ${RESULTS_JSON}"
            echo "Override with RESULTS_JSON=/path/to/test_results.json"
            exit 1
        fi
        python -m src.analysis.qualitative_occlusion \
            --config_filepath "${RESOLVED_CFG}" \
            --results_json "${RESULTS_JSON}" \
            --out_dir "${OUT_DIR}" \
            --seed "${SEED}" \
            --split held_out \
            --n_cases 3 \
            --dump_only
        ;;
    preview_surface)
        mkdir -p "${OUT_DIR}"
        python -m src.analysis.preview_surface_forms \
            --config_filepath "${RESOLVED_CFG}" \
            --split held_out \
            --n 3 \
            --forms text raw scramble_hash scramble_shuffle \
            | tee "${OUT_DIR}/preview.txt"
        ;;
    plot_convergence)
        OUT_DIR="${OUT_DIR:-${BASE_DIR}/analysis_out/convergence-${MODEL}-${TASK}}"
        mkdir -p "${OUT_DIR}"
        PLOT_ARGS=(
            --task "${TASK}"
            --model "${MODEL}"
            --seed "${SEED}"
            --config "${RESOLVED_CFG}"
            --out-dir "${OUT_DIR}"
        )
        if [[ -n "${ENCODER_LOG:-}" ]]; then
            PLOT_ARGS+=(--encoder-log "${ENCODER_LOG}")
        fi
        if [[ -n "${LLM_RESULTS_DIR:-}" ]]; then
            PLOT_ARGS+=(--llm-results-dir "${LLM_RESULTS_DIR}")
        fi
        echo "OUT_DIR → ${OUT_DIR}"
        python -m src.analysis.plot_training_convergence "${PLOT_ARGS[@]}"
        ;;
esac

echo "Pipeline finished."
conda deactivate
