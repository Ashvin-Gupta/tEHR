#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/mimic-cpu-%x-%j.log
#SBATCH --job-name=mimic-cpu
#
# =============================================================================
# MIMIC CPU experiment dispatcher (no GPU request)
# =============================================================================
#
# Usage:
#   sbatch src/submissions/mimic/run_cpu.sh <experiment> [task]
#
# CPU experiments:
#   xgboost | clinicalbert
#   rare_code | qualitative_dump | preview_surface
#
# Tasks: icu_mortality | hospital_mortality | hospital_readmission
#        (default: icu_mortality)
#
# Optional env for analysis jobs: MODEL SEED RESULTS_JSON COMPARE_RESULTS_JSON
#   COMPARE_NAME OUT_DIR
#
# Examples:
#   sbatch src/submissions/mimic/run_cpu.sh xgboost icu_mortality
#   sbatch src/submissions/mimic/run_cpu.sh rare_code icu_mortality
#   sbatch src/submissions/mimic/run_cpu.sh qualitative_dump hospital_mortality
#
# Note: this cluster only has the `compute` partition (shared with GPU jobs).
# CPU jobs still wait when nodes are full; they just don't need a free GPU.
# =============================================================================

set -euo pipefail

EXPERIMENT="${1:-}"
shift || true

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Usage: sbatch $0 <experiment> [task]"
    echo "  CPU experiments: xgboost clinicalbert rare_code qualitative_dump preview_surface"
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

# SLURM copies this script to /var/spool/.../slurm_script, so BASH_SOURCE is
# unreliable — always source common.sh from the repo path.
BASE_DIR="/home/ag619/TextCancEHR2"
# shellcheck source=common.sh
source "${BASE_DIR}/src/submissions/mimic/common.sh"

MODEL="${MODEL:-mistral}"
SEED="${SEED:-42}"
RESULTS_ROOT="/home/ag619/experiments/Results"
DEFAULT_RESULTS_JSON="${RESULTS_ROOT}/mimic-classifier-${MODEL}-${TASK}/seed_${SEED}/results/test_results.json"
RESULTS_JSON="${RESULTS_JSON:-${DEFAULT_RESULTS_JSON}}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/analysis_out/${EXPERIMENT}-${MODEL}-${TASK}}"

echo "============================================================"
echo " MIMIC CPU run"
echo "   experiment : ${EXPERIMENT}"
echo "   task       : ${TASK}"
echo "   model      : ${MODEL}"
echo "   host       : $(hostname)"
echo "============================================================"

RESOLVED_DIR="${BASE_DIR}/src/configs/mimic/_resolved"
mkdir -p "${RESOLVED_DIR}"

case "${EXPERIMENT}" in
    xgboost|clinicalbert)
        RESOLVED_CFG="${RESOLVED_DIR}/${EXPERIMENT}-${TASK}.yaml"
        python -m src.training.config_utils build \
            --experiment "${EXPERIMENT}" \
            --task "${TASK}" \
            --out "${RESOLVED_CFG}"
        echo "Resolved config → ${RESOLVED_CFG}"
        ;;
    rare_code|qualitative_dump|preview_surface)
        RESOLVED_CFG="${RESOLVED_DIR}/classifier-${MODEL}-${TASK}.yaml"
        python -m src.training.config_utils build \
            --experiment classifier \
            --model "${MODEL}" \
            --task "${TASK}" \
            --out "${RESOLVED_CFG}"
        echo "Resolved classifier config → ${RESOLVED_CFG}"
        ;;
    *)
        echo "ERROR: '${EXPERIMENT}' is not a CPU experiment."
        echo "Use run_cpu.sh for: xgboost clinicalbert rare_code qualitative_dump preview_surface"
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
    rare_code)
        mkdir -p "${OUT_DIR}"
        echo "RESULTS_JSON → ${RESULTS_JSON}"
        echo "OUT_DIR      → ${OUT_DIR}"
        if [[ ! -f "${RESULTS_JSON}" ]]; then
            echo "ERROR: results file not found: ${RESULTS_JSON}"
            echo "Override with RESULTS_JSON=/path/to/test_results.json"
            exit 1
        fi
        RARE_ARGS=(
            --config_filepath "${RESOLVED_CFG}"
            --results_json "${RESULTS_JSON}"
            --model_name "text_llm"
            --split held_out
            --out_dir "${OUT_DIR}"
            --save_per_example
        )
        if [[ -n "${COMPARE_RESULTS_JSON:-}" ]]; then
            RARE_ARGS+=(--compare_results_json "${COMPARE_RESULTS_JSON}")
            RARE_ARGS+=(--compare_name "${COMPARE_NAME:-encoder}")
        fi
        python -m src.analysis.rare_code_subgroup "${RARE_ARGS[@]}"
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
esac

echo "Pipeline finished."
conda deactivate
