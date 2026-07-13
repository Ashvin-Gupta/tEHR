#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
# CPU experiments: xgboost | clinicalbert
# Tasks: icu_mortality | hospital_mortality | hospital_readmission
#        (default: icu_mortality)
#
# Examples:
#   sbatch src/submissions/mimic/run_cpu.sh xgboost icu_mortality
#   sbatch src/submissions/mimic/run_cpu.sh clinicalbert hospital_readmission
#
# Note: this cluster only has the `compute` partition (shared with GPU jobs).
# CPU jobs still wait when nodes are full; they just don't need a free GPU.
# =============================================================================

set -euo pipefail

EXPERIMENT="${1:-}"
shift || true

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Usage: sbatch $0 <experiment> [task]"
    echo "  CPU experiments: xgboost | clinicalbert"
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

echo "============================================================"
echo " MIMIC CPU run"
echo "   experiment : ${EXPERIMENT}"
echo "   task       : ${TASK}"
echo "   host       : $(hostname)"
echo "============================================================"

RESOLVED_DIR="${BASE_DIR}/src/configs/mimic/_resolved"
mkdir -p "${RESOLVED_DIR}"
RESOLVED_CFG="${RESOLVED_DIR}/${EXPERIMENT}-${TASK}.yaml"

python -m src.training.config_utils build \
    --experiment "${EXPERIMENT}" \
    --task "${TASK}" \
    --out "${RESOLVED_CFG}"
echo "Resolved config → ${RESOLVED_CFG}"

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
    *)
        echo "ERROR: '${EXPERIMENT}' is not a CPU experiment."
        echo "Use run_cpu.sh for: xgboost clinicalbert"
        echo "Use run.sh (GPU) for everything else."
        exit 1
        ;;
esac

echo "Pipeline finished."
conda deactivate
