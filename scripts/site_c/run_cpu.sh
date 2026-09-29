#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=256G
#SBATCH --partition=medium
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=tehr-cpu
#
# Site C (hx2) CPU dispatcher.
# small QoS caps memory at 64G; MIMIC pickle loads need the medium partition.
# Site C (hx2) CPU dispatcher.
# Usage: sbatch scripts/site_c/run_cpu.sh <dataset> <experiment> [task]
# Experiments: xgboost_bow | plot_ablation | plot_ablation_deltas | bootstrap_ci | seed_ci | paired_compare
#
# Examples:
#   sbatch scripts/site_c/run_cpu.sh mimic xgboost_bow icu_mortality
#   SEED=43 sbatch scripts/site_c/run_cpu.sh mimic xgboost_bow hospital_readmission
#   sbatch scripts/site_c/run_cpu.sh seed_ci
#   sbatch scripts/site_c/run_cpu.sh mimic xgboost_bow icu_mortality evaluate

set -euo pipefail

_FROM_SCRIPT=""
if _dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; then
    _FROM_SCRIPT="${_dir}"
fi
_GIVEN_REPO="${REPO_ROOT:-}"
REPO_ROOT=""
for _cand in "${_GIVEN_REPO}" "${_FROM_SCRIPT}" "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${HOME}/TextCancEHR2"; do
    [[ -n "${_cand}" && -f "${_cand}/scripts/lib/dispatch.sh" ]] || continue
    REPO_ROOT="$(cd "${_cand}" && pwd)"
    break
done
if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: could not find TextCancEHR2 (missing scripts/lib/dispatch.sh)."
    echo "  BASH_SOURCE=${BASH_SOURCE[0]}"
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "  PWD=${PWD}"
    echo "Set REPO_ROOT to the repo directory and retry."
    exit 1
fi
export REPO_ROOT
SITE_DIR="${REPO_ROOT}/scripts/site_c"
# shellcheck source=../lib/dispatch.sh
source "${REPO_ROOT}/scripts/lib/dispatch.sh"
# shellcheck source=../lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"
# shellcheck source=../lib/parse_cpu.sh
source "${REPO_ROOT}/scripts/lib/parse_cpu.sh"
