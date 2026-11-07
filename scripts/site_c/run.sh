#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=tehr-run
#
# Site C (hx2) GPU dispatcher. Default allocation: 1x H200.
# Multi-GPU: sbatch --gres=gpu:h200:N --cpus-per-task=$((N*4)) scripts/site_c/run.sh ...
#
# Usage (submit from the repository root):
#   sbatch scripts/site_c/run.sh <dataset> <experiment> [model] [task]
#
# Examples:
#   sbatch scripts/site_c/run.sh mimic encoder icu_mortality
#   sbatch scripts/site_c/run.sh mimic tehr_stage2 mistral icu_mortality
#   SEED=43 sbatch scripts/site_c/run.sh mimic encoder icu_mortality
#   sbatch scripts/site_c/run.sh mimic encoder icu_mortality evaluate
#   CHECKPOINT=/path/to/best_model.pt sbatch scripts/site_c/run.sh mimic encoder icu_mortality evaluate

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
DEFAULT_NPROC=1

# shellcheck source=../lib/dispatch.sh
source "${REPO_ROOT}/scripts/lib/dispatch.sh"
# shellcheck source=../lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"
# shellcheck source=../lib/parse_gpu.sh
source "${REPO_ROOT}/scripts/lib/parse_gpu.sh"

if [[ -z "${SLURM_GPUS_ON_NODE:-}${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: job ${SLURM_JOB_ID:-local} has no GPUs visible to this shell."
    echo "  Submit with sbatch, or run inside an srun allocation that has --gres=gpu:h200:N."
    exit 1
fi

echo "============================================================"
echo " tEHR run (site_c / hx2)"
echo "   slurm job  : ${SLURM_JOB_ID:-local}"
echo "   command    : sbatch scripts/site_c/run.sh ${DATASET} ${EXPERIMENT}${MODEL:+ ${MODEL}}${TASK:+ ${TASK}}${EVAL_ONLY:+ evaluate}"
echo "   dataset    : ${DATASET}"
echo "   experiment : ${EXPERIMENT}"
echo "   model      : ${MODEL:-<none>}"
echo "   task       : ${TASK:-<none>}"
echo "   host       : $(hostname)"
if [[ -n "${EVAL_SHARD:-}" ]]; then
    echo "   eval shard : ${EVAL_SHARD}"
fi
if [[ -n "${EVAL_SPLIT:-}" ]]; then
    echo "   eval split : ${EVAL_SPLIT}"
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
    echo "   checkpoint : ${CHECKPOINT}"
fi
echo "============================================================"
nvidia-smi || true

dispatch_experiment
echo "Pipeline finished."
if command -v conda >/dev/null 2>&1; then
    conda deactivate || true
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    deactivate || true
fi
