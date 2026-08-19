#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=tehr-run
#
# Site B GPU dispatcher. Default allocation: 4 GPUs.
# Override at submit time: sbatch --gpus=1 scripts/site_b/run.sh ...
#
# Usage (submit from the repository root):
#   sbatch scripts/site_b/run.sh <dataset> <experiment> [model] [task]

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch, e.g.:"
    echo "  sbatch scripts/site_b/run.sh mimic tehr_stage2 mistral icu_mortality"
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SITE_DIR="${REPO_ROOT}/scripts/site_b"
DEFAULT_NPROC=4

# shellcheck source=../lib/dispatch.sh
source "${REPO_ROOT}/scripts/lib/dispatch.sh"
# shellcheck source=../lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"
# shellcheck source=../lib/parse_gpu.sh
source "${REPO_ROOT}/scripts/lib/parse_gpu.sh"

if [[ -z "${SLURM_GPUS_ON_NODE:-}${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: job ${SLURM_JOB_ID} has no GPUs visible to this shell."
    exit 1
fi

echo "============================================================"
echo " tEHR run (site_b)"
echo "   dataset    : ${DATASET}"
echo "   experiment : ${EXPERIMENT}"
echo "   model      : ${MODEL:-<none>}"
echo "   task       : ${TASK:-<none>}"
echo "   host       : $(hostname)"
echo "============================================================"

dispatch_experiment
echo "Pipeline finished."
conda deactivate || true
