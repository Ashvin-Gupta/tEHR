#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=tehr-run
#
# Site B GPU dispatcher. Default allocation: 4 GPUs (training).
# Evaluate / zero-shot only need one: sbatch --gpus=1 scripts/site_b/run.sh ...
#
# Usage (from the repository root):
#   sbatch scripts/site_b/run.sh <dataset> <experiment> [model] [task]
# Interactive (srun pty shell on a compute node; salloc is not used here):
#   srun --gpus=1 --reservation=interactive --pty --time=04:00:00 bash -i
#   cd ~/TextCancEHR2
#   bash scripts/site_b/run.sh <dataset> <experiment> [model] [task]

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch, or run it inside an srun pty shell, e.g.:"
    echo "  sbatch scripts/site_b/run.sh mimic tehr_stage2 mistral icu_mortality"
    echo "  srun --gpus=1 --reservation=interactive --pty --time=04:00:00 bash -i"
    echo "  cd ~/TextCancEHR2 && bash scripts/site_b/run.sh mimic tehr_stage2 mistral icu_mortality"
    exit 1
fi

# sbatch copies this file to /var/spool, so BASH_SOURCE is not the repo.
# srun --pty keeps SLURM_SUBMIT_DIR as the directory you launched from (often $HOME).
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
echo "   slurm job  : ${SLURM_JOB_ID}"
echo "   command    : sbatch scripts/site_b/run.sh ${DATASET} ${EXPERIMENT}${MODEL:+ ${MODEL}}${TASK:+ ${TASK}}${EVAL_ONLY:+ evaluate}"
echo "   dataset    : ${DATASET}"
echo "   experiment : ${EXPERIMENT}"
echo "   model      : ${MODEL:-<none>}"
echo "   task       : ${TASK:-<none>}"
echo "   host       : $(hostname)"
if [[ -n "${EVAL_SHARD:-}" ]]; then
    echo "   eval shard : ${EVAL_SHARD}"
fi
echo "============================================================"

dispatch_experiment
echo "Pipeline finished."
conda deactivate || true
