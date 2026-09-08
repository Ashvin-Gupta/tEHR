#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=tehr-run
#
# Site A GPU dispatcher. Cluster account / partition: pass extra sbatch flags
# or set them in this site's gitignored local.env comments.
#
# Usage (submit from the repository root):
#   sbatch scripts/site_a/run.sh <dataset> <experiment> [model] [task]
#
# Examples:
#   sbatch scripts/site_a/run.sh mimic tehr_stage2 mistral icu_mortality
#   sbatch scripts/site_a/run.sh cprd sft qwen8 6m
#   sbatch --gres=gpu:2 scripts/site_a/run.sh mimic tehr_stage1 mistral

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
SITE_DIR="${REPO_ROOT}/scripts/site_a"
DEFAULT_NPROC=1

# shellcheck source=../lib/dispatch.sh
source "${REPO_ROOT}/scripts/lib/dispatch.sh"
# shellcheck source=../lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"
# shellcheck source=../lib/parse_gpu.sh
source "${REPO_ROOT}/scripts/lib/parse_gpu.sh"

echo "============================================================"
echo " tEHR run (site_a)"
echo "   slurm job  : ${SLURM_JOB_ID}"
echo "   command    : sbatch scripts/site_a/run.sh ${DATASET} ${EXPERIMENT}${MODEL:+ ${MODEL}}${TASK:+ ${TASK}}${EVAL_ONLY:+ evaluate}"
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
