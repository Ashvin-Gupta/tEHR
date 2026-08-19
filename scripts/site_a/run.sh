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

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
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
echo "   dataset    : ${DATASET}"
echo "   experiment : ${EXPERIMENT}"
echo "   model      : ${MODEL:-<none>}"
echo "   task       : ${TASK:-<none>}"
echo "   host       : $(hostname)"
echo "============================================================"

dispatch_experiment
echo "Pipeline finished."
conda deactivate || true
