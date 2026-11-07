#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=256G
#SBATCH --partition=medium
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=paired-seeds
#
# Multi-seed paired MIMIC bootstrap. CPU only.
#   sbatch scripts/site_c/paired_multiseed.sh

set -euo pipefail

REPO_ROOT="/hx2-weka/home/ag619/TextCancEHR2"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/site_c/local.env"
# shellcheck disable=SC1091
source "${VENV_ACTIVATE}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m src.analysis.paired_multiseed
