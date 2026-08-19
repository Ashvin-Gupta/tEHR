#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=tehr-cpu
#
# Site B CPU dispatcher.
# Usage: sbatch scripts/site_b/run_cpu.sh <dataset> <experiment> [task]
# Experiments: xgboost_bow | plot_ablation | hdbscan | plot_labs | plot_rr

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SITE_DIR="${REPO_ROOT}/scripts/site_b"
# shellcheck source=../lib/dispatch.sh
source "${REPO_ROOT}/scripts/lib/dispatch.sh"
# shellcheck source=../lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"
# shellcheck source=../lib/parse_cpu.sh
source "${REPO_ROOT}/scripts/lib/parse_cpu.sh"
