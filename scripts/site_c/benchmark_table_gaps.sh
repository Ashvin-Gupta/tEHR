#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=icu-gaps
#
# Missing compute-table cells: tEHR Qwen3-0.6B, tEHR Qwen3-8B,
# event-token Mistral, CLMBR-T frozen batch-1 latency.
# Submit from the repository root:
#   sbatch scripts/site_c/benchmark_table_gaps.sh

set -euo pipefail

unset TEHR_STOP_AFTER_STAGE1 || true
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

_FROM_SCRIPT=""
if _dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; then
    _FROM_SCRIPT="${_dir}"
fi
_GIVEN_REPO="${REPO_ROOT:-}"
REPO_ROOT=""
for _cand in "${_GIVEN_REPO}" "${_FROM_SCRIPT}" "${SLURM_SUBMIT_DIR:-}" "${PWD}" "${HOME}/TextCancEHR2"; do
    [[ -n "${_cand}" && -f "${_cand}/scripts/lib/common.sh" ]] || continue
    REPO_ROOT="$(cd "${_cand}" && pwd)"
    break
done
if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: could not find TextCancEHR2"
    exit 1
fi
export REPO_ROOT
SITE_DIR="${REPO_ROOT}/scripts/site_c"

# shellcheck source=../lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

echo "============================================================"
echo " ICU compute table gaps"
echo "   slurm job : ${SLURM_JOB_ID:-local}"
echo "   host      : $(hostname)"
echo "   python    : $(command -v python)"
echo "============================================================"
nvidia-smi || true
python -u "${REPO_ROOT}/scripts/benchmark_table_gaps.py" "$@"
echo "Benchmark finished."
