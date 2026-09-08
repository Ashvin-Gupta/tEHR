#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=ehrshot-tokenize
#
# Build tEHR pickle shards + ACES labels from EHRSHOT_ASSETS (full timelines).
# Usage (from the repository root):
#   sbatch scripts/site_b/tokenize_ehrshot.sh

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
    exit 1
fi
export REPO_ROOT
SITE_DIR="${REPO_ROOT}/scripts/site_b"
# shellcheck source=../lib/common.sh
source "${REPO_ROOT}/scripts/lib/common.sh"

python -m src.ehrshot.tokenize "$@"
