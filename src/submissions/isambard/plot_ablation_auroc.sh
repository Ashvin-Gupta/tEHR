#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/isa-plot-ablation-%j.log
#SBATCH --job-name=isa-plot-ablation
#
# Render ablation AUROC figure variants (dot_2x2, matrix, bar_2x2, dot_two_task).
#
# Usage:
#   sbatch src/submissions/isambard/plot_ablation_auroc.sh
#   sbatch src/submissions/isambard/plot_ablation_auroc.sh --style dot_2x2 --annotate
#   STYLE=matrix sbatch src/submissions/isambard/plot_ablation_auroc.sh
#
set -euo pipefail

BASE_DIR="${HOME}/TextCancEHR2"
# shellcheck source=common.sh
source "${BASE_DIR}/src/submissions/isambard/common.sh"

CSV="${CSV:-${BASE_DIR}/analysis_out/ablations/results.csv}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/analysis_out/ablations}"
STYLE="${STYLE:-all}"

mkdir -p "${OUT_DIR}" "${BASE_DIR}/logs"

echo "CSV     → ${CSV}"
echo "OUT_DIR → ${OUT_DIR}"
echo "STYLE   → ${STYLE}"

python -m src.analysis.plot_ablation_auroc \
    --csv "${CSV}" \
    --out-dir "${OUT_DIR}" \
    --style "${STYLE}" \
    "$@"

echo "Done."
