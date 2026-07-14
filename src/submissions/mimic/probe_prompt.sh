#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=mimic-probe
#
# Short GPU probe: compare MedGemma/Mistral prompt+generation strategies.
#
# Usage:
#   sbatch src/submissions/mimic/probe_prompt.sh medgemma
#   sbatch src/submissions/mimic/probe_prompt.sh mistral
#
set -euo pipefail

MODEL="${1:-medgemma}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

case "${MODEL}" in
  medgemma)
    MODEL_NAME="google/medgemma-1.5-4b-it"
    EXTRA=( )
    ;;
  mistral)
    MODEL_NAME="unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    EXTRA=( --use-unsloth )
    ;;
  *)
    echo "Unknown model '${MODEL}'. Use: medgemma | mistral"
    exit 1
    ;;
esac

echo "============================================================"
echo " MIMIC prompt probe"
echo "   model      : ${MODEL}"
echo "   model_name : ${MODEL_NAME}"
echo "   host       : $(hostname)"
echo "============================================================"

python src/diagnostics/probe_prompt_generation.py \
  --model-name "${MODEL_NAME}" \
  "${EXTRA[@]}" \
  --max-new-tokens 16
