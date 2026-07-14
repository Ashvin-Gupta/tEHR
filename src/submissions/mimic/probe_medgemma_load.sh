#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=mimic-mg-load
#
# Compare MedGemma load paths for text-only EHR:
#   A) CausalLM + 4-bit (current production)
#   B) ImageTextToText + Processor + bf16 (HF card, text-only message)
#   (+ optional B with 4-bit)
#
# Usage:
#   sbatch src/submissions/mimic/probe_medgemma_load.sh
#
set -euo pipefail

BASE_DIR="/home/ag619/TextCancEHR2"
# shellcheck source=/dev/null
source "${BASE_DIR}/src/submissions/mimic/common.sh"

echo "============================================================"
echo " MedGemma load-path probe"
echo "   host : $(hostname)"
echo "============================================================"

python src/diagnostics/probe_medgemma_load.py \
  --also-b-4bit \
  --max-new-tokens 16
