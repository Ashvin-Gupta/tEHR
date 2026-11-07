#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/%x-%j.log
#SBATCH --job-name=emb-run
#
# Phrase-embedding transformer for one embedder, all four MIMIC tasks.
# The first task writes the shared phrase cache; the rest reuse it.
#
#   TEHR_EMBEDDER_TAG=modernbert \
#   TEHR_EMBEDDER_MODEL=thomas-sounack/BioClinical-ModernBERT-base \
#   sbatch --job-name=emb-modernbert scripts/site_c/text_embedding_embedder.sh

set -euo pipefail

if [[ -z "${TEHR_EMBEDDER_TAG:-}" || -z "${TEHR_EMBEDDER_MODEL:-}" ]]; then
    echo "ERROR: set TEHR_EMBEDDER_TAG and TEHR_EMBEDDER_MODEL"
    exit 1
fi

REPO_ROOT="/hx2-weka/home/ag619/TextCancEHR2"
cd "${REPO_ROOT}"
export TEHR_EMBEDDER_TAG TEHR_EMBEDDER_MODEL

echo "Embedder ${TEHR_EMBEDDER_TAG}: ${TEHR_EMBEDDER_MODEL}"
for task in icu_mortality icu_readmission hospital_mortality hospital_readmission; do
    echo "============================================================"
    echo " text_embedding ${TEHR_EMBEDDER_TAG} ${task}"
    echo "============================================================"
    bash scripts/site_c/run.sh mimic text_embedding "${task}"
done
