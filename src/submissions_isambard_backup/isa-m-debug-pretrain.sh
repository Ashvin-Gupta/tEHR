#!/bin/bash
#SBATCH --job-name=isa-m-debug-pretrain
#SBATCH --output=logs/isa-m-debug-pretrain_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -e

BASE_DIR="${HOME}/TextCancEHR2"

mkdir -p "${HOME}/TextCancEHR2/logs"
cd "${BASE_DIR}"
echo "Starting pretrain debug from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

source ~/miniforge3/bin/activate
conda activate meds_train_clean_311

# Defaults match mimic_pretrain.yaml (envsubst paths resolved at runtime on cluster).
export DEBUG_MODEL_NAME="${DEBUG_MODEL_NAME:-unsloth/mistral-7b-v0.3-bnb-4bit}"
export DEBUG_MAX_LENGTH="${DEBUG_MAX_LENGTH:-16384}"
export DEBUG_PACKED_CACHE="${DEBUG_PACKED_CACHE:-${PROJECTDIR}/TextCancEHR2/experiments/Results/Pretrain-Mistral-7B-Mimic/data_cache/train_packed}"
export DEBUG_VAL_PACKED_CACHE="${DEBUG_VAL_PACKED_CACHE:-${PROJECTDIR}/TextCancEHR2/experiments/Results/Pretrain-Mistral-7B-Mimic/data_cache/val_packed}"
export DEBUG_DATA_DIR="${DEBUG_DATA_DIR:-${HOME}/MIMIC-IV_tokenisation}"
export DEBUG_VOCAB_FILE="${DEBUG_VOCAB_FILE:-${HOME}/MIMIC-IV_tokenisation/vocab.csv}"
export DEBUG_LOOKUP_DIR="${DEBUG_LOOKUP_DIR:-${HOME}/MIMIC_data/hosp}"
export DEBUG_ACES_LABEL_DIR="${DEBUG_ACES_LABEL_DIR:-${HOME}/MIMIC_data/labels/icu_mortality}"
export DEBUG_MAX_CHUNKS="${DEBUG_MAX_CHUNKS:-40}"
export DEBUG_MAX_PATIENTS="${DEBUG_MAX_PATIENTS:-10}"
export DEBUG_SKIP_RAW="${DEBUG_SKIP_RAW:-1}"

echo "DEBUG_PACKED_CACHE=${DEBUG_PACKED_CACHE}"
echo "DEBUG_VAL_PACKED_CACHE=${DEBUG_VAL_PACKED_CACHE}"

python src/debug/debug_pretrain_packed.py

echo "Debug finished."
conda deactivate
