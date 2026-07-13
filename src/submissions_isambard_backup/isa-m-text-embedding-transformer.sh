#!/bin/bash
#SBATCH --job-name=isa-m-text-embedding-transformer
#SBATCH --output=logs/isa-m-text-embedding-transformer_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00

set -e

BASE_DIR="${HOME}/TextCancEHR2"

# --- Paths ---
mkdir -p "${HOME}/TextCancEHR2/logs"
mkdir -p "${PROJECTDIR}/TextCancEHR2/experiments/Results/mimic-baselines/text_embedding_transformer"

EMBEDDINGS_PATH="${PROJECTDIR}/TextCancEHR2/experiments/Results/mimic-baselines/text_embedding_transformer/token_embeddings_qwen3_0.6B.pt"
VOCAB_PATH="${HOME}/MIMIC-IV_tokenisation/vocab.csv"
LOOKUP_DIR="${HOME}/MIMIC_data/hosp"

# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Starting text-embedding transformer from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source ~/miniforge3/bin/activate
conda activate meds_train_clean_311

# ---------------------------------------------------------------------------
# Step 1: Precompute token embeddings (skip if already done)
# ---------------------------------------------------------------------------
if [ ! -f "${EMBEDDINGS_PATH}" ]; then
    echo "Precomputing token embeddings..."
    python -m src.pipelines.precompute_token_embeddings \
        --vocab_filepath  "${VOCAB_PATH}" \
        --lookup_dir      "${LOOKUP_DIR}" \
        --model_name      "Qwen/Qwen3-Embedding-0.6B" \
        --output_path     "${EMBEDDINGS_PATH}" \
        --batch_size      256 \
        --max_length      64 \
        --device          cuda
    echo "Precompute done."
else
    echo "Embeddings already exist at ${EMBEDDINGS_PATH}, skipping precompute."
fi

# ---------------------------------------------------------------------------
# Step 2: Train and evaluate the text-embedding transformer
# ---------------------------------------------------------------------------
export EMBEDDINGS_PATH

TEMP_CONFIG=$(mktemp /tmp/isa_config_XXXXXX.yaml)
trap "rm -f ${TEMP_CONFIG}" EXIT
envsubst < "${BASE_DIR}/src/configs_isambard/baseline_text_embedding_transformer_mimic_config.yaml" > "${TEMP_CONFIG}"

echo "Running text-embedding transformer baseline..."
python -m src.pipelines.run_baseline \
    --baseline text_embedding_transformer \
    --config_filepath "${TEMP_CONFIG}"

echo "Pipeline finished."
conda deactivate
