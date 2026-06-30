#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ix-m-text-embedding-transformer.log
#SBATCH --job-name=ix-m-text-embedding-transformer

set -e

BASE_DIR="/home/ag619/TextCancEHR2"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

cd "${BASE_DIR}"
echo "Starting text-embedding transformer from: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# Step 1: Precompute token embeddings (skip if already done)
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH="/home/ag619/experiments/Results/mimic-baselines/text_embedding_transformer/token_embeddings_qwen3_0.6B.pt"

if [ ! -f "${EMBEDDINGS_PATH}" ]; then
    echo "Precomputing token embeddings..."
    python -m src.pipelines.precompute_token_embeddings \
        --vocab_filepath  "/home/ag619/MIMIC-IV tokensiation/vocab.csv" \
        --lookup_dir      "/home/ag619/MIMIC_data/hosp" \
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
echo "Running text-embedding transformer baseline..."
python -m src.pipelines.run_baseline \
    --baseline text_embedding_transformer \
    --config_filepath src/configs/baseline_text_embedding_transformer_mimic_config.yaml

echo "Pipeline finished."
conda deactivate
