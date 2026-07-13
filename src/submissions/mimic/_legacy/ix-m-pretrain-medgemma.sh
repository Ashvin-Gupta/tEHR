#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/ix-m-pretrain-medgemma.log
#SBATCH --job-name=ix-m-pretrain-mg

set -e

BASE_DIR="/home/ag619/TextCancEHR2"
export WANDB_API_KEY="REDACTED"

# HuggingFace token required to access google/medgemma-1.5-4b-it
# Set HF_TOKEN in your environment or uncomment and fill in below:
# export HF_TOKEN="hf_..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

cd "${BASE_DIR}"
echo "Starting MedGemma pretraining from: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
MASTER_PORT=$(get_free_port)
export MASTER_PORT
echo "Using MASTER_PORT=$MASTER_PORT"

torchrun --nproc_per_node=2 src/pipelines/llm_pretrain.py \
    --config_filepath src/configs/mimic_pretrain_medgemma.yaml

echo "Pipeline finished."
conda deactivate
