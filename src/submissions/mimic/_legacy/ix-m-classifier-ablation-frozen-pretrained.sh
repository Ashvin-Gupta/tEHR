#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ix-m-ablation-frozen-pretrained.log
#SBATCH --job-name=ix-m-ablation-fp

set -e

BASE_DIR="/home/ag619/TextCancEHR2"
export WANDB_API_KEY="REDACTED"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

cd "${BASE_DIR}"
echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
MASTER_PORT=$(get_free_port)
export MASTER_PORT
echo "Using MASTER_PORT=$MASTER_PORT"

# Ablation 5b: pretrained Mistral, frozen, mean-pool, head only
torchrun --nproc_per_node=1 src/pipelines/finetune_llm_classifier.py \
    --config_filepath src/configs/mimic_classifier_ablation_frozen_pretrained.yaml

echo "Pipeline finished."
conda deactivate
