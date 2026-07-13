#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/i-m-evaluate2.log
#SBATCH --job-name=i-m-evaluate2

set -e

# Set the base directory for your project
BASE_DIR="/home/ag619/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

MASTER_PORT=$(get_free_port)
export MASTER_PORT
# export MASTER_PORT=47001

echo "Using MASTER_PORT=$MASTER_PORT for torchrun"

# torchrun --nproc_per_node=2 -m src.evaluations.evaluate_model \
#   --config_filepath src/configs/mimic_classifier.yaml \
#   --model_checkpoint /home/ag619/experiments/Results/mimic-hospital-mortality-classification/seed_2802607356/checkpoint-1000
  
torchrun --nproc_per_node=1 -m src.evaluations.evaluate_model \
  --config_filepath src/configs/baseline_qwen3_embedding_mimic_config.yaml \
  --model_checkpoint /home/ag619/experiments/Results/mimic-baselines/qwen3-embedding-8B/checkpoints/checkpoint-27000


echo "Model evaluation finished."
deactivate
