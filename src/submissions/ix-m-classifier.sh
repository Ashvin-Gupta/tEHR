#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/ix-m-hospital-mortality-classification.log
#SBATCH --job-name=ix-m-hospital-mortality-classification

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

# python -m src.pipelines.finetune_llm_classifier --config_filepath src/configs/classification_config.yaml
torchrun --nproc_per_node=2 src/pipelines/finetune_llm_classifier.py --config_filepath src/configs/mimic_classifier.yaml

# torchrun --nproc_per_node=3 --master_port="$MASTER_PORT" src/pipelines/finetune_llm_classifier.py \
#   --config_filepath src/configs/mimic_classifier.yaml \
#   --seed 1081433188 \
#   --checkpoint_path /home/ag619/experiments/Results/mimic-hospital-readmission-classification/seed_1081433188/checkpoint-3000

echo "Pipeline finished."
conda deactivate
