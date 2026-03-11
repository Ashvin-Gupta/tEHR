#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/ix-m-classifier.log
#SBATCH --job-name=ix-m-classifier

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

# python -m src.pipelines.finetune_llm_classifier --config_filepath src/configs/classification_config.yaml
torchrun --nproc_per_node=2 src/pipelines/finetune_llm_classifier.py --config_filepath src/configs/mimic_classifier.yaml

echo "Pipeline finished."
conda deactivate
