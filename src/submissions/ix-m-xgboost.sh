#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/ix-m-xgboost.log
#SBATCH --job-name=ix-m-xgboost

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

python -m src.pipelines.run_baseline --baseline xgboost_bow --config_filepath src/configs/baseline_xgboost_bow_config.yaml 

echo "Pipeline finished."
conda deactivate
