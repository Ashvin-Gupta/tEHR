#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/ix-m-debug.log
#SBATCH --job-name=ix-m-debug

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

# python -m src.debug.debug_aces --config_filepath src/configs/mimic_debug.yaml
# python -m tests.check_mimic_leakage
python src/analysis/summarize_mimic_labels.py \
    --labels-dir /home/ag619/EHR-JEPA-Data/labels

echo "Pipeline finished."
conda deactivate
