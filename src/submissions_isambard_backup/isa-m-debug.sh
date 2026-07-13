#!/bin/bash
#SBATCH --job-name=isa-m-debug
#SBATCH --output=logs/isa-m-debug_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -e

BASE_DIR="${HOME}/TextCancEHR2"

# --- Paths ---
mkdir -p "${HOME}/TextCancEHR2/logs"

# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Starting leakage check from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# --- Environment Setup ---
source ~/miniforge3/bin/activate
conda activate meds_train_clean_311

python -m tests.check_mimic_leakage

echo "Pipeline finished."
conda deactivate
