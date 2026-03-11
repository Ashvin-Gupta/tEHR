#!/bin/bash
#SBATCH -n 4
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=1G
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Base/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Base/loge/%x.e%j

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
module load python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate
echo "Environment setup complete."
# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python -m src.analysis.cluster_from_embeddings

echo "Pipeline finished."
deactivate