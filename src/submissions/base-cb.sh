#!/bin/bash         
#SBATCH -n 8
#SBATCH -t 48:0:0
#SBATCH --mem-per-cpu=8G
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Base/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Base/loge/%x.e%j

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
module load intel intel-mpi python
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python -m src.pipelines.run_baseline --baseline clinicalbert --config_filepath src/configs/baseline_clinicalbert_config.yaml

echo "Pipeline finished."
deactivate