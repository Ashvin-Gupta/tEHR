#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=24:0:0
#$ -l h_vmem=4G
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR2/HPC_Base/logo/
#$ -e /data/home/qc25022/TextCancEHR2/HPC_Base/loge/

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

python -m src.pipelines.run_baseline_clinicalbert --config_filepath src/configs/baseline_clinicalbert_config.yaml 

echo "Pipeline finished."
deactivate