#!/bin/bash
#$ -cwd                 
#$ -pe smp 12
#$ -l h_rt=24:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -l cluster=andrena
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

python -m src.pipelines.run_baseline_pretrained_llm_linear --config_filepath src/configs/baseline_pretrained_llm_linear_config.yaml 

echo "Pipeline finished."
deactivate