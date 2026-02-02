#!/bin/bash
#$ -cwd                 
#$ -pe smp 8
#$ -l h_rt=1:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR2/HPC_Pretrain/logo/
#$ -e /data/home/qc25022/TextCancEHR2/HPC_Pretrain/loge/

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

python -m src.pipelines.llm_pretrain --config_filepath src/configs/llm_pretrain.yaml 
# torchrun --nproc_per_node=2 src/pipelines/llm_pretrain.py --config_filepath src/configs/llm_pretrain.yaml

echo "Pipeline finished."
deactivate

