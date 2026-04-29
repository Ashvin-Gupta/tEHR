#!/bin/bash                 
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --gres=gpu:2                  # Request 1 GPU of any type
#SBATCH --cpus-per-gpu=24                 # 12 CPU cores 
#SBATCH -t 240:0:0                        # 1 hour runtime
#SBATCH --mem-per-cpu=7500M             # Total memory 90G (approx 12 * 7500M)
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Pretrain/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Pretrain/loge/%x.e%j


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

# python -m src.pipelines.llm_pretrain --config_filepath src/configs/llm_pretrain.yaml 
torchrun --nproc_per_node=12 src/pipelines/llm_pretrain.py --config_filepath src/configs/llm_pretrain.yaml

echo "Pipeline finished."
deactivate

