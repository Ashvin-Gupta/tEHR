#!/bin/bash
#$ -cwd                 
#$ -pe smp 12
#$ -l h_rt=240:0:0
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
cd "${BASE_DIR}"

echo "Starting resume pretraining from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Example usage:
# python -m src.pipelines.resume_pretrain \
#   --config_filepath src/configs/resume_pretrain_config.yaml \
#   --checkpoint_path /data/scratch/qc25022/pancreas/experiments/Pretrain-Qwen3-8B-Pancreas-final/checkpoints/checkpoint-5000

python -m src.pipelines.resume_pretrain \
  --config_filepath src/configs/resume_pretrain_config.yaml \
  --checkpoint_path  /data/scratch/qc25022/pancreas/experiments/Pretrain-Qwen3-8B-Pancreas-final/checkpoint-9000
echo "Resume pretraining finished."
deactivate

