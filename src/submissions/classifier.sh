#!/bin/bash
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --gres=gpu:2                   # Request 1 GPU of any type
#SBATCH --cpus-per-gpu=12               # 12 CPU cores 
#SBATCH -t 240:0:0                        # 1 hour runtime
#SBATCH --mem-per-cpu=7500M             # Total memory 90G (approx 12 * 7500M)
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Classifier/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Classifier/loge/%x.e%j

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

# Avoid torch.distributed "EADDRINUSE" when multiple jobs share a node.
# Pick a stable, job-unique port (and array-task-unique if using arrays).
JOB_PORT_SEED="${SLURM_JOB_ID:-0}${SLURM_ARRAY_TASK_ID:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (10#${JOB_PORT_SEED} % 20000)))}"

# python -m src.pipelines.finetune_llm_classifier --config_filepath src/configs/classification_config.yaml
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  src/pipelines/finetune_llm_classifier.py \
  --config_filepath src/configs/classification_config2.yaml

echo "Pipeline finished."
deactivate

