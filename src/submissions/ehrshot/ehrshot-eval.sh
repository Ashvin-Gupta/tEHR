#!/bin/bash                 
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --gres=gpu:1                   # Request 1 GPU of any type
#SBATCH --cpus-per-gpu=12                 # 12 CPU cores 
#SBATCH -t 1:0:0                        # 1 hour runtime
#SBATCH --mem-per-cpu=7500M             # Total memory 90G (approx 12 * 7500M)
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Eval/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Eval/loge/%x.e%j


set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

python -m src.evaluations.evaluate_model --config_filepath src/configs/ehrshot_evaluate_config.yaml --model_checkpoint /gpfs/scratch/qc25022/pancreas/experiments/Results/lora-12-month-logistic-refactored-new-time-mistral-7B/seed_3078332855/checkpoint-7372

echo "Pipeline finished."
deactivate

