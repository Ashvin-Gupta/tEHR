#!/bin/bash
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Classifier/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Classifier/loge/%x.e%j

set -e

BASE_DIR="/data/home/qc25022/TextCancEHR2"
export WANDB_API_KEY="REDACTED"

source /data/home/qc25022/CancEHR-Training/venv/bin/activate
cd "${BASE_DIR}"

echo "Starting EHRSHOT finetuning from: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Optional: resume from an existing classification checkpoint that already
# contains LoRA + classifier head training state.
# Example:
#   CHECKPOINT_PATH="/path/to/seed_123456/checkpoint-2000" sbatch ehrshot-finetune.sh
CHECKPOINT_PATH=/gpfs/scratch/qc25022/pancreas/experiments/Results/lora-12-month-logistic-refactored-new-time-mistral-7B/seed_3078332855/checkpoint-7372"

python -m src.pipelines.finetune_llm_classifier \
  --config_filepath src/configs/ehrshot_finetune_config.yaml \
  ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"}

echo "EHRSHOT finetune pipeline finished."
deactivate
