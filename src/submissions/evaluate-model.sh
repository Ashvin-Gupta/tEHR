#!/bin/bash
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --mem-per-cpu=7500M             # Total memory 90G (approx 12 * 7500M)
#SBATCH --gres=gpu:1                    # Request 1 GPU of any type
#SBATCH --cpus-per-task=12              # 12 CPU cores 
#SBATCH -t 4:0:0                        # 1 hour runtime
#SBATCH -o /data/home/qc25022/TextCancEHR2/HPC_Classifier/logo/%x.o%j
#SBATCH -e /data/home/qc25022/TextCancEHR2/HPC_Classifier/loge/%x.e%j

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source /data/home/qc25022/CancEHR-Training/venv/bin/activate

# --- Execute from Project Root ---
cd "${BASE_DIR}"

echo "Starting model evaluation from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Example usage for different baselines:
# 
# For frozen_llm_linear:
# python -m src.pipelines.evaluate_model \
#   --config_filepath src/configs/baseline_frozen_llm_linear_config.yaml \
#   --model_checkpoint /data/scratch/qc25022/pancreas_MEDS/experiments/baselines/frozen_llm_linear/final_model
#
# For pretrained_llm_linear:
# python -m src.pipelines.evaluate_model \
#   --config_filepath src/configs/baseline_pretrained_llm_linear_config.yaml \
#   --model_checkpoint /data/scratch/qc25022/pancreas_MEDS/experiments/baselines/pretrained_llm_linear/final_model
#
# For lora_llm_linear:
# python -m src.pipelines.evaluate_model \
#   --config_filepath src/configs/baseline_lora_llm_linear_config.yaml \
#   --model_checkpoint /data/scratch/qc25022/pancreas_MEDS/experiments/baselines/lora_llm_linear/final_model

# python -m src.pipelines.evaluate_model \
#   --config_filepath src/configs/classification_config.yaml \
#   --model_checkpoint /data/scratch/qc25022/pancreas_MEDS/experiments/lora-3-month-logistic-refactored/final_model

python -m src.pipelines.evaluate_model \
  --config_filepath src/configs/baseline_qwen3_embedding_config.yaml \
  --model_checkpoint /gpfs/scratch/qc25022/pancreas/experiments/Results/baselines/qwen3-embedding-8B-12-months/final_model
  
echo "Model evaluation finished."
deactivate

