#!/bin/bash
#$ -cwd                 
#$ -pe smp 12
#$ -l h_rt=1:0:0
#$ -l h_vmem=7.5G
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -l cluster=andrena
#$ -j n
#$ -o /data/home/qc25022/TextCancEHR2/HPC_Classifier/logo/
#$ -e /data/home/qc25022/TextCancEHR2/HPC_Classifier/loge/

set -e 

# Set the base directory for your project
BASE_DIR="/data/home/qc25022/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
module load intel intel-mpi python
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

python -m src.pipelines.evaluate_model \
  --config_filepath src/configs/classification_config.yaml \
  --model_checkpoint /data/scratch/qc25022/pancreas_MEDS/experiments/lora-6-month-logistic-refactored/checkpoint-5000

echo "Model evaluation finished."
deactivate

