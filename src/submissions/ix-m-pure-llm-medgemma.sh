#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ix-m-pure-llm-medgemma.log
#SBATCH --job-name=ix-m-pure-mg

set -e

BASE_DIR="/home/ag619/TextCancEHR2"
export WANDB_API_KEY="REDACTED"
# export HF_TOKEN="hf_..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

cd "${BASE_DIR}"
echo "Starting MedGemma zero-shot prompting from: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Exp 2 (MedGemma): zero-shot logit-based classification
python -m src.pipelines.run_baseline \
    --baseline pure_llm \
    --config_filepath src/configs/baseline_pure_llm_mimic_medgemma_config.yaml

echo "Pipeline finished."
conda deactivate
