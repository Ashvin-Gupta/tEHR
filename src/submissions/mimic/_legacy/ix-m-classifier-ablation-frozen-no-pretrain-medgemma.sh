#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ix-m-ablation-frozen-no-pretrain-medgemma.log
#SBATCH --job-name=ix-m-abl-fnp-mg

set -e

BASE_DIR="/home/ag619/TextCancEHR2"
export WANDB_API_KEY="REDACTED"
# export HF_TOKEN="hf_..."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

cd "${BASE_DIR}"
echo "Starting MedGemma ablation (frozen, no pretrain) from: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Ablation 5c (MedGemma): no pretrain, frozen MedGemma, train head only
python -m src.pipelines.run_baseline \
    --baseline frozen_llm_linear \
    --config_filepath src/configs/baseline_frozen_llm_linear_mimic_medgemma_config.yaml

echo "Pipeline finished."
conda deactivate
