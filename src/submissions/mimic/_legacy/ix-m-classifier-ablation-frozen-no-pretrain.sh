#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ix-m-ablation-frozen-no-pretrain.log
#SBATCH --job-name=ix-m-ablation-fnp

set -e

BASE_DIR="/home/ag619/TextCancEHR2"
export WANDB_API_KEY="REDACTED"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

cd "${BASE_DIR}"
echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Ablation 5c: no pretrain, frozen Mistral, head only
python -m src.pipelines.run_baseline \
    --baseline frozen_llm_linear \
    --config_filepath src/configs/baseline_frozen_llm_linear_mimic_config.yaml

echo "Pipeline finished."
conda deactivate
