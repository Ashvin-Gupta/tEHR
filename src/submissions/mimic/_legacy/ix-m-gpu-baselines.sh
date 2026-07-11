#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ix-m-gpu-baselines-encoder-transformer-hospital-readmission.log
#SBATCH --job-name=ix-m-gpu-baselines-encoder-transformer-hospital-readmission



set -e

# Set the base directory for your project
BASE_DIR="/home/ag619/TextCancEHR2"

export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

# --- Execute from Project Root ---
# Change to the base directory before running the python command
cd "${BASE_DIR}"

echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

MASTER_PORT=$(get_free_port)
export MASTER_PORT
# export MASTER_PORT=47001

echo "Using MASTER_PORT=$MASTER_PORT for torchrun"


# python -m src.pipelines.run_baseline --baseline encoder_transformer --config_filepath src/configs/baseline_encoder_transformer_mimic_config.yaml
# python -m src.pipelines.run_baseline --baseline clinicalbert --config_filepath src/configs/baseline_clinicalbert_mimic_config.yaml
python -m src.pipelines.run_baseline --baseline pure_llm --config_filepath src/configs/baseline_pure_llm_config.yaml

echo "Pipeline finished."
conda deactivate



#SBATCH --output=logs/ix-m-icu-mortality-te.log
#SBATCH --job-name=ix-m-icu-mortality-te