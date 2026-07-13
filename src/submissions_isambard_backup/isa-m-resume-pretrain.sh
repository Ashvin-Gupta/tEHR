#!/bin/bash
#SBATCH --job-name=isa-m-resume-pretrain
#SBATCH --output=logs/isa-m-resume-pretrain_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00

set -e

BASE_DIR="${HOME}/TextCancEHR2"

# Update to the checkpoint you want to resume from.
CHECKPOINT_PATH="${HOME}/Pretrain-Mistral-7B-Mimic/checkpoint-14000"

# --- Paths ---
mkdir -p "${HOME}/TextCancEHR2/logs"

# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Starting resume-pretrain from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source ~/miniforge3/bin/activate
conda activate meds_train_clean_311

TEMP_CONFIG=$(mktemp /tmp/isa_config_XXXXXX.yaml)
trap "rm -f ${TEMP_CONFIG}" EXIT
envsubst < "${BASE_DIR}/src/configs_isambard/mimic_resume_pretrain_config.yaml" > "${TEMP_CONFIG}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
MASTER_PORT=$(get_free_port)
export MASTER_PORT
echo "Using MASTER_PORT=$MASTER_PORT for torchrun"

torchrun --nproc_per_node=4 src/pipelines/resume_pretrain.py \
    --config_filepath "${TEMP_CONFIG}" \
    --checkpoint_path "${CHECKPOINT_PATH}"

echo "Pipeline finished."
conda deactivate
