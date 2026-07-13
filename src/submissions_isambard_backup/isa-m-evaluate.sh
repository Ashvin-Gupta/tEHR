#!/bin/bash
#SBATCH --job-name=isa-m-evaluate
#SBATCH --output=logs/isa-m-evaluate_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00

set -e

BASE_DIR="${HOME}/TextCancEHR2"

# --- Update these before submitting ---
EVAL_CONFIG_TEMPLATE="${BASE_DIR}/src/configs_isambard/mimic_classifier.yaml"
MODEL_CHECKPOINT="${PROJECTDIR}/TextCancEHR2/experiments/Results/mimic-icu-mortality/seed_XXXXX/checkpoint-XXXX"

# --- Paths ---
mkdir -p "${HOME}/TextCancEHR2/logs"

# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Starting evaluation from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source ~/miniforge3/bin/activate
conda activate meds_train_clean_311

TEMP_CONFIG=$(mktemp /tmp/isa_config_XXXXXX.yaml)
trap "rm -f ${TEMP_CONFIG}" EXIT
envsubst < "${EVAL_CONFIG_TEMPLATE}" > "${TEMP_CONFIG}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
MASTER_PORT=$(get_free_port)
export MASTER_PORT
echo "Using MASTER_PORT=$MASTER_PORT for torchrun"

torchrun --nproc_per_node=4 -m src.evaluations.evaluate_model \
    --config_filepath "${TEMP_CONFIG}" \
    --model_checkpoint "${MODEL_CHECKPOINT}"

echo "Model evaluation finished."
conda deactivate
