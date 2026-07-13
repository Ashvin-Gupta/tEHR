#!/bin/bash
#SBATCH --job-name=isa-m-icu-mortality-classifier
#SBATCH --output=logs/isa-m-icu-mortality-classifier_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

set -e

# Task to run — overrides aces_label_dir, output_dir, and wandb run_name in the config.
# Set via --export when submitting, e.g.:
#   sbatch --export=ALL,TASK=icu_mortality        src/submissions_isambard/isa-m-classifier.sh
#   sbatch --export=ALL,TASK=hospital_mortality   src/submissions_isambard/isa-m-classifier.sh
#   sbatch --export=ALL,TASK=hospital_readmission src/submissions_isambard/isa-m-classifier.sh
# Defaults to icu_mortality if TASK is not set.
TASK="${TASK:-icu_mortality}"

BASE_DIR="${HOME}/TextCancEHR2"

# --- Paths ---
mkdir -p "${HOME}/TextCancEHR2/logs"

# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Starting classifier experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source ~/miniforge3/bin/activate
conda activate meds_train_clean

TEMP_CONFIG=$(mktemp /tmp/isa_config_XXXXXX.yaml)
trap "rm -f ${TEMP_CONFIG}" EXIT
envsubst < "${BASE_DIR}/src/configs_isambard/mimic_classifier.yaml" > "${TEMP_CONFIG}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
MASTER_PORT=$(get_free_port)
export MASTER_PORT
echo "Using MASTER_PORT=$MASTER_PORT for torchrun"

torchrun --nproc_per_node=4 src/pipelines/finetune_llm_classifier.py \
    --config_filepath "${TEMP_CONFIG}" \
    --task "${TASK}"

echo "Pipeline finished."
conda deactivate
