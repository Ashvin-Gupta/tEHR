#!/bin/bash
#SBATCH --job-name=isa-m-resume-classifier
#SBATCH --output=logs/isa-m-resume-classifier_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00

set -e

TASK="${TASK:-hospital_readmission}"
SEED="${SEED:-1849542118}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECTDIR}/TextCancEHR2/experiments/Results/mimic-hospital-readmission/seed_1849542118/checkpoint-1600}"

BASE_DIR="${HOME}/TextCancEHR2"

mkdir -p "${BASE_DIR}/logs"
cd "${BASE_DIR}"
echo "Starting experiment from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export WANDB_API_KEY="REDACTED"

source ~/miniforge3/bin/activate
conda activate meds_train_clean_311

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
    --task "${TASK}" \
    --seed "${SEED}" \
    --init_from_checkpoint "${CHECKPOINT_PATH}"

echo "Pipeline finished."
conda deactivate
