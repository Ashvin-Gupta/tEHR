#!/bin/bash
#SBATCH --job-name=isa-m-xgboost
#SBATCH --output=logs/isa-m-xgboost_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -e

BASE_DIR="${HOME}/TextCancEHR2"

# --- Paths ---
mkdir -p "${HOME}/TextCancEHR2/logs"

# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Starting XGBoost baseline from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source ~/miniforge3/bin/activate
conda activate meds_train_clean_311

TEMP_CONFIG=$(mktemp /tmp/isa_config_XXXXXX.yaml)
trap "rm -f ${TEMP_CONFIG}" EXIT
envsubst < "${BASE_DIR}/src/configs_isambard/baseline_xgboost_bow_config.yaml" > "${TEMP_CONFIG}"

python -m src.pipelines.run_baseline \
    --baseline xgboost_bow \
    --config_filepath "${TEMP_CONFIG}"

echo "Pipeline finished."
conda deactivate
