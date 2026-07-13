#!/bin/bash
#SBATCH --job-name=isa-m-cpu-baselines-clinicalbert
#SBATCH --output=logs/isa-m-cpu-baselines-clinicalbert_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e

BASE_DIR="${HOME}/TextCancEHR2"

# --- Paths ---
mkdir -p "${HOME}/TextCancEHR2/logs"

# --- Execute from Project Root ---
cd "${BASE_DIR}"
echo "Starting ClinicalBERT baseline from directory: $(pwd)"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
export WANDB_API_KEY="REDACTED"

# --- Environment Setup ---
source ~/miniforge3/bin/activate
conda activate meds_train_clean_311

TEMP_CONFIG=$(mktemp /tmp/isa_config_XXXXXX.yaml)
trap "rm -f ${TEMP_CONFIG}" EXIT
envsubst < "${BASE_DIR}/src/configs_isambard/baseline_clinicalbert_mimic_config.yaml" > "${TEMP_CONFIG}"

python -m src.pipelines.run_baseline \
    --baseline clinicalbert \
    --config_filepath "${TEMP_CONFIG}"

echo "Pipeline finished."
conda deactivate
