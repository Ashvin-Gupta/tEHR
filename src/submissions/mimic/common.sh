#!/bin/bash
# Shared environment setup for MIMIC SLURM jobs.
# Sourced by src/submissions/mimic/run.sh

BASE_DIR="/home/ag619/TextCancEHR2"
export WANDB_API_KEY="${WANDB_API_KEY:-REDACTED}"
export HF_TOKEN="REDACTED"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meds_train_clean

cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
export MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"
