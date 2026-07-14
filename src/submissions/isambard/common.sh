#!/bin/bash
# Shared environment setup for Isambard MIMIC SLURM jobs.
# Sourced by src/submissions/isambard/run.sh and run_cpu.sh

BASE_DIR="${HOME}/TextCancEHR2"
export WANDB_API_KEY="${WANDB_API_KEY:-REDACTED}"
export HF_TOKEN="${HF_TOKEN:-REDACTED}"

# Prefer the aarch64 Unsloth stack used for training on Isambard.
# `source .../activate` reads the current shell's "$@", so clear job args
# (e.g. mistral icu_mortality) first or activate treats them as env names.
__conda_saved_args=("$@")
set --
source ~/miniforge3/bin/activate
conda activate meds_train_clean_311
set -- "${__conda_saved_args[@]}"
unset __conda_saved_args

cd "${BASE_DIR}"
mkdir -p "${BASE_DIR}/logs"
export PYTHONPATH="${BASE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
export MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"
