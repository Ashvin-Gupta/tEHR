# Shared environment for GPU/CPU jobs. Sourced after REPO_ROOT is known.
set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "ERROR: REPO_ROOT is not set"
    exit 1
fi

if [[ -f "${SITE_DIR}/local.env" ]]; then
    # shellcheck disable=SC1091
    source "${SITE_DIR}/local.env"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"

if [[ -n "${CONDA_SETUP:-}" ]]; then
    __conda_saved_args=("$@")
    set --
    # shellcheck disable=SC1090
    source "${CONDA_SETUP}"
    if [[ -n "${CONDA_ENV:-}" ]]; then
        conda activate "${CONDA_ENV}"
    fi
    set -- "${__conda_saved_args[@]}"
    unset __conda_saved_args
fi

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/configs/_resolved"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
eval "$(python -m src.experiment_map emit_bash)"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
export MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"
