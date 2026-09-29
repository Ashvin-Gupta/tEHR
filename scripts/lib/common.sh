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
elif [[ -n "${VENV_ACTIVATE:-}" ]]; then
    if [[ ! -f "${VENV_ACTIVATE}" ]]; then
        echo "ERROR: VENV_ACTIVATE not found: ${VENV_ACTIVATE}"
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${VENV_ACTIVATE}"
fi

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/configs/_resolved"

# Triton compiles cuda_utils.c against Python.h. python3.11-libs on this
# cluster does not ship headers; use a user-extracted python3.11-devel tree.
if [[ -z "${PYTHON_INCLUDE_DIR:-}" ]]; then
    _pyh_cand="${HOME}/.local/opt/python3.11-devel/usr/include/python3.11"
    if [[ -f "${_pyh_cand}/Python.h" ]]; then
        PYTHON_INCLUDE_DIR="${_pyh_cand}"
    fi
    unset _pyh_cand
fi
if [[ -n "${PYTHON_INCLUDE_DIR:-}" && -f "${PYTHON_INCLUDE_DIR}/Python.h" ]]; then
    export PYTHON_INCLUDE_DIR
    export CPATH="${PYTHON_INCLUDE_DIR}${CPATH:+:${CPATH}}"
    export C_INCLUDE_PATH="${PYTHON_INCLUDE_DIR}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
    echo "Triton Python.h: ${PYTHON_INCLUDE_DIR}/Python.h"
fi

# sitecustomize.py patches sysconfig include so Triton's gcc -I finds Python.h.
export PYTHONPATH="${REPO_ROOT}/scripts/lib:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
eval "$(python -m src.experiment_map emit_bash)"

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
export MASTER_PORT="${MASTER_PORT:-$(get_free_port)}"
