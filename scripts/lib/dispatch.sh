# Shared GPU experiment routing. Sourced by scripts/site_a/run.sh and site_b/run.sh
# after local.env has been loaded and the config has been resolved.
#
# Experiment lists (ALL_EXPERIMENTS, LLM_DDP_EXPERIMENTS, …) are loaded in
# common.sh via ``python -m src.experiment_map emit_bash``.
#
# Python entry point is always src/pipelines/run.py. This shell only chooses
# torchrun vs python, pins a GPU when 4-bit training cannot use DDP, and
# precomputes phrase embeddings for text_embedding.

is_in() {
    local needle="$1"; shift
    local x
    for x in "$@"; do
        [[ "$x" == "$needle" ]] && return 0
    done
    return 1
}

require_model() {
    if [[ -z "${MODEL}" ]]; then
        echo "ERROR: ${EXPERIMENT} requires a model (${KNOWN_MODELS[*]})"
        exit 1
    fi
}

run_torch() {
    torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
        src/pipelines/run.py --config_filepath "${RESOLVED_CFG}" \
        "${PIPELINE_EXTRA_ARGS[@]}" "$@"
}

run_python() {
    python -m src.pipelines.run --config_filepath "${RESOLVED_CFG}" \
        "${PIPELINE_EXTRA_ARGS[@]}" "$@"
}

pin_one_gpu() {
    local first="${CUDA_VISIBLE_DEVICES:-}"
    first="${first%%,*}"
    export CUDA_VISIBLE_DEVICES="${first:-0}"
    echo "pinning CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}

maybe_precompute_text_embeddings() {
    python - <<'PY'
import os, sys, yaml
path = os.environ["RESOLVED_CFG"]
with open(path) as f:
    cfg = yaml.safe_load(f)
emb = (cfg.get("embedder") or {}).get("precomputed_embeddings_path")
if not emb:
    sys.exit("ERROR: embedder.precomputed_embeddings_path missing from resolved config")
print(emb)
PY
}

dispatch_experiment() {
    if [[ -n "${EVAL_ONLY}" ]] && ! is_in "${EXPERIMENT}" "${EVAL_SUPPORTED[@]}"; then
        echo "ERROR: 'evaluate' is not supported for experiment '${EXPERIMENT}'."
        echo "  Supported: ${EVAL_SUPPORTED[*]}"
        exit 1
    fi
    if is_in "${EXPERIMENT}" "${NEEDS_MODEL[@]}"; then
        require_model
    fi
    if ! is_in "${EXPERIMENT}" "${ALL_EXPERIMENTS[@]}"; then
        echo "ERROR: unknown experiment '${EXPERIMENT}'"
        echo "Known: ${ALL_EXPERIMENTS[*]}"
        exit 1
    fi

    local extra=()
    if [[ -n "${EVAL_ONLY}" ]]; then
        echo "evaluate: scoring newest checkpoint (no training)"
        extra+=(--eval_only)
        [[ -n "${SEED:-}" ]] && extra+=(--seed "${SEED}")
        [[ -n "${CHECKPOINT:-}" ]] && extra+=(--checkpoint_path "${CHECKPOINT}")
        [[ -n "${CKPT_SELECT:-}" ]] && extra+=(--checkpoint_selection "${CKPT_SELECT}")
        case "${OVERWRITE:-}" in
            1|true|yes|on) extra+=(--overwrite) ;;
        esac
        case "${EVAL_WANDB:-}" in
            1|true|yes|on) extra+=(--wandb) ;;
        esac
    fi

    if [[ "${EXPERIMENT}" == "text_embedding" ]]; then
        EMBEDDINGS_PATH="$(maybe_precompute_text_embeddings)"
        if [[ ! -f "${EMBEDDINGS_PATH}" ]]; then
            echo "Precomputing phrase embeddings at ${EMBEDDINGS_PATH}"
            mkdir -p "$(dirname "${EMBEDDINGS_PATH}")"
            DATA_DIR="$(python -c "import yaml,os; c=yaml.safe_load(open(os.environ['RESOLVED_CFG'])); print(c['data']['data_dir'])")"
            VOCAB="$(python -c "import yaml,os; c=yaml.safe_load(open(os.environ['RESOLVED_CFG'])); print(c['data']['vocab_filepath'])")"
            MODEL_NAME="$(python -c "import yaml,os; c=yaml.safe_load(open(os.environ['RESOLVED_CFG'])); print(c['embedder']['model_name'])")"
            PRE_ARGS=(
                --dataset "${DATASET}"
                --vocab_filepath "${VOCAB}"
                --data_dir "${DATA_DIR}"
                --model_name "${MODEL_NAME}"
                --output_path "${EMBEDDINGS_PATH}"
                --batch_size 256
                --max_length 128
                --device cuda
            )
            if [[ "${DATASET}" == "mimic" ]]; then
                LOOKUP="$(python -c "import yaml,os; c=yaml.safe_load(open(os.environ['RESOLVED_CFG'])); print(c['data']['lookup_dir'])")"
                PRE_ARGS+=(--lookup_dir "${LOOKUP}")
            else
                for key in medical_lookup_filepath lab_lookup_filepath region_lookup_filepath time_lookup_filepath; do
                    val="$(python -c "import yaml,os; c=yaml.safe_load(open(os.environ['RESOLVED_CFG'])); print(c['data']['${key}'])")"
                    PRE_ARGS+=(--"${key}" "${val}")
                done
            fi
            python -m src.pipelines.precompute_token_embeddings "${PRE_ARGS[@]}"
        else
            echo "Phrase embeddings already exist at ${EMBEDDINGS_PATH}"
        fi
    fi

    if is_in "${EXPERIMENT}" "${LLM_DDP_EXPERIMENTS[@]}"; then
        run_torch "${extra[@]}"
    elif is_in "${EXPERIMENT}" "${ONE_GPU_EXPERIMENTS[@]}"; then
        pin_one_gpu
        run_python "${extra[@]}"
    else
        run_python "${extra[@]}"
    fi
}
