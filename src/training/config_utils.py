"""
Config loading with ``extends`` support, deep-merge, and a dataset/experiment builder.

Usage in YAML
-------------
::

    extends:
      - ../shared/lora.yaml

    model:
      train_lora: true   # overrides anything from parents

Paths in ``extends`` are resolved relative to the child config file.

Machine-specific roots live in ``configs/local.yaml`` (gitignored).
Committed experiment YAMLs use placeholders such as ``{results_root}``.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import yaml

from src.experiment_map import (
    CPRD_TASKS,
    DATASETS,
    DEFAULT_TASK,
    EHRSHOT_TASKS,
    MIMIC_TASKS,
    MODEL_EXPERIMENTS,
    MODELS,
    dataset_allowed,
    get_spec,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIGS_ROOT = os.path.join(REPO_ROOT, "configs")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_extends_path(parent_path: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(os.path.dirname(parent_path), rel))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config, recursively resolving ``extends`` lists."""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    extends = raw.pop("extends", None) or []
    if isinstance(extends, str):
        extends = [extends]

    merged: Dict[str, Any] = {}
    for parent_rel in extends:
        parent_path = _resolve_extends_path(config_path, parent_rel)
        parent_cfg = load_config(parent_path)
        merged = deep_merge(merged, parent_cfg)

    return deep_merge(merged, raw)


def _flatten(obj: Any, prefix: str = "") -> Dict[str, str]:
    """Flatten nested dicts to dotted keys for ``{mimic.data_dir}`` substitution."""
    out: Dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(_flatten(value, dotted))
            elif value is not None and not isinstance(value, (list, tuple)):
                out[dotted] = str(value)
                if not prefix:
                    out[str(key)] = str(value)
    elif obj is not None and not isinstance(obj, (list, tuple)):
        if prefix:
            out[prefix] = str(obj)
    return out


def load_local_config(configs_root: Optional[str] = None) -> Dict[str, Any]:
    configs_root = configs_root or CONFIGS_ROOT
    path = os.path.join(configs_root, "local.yaml")
    if not os.path.exists(path):
        example = os.path.join(configs_root, "local.yaml.example")
        raise FileNotFoundError(
            f"Missing {path}. Copy {example} to configs/local.yaml "
            "and fill in machine-specific paths."
        )
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"local.yaml root must be a mapping: {path}")
    return raw


def _substitute_placeholders(obj: Any, mapping: Dict[str, str]) -> Any:
    """Replace ``{key}`` in string leaves. Longer keys first so prefixes do not clash."""
    if isinstance(obj, dict):
        return {k: _substitute_placeholders(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_placeholders(v, mapping) for v in obj]
    if isinstance(obj, str):
        ordered = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
        for key, value in ordered:
            obj = obj.replace("{" + key + "}", value)
        return obj
    return obj


def _list_yaml_stems(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        f[:-5]
        for f in os.listdir(directory)
        if f.endswith(".yaml") and not f.startswith(".")
    )


def _apply_text_embedder_override(cfg, experiment, dataset, task, results_root) -> None:
    """Keep a non-default phrase embedder off the shared Qwen cache and result dirs."""
    if experiment != "text_embedding":
        return
    tag = (os.environ.get("TEHR_EMBEDDER_TAG") or "").strip()
    if not tag:
        return
    if not tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"TEHR_EMBEDDER_TAG must be alphanumeric (got {tag!r}).")
    model_name = (os.environ.get("TEHR_EMBEDDER_MODEL") or "").strip()
    if not model_name:
        raise ValueError("TEHR_EMBEDDER_TAG is set but TEHR_EMBEDDER_MODEL is empty.")
    emb_dir = os.path.join(results_root, f"{dataset}-text_embedding-{tag}")
    embedder = cfg.setdefault("embedder", {})
    embedder["model_name"] = model_name
    embedder["precomputed_embeddings_path"] = os.path.join(emb_dir, "phrase_embeddings.pt")
    task_name = task or "na"
    cfg.setdefault("training", {})["output_dir"] = os.path.join(
        results_root, f"{dataset}-text_embedding-{tag}-{task_name}"
    )
    cfg["name"] = f"{dataset}_text_embedding_{tag}_{task_name}"
    cfg.setdefault("wandb", {})["run_name"] = f"text-embedding-{tag}-{dataset}-{task_name}"
    print(
        f"  - Phrase embedder {tag}: {model_name}\n"
        f"  - Embeddings: {embedder['precomputed_embeddings_path']}",
        flush=True,
    )


def build_config(
    dataset: str,
    experiment: str,
    model: Optional[str] = None,
    task: Optional[str] = None,
    configs_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge shared + dataset + experiment + model + task, then fill paths from
    ``configs/local.yaml``.
    """
    configs_root = configs_root or CONFIGS_ROOT
    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected: {DATASETS}")

    try:
        spec = get_spec(experiment)
    except KeyError as exc:
        raise ValueError(str(exc)) from None

    if not dataset_allowed(experiment, dataset):
        spec = get_spec(experiment)
        allowed = spec.allowed_datasets or (("mimic",) if spec.mimic_only else DATASETS)
        raise ValueError(
            f"Experiment '{experiment}' cannot run on dataset '{dataset}'. "
            f"Allowed: {', '.join(allowed)}"
        )

    if spec.needs_model and not model:
        raise ValueError(
            f"Experiment '{experiment}' requires a model ({', '.join(MODELS)})."
        )
    if spec.allowed_models is not None and model not in spec.allowed_models:
        raise ValueError(
            f"Experiment '{experiment}' only supports: "
            f"{', '.join(spec.allowed_models)}."
        )

    if task is None and not spec.task_optional:
        task = DEFAULT_TASK[dataset]

    if dataset == "mimic" and task is not None and task not in MIMIC_TASKS:
        raise ValueError(f"Unknown MIMIC task '{task}'. Expected: {MIMIC_TASKS}")
    if dataset == "cprd" and task is not None and task not in CPRD_TASKS:
        raise ValueError(f"Unknown CPRD task '{task}'. Expected: {CPRD_TASKS}")
    if dataset == "ehrshot" and task is not None and task not in EHRSHOT_TASKS:
        raise ValueError(
            f"Unknown EHRSHOT task '{task}'. Expected: {EHRSHOT_TASKS}"
        )

    local = load_local_config(configs_root)
    local.setdefault("repo_root", REPO_ROOT)

    exp_path = os.path.join(configs_root, "experiments", f"{experiment}.yaml")
    if not os.path.exists(exp_path):
        available = _list_yaml_stems(os.path.join(configs_root, "experiments"))
        raise FileNotFoundError(
            f"Unknown experiment '{experiment}'. Expected: {exp_path}\n"
            f"Available: {available}"
        )

    with open(exp_path, "r") as f:
        exp_leaf = yaml.safe_load(f) or {}
    exp_model_leaf = dict(exp_leaf.get("model") or {}) if isinstance(exp_leaf, dict) else {}
    exp_training_leaf = (
        dict(exp_leaf.get("training") or {}) if isinstance(exp_leaf, dict) else {}
    )

    cfg: Dict[str, Any] = {}
    repro_path = os.path.join(configs_root, "shared", "reproducibility.yaml")
    if os.path.exists(repro_path):
        cfg = load_config(repro_path)

    dataset_path = os.path.join(configs_root, "datasets", f"{dataset}.yaml")
    dataset_cfg = load_config(dataset_path)
    cfg = deep_merge(cfg, dataset_cfg)
    cfg = deep_merge(cfg, load_config(exp_path))

    if model:
        model_path = os.path.join(configs_root, "models", f"{model}.yaml")
        if not os.path.exists(model_path):
            available = _list_yaml_stems(os.path.join(configs_root, "models"))
            raise FileNotFoundError(
                f"Unknown model '{model}'. Expected: {model_path}\nAvailable: {available}"
            )
        allowed_exps = MODEL_EXPERIMENTS.get(model)
        if allowed_exps is not None and experiment not in allowed_exps:
            raise ValueError(
                f"Model '{model}' only supports: {', '.join(allowed_exps)}."
            )
        model_file = load_config(model_path)
        model_keys = set((model_file.get("model") or {}))
        cfg = deep_merge(cfg, model_file)
        # Experiment model keys fill defaults. A model yaml that sets the same
        # key wins (e.g. medgemma sets load_in_4bit: false).
        if exp_model_leaf:
            model_cfg = cfg.setdefault("model", {})
            for key, value in exp_model_leaf.items():
                if key not in model_keys:
                    model_cfg[key] = copy.deepcopy(value)
        if exp_training_leaf:
            cfg.setdefault("training", {}).update(copy.deepcopy(exp_training_leaf))

    if task:
        task_path = os.path.join(configs_root, "tasks", dataset, f"{task}.yaml")
        if not os.path.exists(task_path):
            available = _list_yaml_stems(os.path.join(configs_root, "tasks", dataset))
            raise FileNotFoundError(
                f"Unknown task '{task}'. Expected: {task_path}\nAvailable: {available}"
            )
        cfg = deep_merge(cfg, load_config(task_path))

    if spec.instruct:
        model_cfg = cfg.setdefault("model", {})
        instruct = model_cfg.get("instruct_model_name")
        if instruct:
            model_cfg["model_name"] = instruct
        if not model_cfg.get("keep_pretrained_checkpoint", False):
            model_cfg["pretrained_checkpoint"] = None

    if spec.clear_pretrained:
        cfg.setdefault("model", {})["pretrained_checkpoint"] = None

    ds_local = local.get(dataset) or {}
    results_root = str(local.get("results_root", "results"))
    labels_cache_root = str(local.get("labels_cache_root", os.path.join(results_root, "labels_cache")))
    seed = int(cfg.get("seed") or (cfg.get("training") or {}).get("seed") or 42)
    seed_dir = f"seed_{seed}"

    stage1_tag = (os.environ.get("TEHR_STAGE1_TAG") or "").strip()
    if stage1_tag and not stage1_tag.replace("_", "").isalnum():
        raise ValueError(
            f"TEHR_STAGE1_TAG must be alphanumeric (got {stage1_tag!r})."
        )

    if spec.uses_stage1_ckpt and model:
        if spec.stage1_experiment:
            ckpt = os.path.join(
                results_root,
                f"{dataset}-{spec.stage1_experiment}-{model}",
                seed_dir,
                "final_model",
            )
        else:
            ckpt = (ds_local.get("checkpoints") or {}).get(model)
            if stage1_tag:
                ckpt = os.path.join(
                    results_root,
                    f"{dataset}-tehr_stage1-{model}-{stage1_tag}",
                    seed_dir,
                    "final_model",
                )
            elif not ckpt:
                ckpt = os.path.join(
                    results_root, f"{dataset}-tehr_stage1-{model}", seed_dir, "final_model"
                )
        cfg.setdefault("model", {})["pretrained_checkpoint"] = ckpt
        if stage1_tag:
            cfg.setdefault("_meta", {})["stage1_tag"] = stage1_tag

    encoder_ckpt = ds_local.get("encoder_mlm_checkpoint") or os.path.join(
        results_root, f"{dataset}-encoder", "models", "pretrained_encoder_stage1_mlm.pt"
    )
    decoder_ckpt = ds_local.get("decoder_ntp_checkpoint") or os.path.join(
        results_root, f"{dataset}-decoder", "models", "pretrained_decoder_stage1_ntp.pt"
    )
    text_emb_path = ds_local.get("text_embedding_path") or os.path.join(
        results_root, f"{dataset}-text_embedding", "phrase_embeddings.pt"
    )

    cfg.setdefault("_meta", {})
    cfg["_meta"]["dataset"] = dataset
    cfg["_meta"]["experiment"] = experiment
    if model:
        cfg["_meta"]["model"] = model
    if task:
        cfg["_meta"]["task"] = task

    mapping = _flatten(local)
    mapping["results_root"] = results_root
    mapping["labels_cache_root"] = labels_cache_root
    mapping["repo_root"] = str(local.get("repo_root", REPO_ROOT))
    mapping["dataset"] = dataset
    mapping["experiment"] = experiment
    mapping["model"] = model or "na"
    mapping["task"] = task or "na"
    mapping["encoder_mlm_checkpoint"] = str(encoder_ckpt)
    mapping["decoder_ntp_checkpoint"] = str(decoder_ckpt)
    mapping["text_embedding_path"] = str(text_emb_path)

    cfg = _substitute_placeholders(cfg, mapping)
    _apply_text_embedder_override(cfg, experiment, dataset, task, results_root)

    training = cfg.setdefault("training", {})
    task_name = (cfg.get("task") or {}).get("name") or task
    if task_name:
        surface = (cfg.get("data") or {}).get("surface_form", "text")
        cache_parts = [str(task_name)]
        if surface and surface != "text":
            cache_parts.append(f"surface_{surface}")
        if (cfg.get("data") or {}).get("drop_time"):
            cache_parts.append("no_time")
        if (cfg.get("data") or {}).get("drop_values"):
            cache_parts.append("no_value")
        cache_subdir = os.path.join(*cache_parts)
        training.setdefault(
            "data_cache_dir",
            os.path.join(labels_cache_root, dataset, cache_subdir),
        )

    overlay = local.get("training")
    if isinstance(overlay, dict) and overlay:
        cfg["training"] = deep_merge(cfg.get("training") or {}, overlay)

    # EHRSHOT dataset training wins last (sparse classifier evals). Experiment
    # YAMLs and local.yaml would otherwise restore the 7-point schedule.
    # Skip for LM continued pretraining: HuggingFace requires save_steps to be
    # a multiple of eval_steps when load_best_model_at_end is set, and stage 1
    # keeps save_steps=1000 / eval_steps=500.
    _stage1 = experiment in {"tehr_stage1", "tehr_stage1_v2", "event_token_stage1"}
    if dataset == "ehrshot" and not _stage1:
        ds_training = dataset_cfg.get("training")
        if isinstance(ds_training, dict) and ds_training:
            cfg["training"] = deep_merge(cfg.get("training") or {}, ds_training)
    elif dataset == "ehrshot" and _stage1:
        # Dataset YAML still merged earlier; drop classifier-only keys.
        (cfg.get("training") or {}).pop("eval_epoch_fractions", None)

    # EHRSHOT labels are bags over long histories, not a fixed endpoint token.
    # Keep last_token on MIMIC. Write mean-pool runs next to the last-token
    # dirs so those numbers stay on disk.
    _ehrshot_mean_pool = {
        "tehr_stage2",
        "ablation_cpt_frozen",
        "ablation_no_time",
        "ablation_no_value",
        "event_token_stage2",
    }
    if dataset == "ehrshot" and experiment in _ehrshot_mean_pool:
        cfg.setdefault("model", {})["pooling"] = "mean"
        training = cfg.setdefault("training", {})
        out = training.get("output_dir")
        if isinstance(out, str) and out and not out.rstrip("/").endswith("-mean"):
            training["output_dir"] = out.rstrip("/") + "-mean"
        wb = cfg.setdefault("wandb", {})
        run_name = wb.get("run_name")
        if isinstance(run_name, str) and run_name and not run_name.endswith("-mean"):
            wb["run_name"] = run_name + "-mean"

    # Score a tagged stage-1 recipe (e.g. TEHR_STAGE1_TAG=v2) without
    # clobbering the default CPT stage-2 dirs.
    if stage1_tag and spec.uses_stage1_ckpt and not spec.stage1_experiment:
        training = cfg.setdefault("training", {})
        out = training.get("output_dir")
        suffix = f"-{stage1_tag}"
        if isinstance(out, str) and out and not out.rstrip("/").endswith(suffix):
            training["output_dir"] = out.rstrip("/") + suffix
        wb = cfg.setdefault("wandb", {})
        run_name = wb.get("run_name")
        if isinstance(run_name, str) and run_name and not run_name.endswith(suffix):
            wb["run_name"] = run_name + suffix

    data = cfg.get("data") or {}
    for key, value in list(data.items()):
        if isinstance(value, str) and value and not os.path.isabs(value) and (
            key.endswith("_filepath") or key.endswith("_dir") or key.endswith("_root")
            or key in {"data_dir", "lookup_dir", "aces_label_dir", "labels_filepath"}
        ):
            data[key] = os.path.normpath(os.path.join(REPO_ROOT, value))

    # mistral.yaml sets max_length 12000 and wins the merge. Event-token runs
    # keep that value as data.tehr_max_length (the text window) and train at
    # event_seq_length.
    if experiment in {"event_token_stage1", "event_token_stage2", "llm_phrase"}:
        model_cfg = cfg.setdefault("model", {})
        cap = model_cfg.get("event_seq_length")
        if cap is None:
            raise ValueError(
                f"{experiment} requires model.event_seq_length "
                "(the event-token context cap)."
            )
        model_cfg["max_length"] = int(cap)

    return cfg


def apply_smoke_config(config: Dict[str, Any], eval_only: bool = False) -> Dict[str, Any]:
    """Redirect a resolved config to ``{results_root}/_smoke/...`` for 1-step / tiny evals.

    Does not touch production ``seed_42/43/44`` trees. Honoured when ``TEHR_SMOKE=1``.
    """
    meta = config.setdefault("_meta", {})
    meta["smoke"] = True
    experiment = str(meta.get("experiment") or "")
    training = config.setdefault("training", {})
    orig = training.get("output_dir") or config.get("output_dir")
    if not orig:
        raise ValueError("apply_smoke_config: config has no training.output_dir")
    orig = os.path.abspath(orig)
    results_root = os.path.dirname(orig)
    tag = os.path.basename(orig.rstrip(os.sep))
    smoke_dir = os.path.join(results_root, "_smoke", tag)
    training["output_dir"] = smoke_dir
    config["output_dir"] = smoke_dir
    config.setdefault("wandb", {})["enabled"] = False
    training["warmup_steps"] = 0
    training["overwrite_output_dir"] = True
    training["eval_max_samples"] = int(training.get("eval_max_samples") or 8)
    debug = config.setdefault("debug", {})
    debug["enabled"] = True
    debug["max_samples"] = int(debug.get("max_samples") or 8)
    clmbr = config.setdefault("clmbr", {})
    clmbr["max_labels"] = 16
    # Previous all-negative 16-row caches must not be reused on a smoke retry.
    clmbr["overwrite_embeddings"] = True
    if eval_only:
        training["skip_final_eval"] = False
    else:
        training["max_steps"] = 1
        training["epochs"] = 1
        training["eval_epoch_fractions"] = False
        training["skip_final_eval"] = experiment != "zero_shot"
        training["smoke_max_patients"] = 32
        config.setdefault("model", {})["n_estimators"] = 2
        config.setdefault("hyperparameter_search", {})["enabled"] = False
        pretrain = config.setdefault("pretrain", {})
        pretrain["force_pretrain"] = False
    return config


def save_config(config: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    return path


def _cli():
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Experiment config builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build and optionally write a resolved config")
    build.add_argument("--dataset", required=True, choices=DATASETS)
    build.add_argument("--experiment", required=True)
    build.add_argument("--model", default=None)
    build.add_argument("--task", default=None)
    build.add_argument("--out", default=None, help="Write resolved YAML here")
    build.add_argument("--print", action="store_true", dest="do_print")

    load = sub.add_parser("load", help="Load a YAML path with extends resolution")
    load.add_argument("path")
    load.add_argument("--print", action="store_true", dest="do_print")

    args = parser.parse_args()

    if args.cmd == "build":
        cfg = build_config(
            dataset=args.dataset,
            experiment=args.experiment,
            model=args.model,
            task=args.task,
        )
        if args.out:
            save_config(cfg, args.out)
            print(args.out)
        if args.do_print or not args.out:
            pprint.pprint(cfg)
    elif args.cmd == "load":
        pprint.pprint(load_config(args.path))


if __name__ == "__main__":
    _cli()
