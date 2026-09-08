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
        cfg = deep_merge(cfg, load_config(model_path))
        if exp_model_leaf:
            cfg.setdefault("model", {}).update(copy.deepcopy(exp_model_leaf))
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

    if spec.uses_stage1_ckpt and model:
        ckpt = (ds_local.get("checkpoints") or {}).get(model)
        if not ckpt:
            ckpt = os.path.join(
                results_root, f"{dataset}-tehr_stage1-{model}", seed_dir, "final_model"
            )
        cfg.setdefault("model", {})["pretrained_checkpoint"] = ckpt

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

    # EHRSHOT dataset training wins last (sparse evals). Experiment YAMLs and
    # local.yaml eval_batch_size would otherwise restore the 7-point schedule.
    if dataset == "ehrshot":
        ds_training = dataset_cfg.get("training")
        if isinstance(ds_training, dict) and ds_training:
            cfg["training"] = deep_merge(cfg.get("training") or {}, ds_training)

    data = cfg.get("data") or {}
    for key, value in list(data.items()):
        if isinstance(value, str) and value and not os.path.isabs(value) and (
            key.endswith("_filepath") or key.endswith("_dir") or key.endswith("_root")
            or key in {"data_dir", "lookup_dir", "aces_label_dir", "labels_filepath"}
        ):
            data[key] = os.path.normpath(os.path.join(REPO_ROOT, value))

    return cfg


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
