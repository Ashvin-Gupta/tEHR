"""
Config loading with ``extends`` support and deep-merge.

Usage in YAML
-------------
::

    extends:
      - ../shared/mimic_data.yaml
      - ../shared/lora.yaml

    model:
      train_lora: true   # overrides anything from parents

Paths in ``extends`` are resolved relative to the child config file.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Union

import yaml


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
    """Resolve an extends path relative to the parent config directory."""
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(os.path.dirname(parent_path), rel))


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config, recursively resolving ``extends`` lists.

    ``extends`` entries are merged first (in order), then the child keys
    override.  The ``extends`` key itself is removed from the result.
    """
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


# Default (ag619 / ix) length-sort cache root. Sites may override via
# ``shared/sites/<site>.yaml`` → ``_site.labels_cache_root``.
DEFAULT_LABELS_CACHE_ROOT = "/home/ag619/experiments/labels_cache"


def _load_site_meta(site: Optional[str], configs_root: str) -> Dict[str, Any]:
    """Load ``shared/sites/<site>.yaml`` metadata (empty for default / ag619)."""
    if not site or site in {"default", "ag619", "ix"}:
        return {
            "name": "default",
            "labels_cache_root": DEFAULT_LABELS_CACHE_ROOT,
            "path_rewrite": [],
            "overlay": {},
        }
    site_path = os.path.join(configs_root, "shared", "sites", f"{site}.yaml")
    if not os.path.exists(site_path):
        sites_dir = os.path.join(configs_root, "shared", "sites")
        available = []
        if os.path.isdir(sites_dir):
            available = sorted(
                f[:-5] for f in os.listdir(sites_dir) if f.endswith(".yaml")
            )
        raise FileNotFoundError(
            f"Unknown site '{site}'. Expected: {site_path}\n"
            f"Available: {available or ['(none)']}"
        )
    raw = load_config(site_path)
    meta = raw.pop("_site", None) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"Site file must contain a `_site` mapping: {site_path}")
    meta.setdefault("name", site)
    meta.setdefault("labels_cache_root", DEFAULT_LABELS_CACHE_ROOT)
    meta.setdefault("path_rewrite", [])
    # Remaining top-level keys (e.g. training.eval_batch_size) override the
    # resolved experiment config for this site.
    meta["overlay"] = raw
    return meta


def _rewrite_paths(obj: Any, rewrites: List[Dict[str, str]]) -> Any:
    """Apply longest-prefix path rewrites to all string leaves in a config tree."""
    if not rewrites:
        return obj
    ordered = sorted(
        rewrites,
        key=lambda r: len(str(r.get("from", ""))),
        reverse=True,
    )

    def _one(s: str) -> str:
        for rule in ordered:
            src = rule.get("from")
            dst = rule.get("to")
            if not src or dst is None:
                continue
            if s == src or s.startswith(src.rstrip("/") + "/"):
                return dst + s[len(src) :]
        return s

    if isinstance(obj, dict):
        return {k: _rewrite_paths(v, rewrites) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_paths(v, rewrites) for v in obj]
    if isinstance(obj, str):
        return _one(obj)
    return obj


def build_mimic_config(
    experiment: str,
    model: Optional[str] = None,
    task: Optional[str] = None,
    configs_root: Optional[str] = None,
    site: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a MIMIC experiment config by merging:

      1. ``mimic/<experiment>.yaml``  (may itself extend shared files)
      2. ``mimic/models/<model>.yaml`` (optional model overlay)
      3. ``mimic/tasks/<task>.yaml``   (optional task overlay; default icu_mortality
         for prediction experiments)
      4. Optional ``shared/sites/<site>.yaml`` path remaps (e.g. ``isambard``)

    For instruct-style experiments (``sft``, ``prompt``), ``model.model_name`` is
    rewritten to ``model.instruct_model_name`` and ``pretrained_checkpoint`` is
    cleared unless the experiment config explicitly sets one after the overlay
    via ``model.keep_pretrained_checkpoint: true``.
    """
    if configs_root is None:
        # src/training/config_utils.py → src/configs
        configs_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
        )

    site_meta = _load_site_meta(site, configs_root)

    # Pretrain has no downstream labels; skip task overlay unless explicitly given.
    TASK_OPTIONAL_EXPERIMENTS = {"pretrain"}
    if task is None and experiment not in TASK_OPTIONAL_EXPERIMENTS:
        task = "icu_mortality"

    exp_path = os.path.join(configs_root, "mimic", f"{experiment}.yaml")
    if not os.path.exists(exp_path):
        available = sorted(
            f[:-5]
            for f in os.listdir(os.path.join(configs_root, "mimic"))
            if f.endswith(".yaml") and not f.startswith(".")
        )
        raise FileNotFoundError(
            f"Unknown experiment '{experiment}'. "
            f"Expected file: {exp_path}\nAvailable: {available}"
        )

    # Snapshot leaf experiment ``model:`` / ``training:`` keys (ignore extends) so
    # they win over the model overlay — e.g. pretrain.yaml ``batch_size: 1``
    # must beat medgemma.yaml ``batch_size: 4``.
    with open(exp_path, "r") as f:
        exp_leaf = yaml.safe_load(f) or {}
    exp_model_leaf = dict(exp_leaf.get("model") or {}) if isinstance(exp_leaf, dict) else {}
    exp_training_leaf = dict(exp_leaf.get("training") or {}) if isinstance(exp_leaf, dict) else {}

    # Shared seed / resume defaults, then experiment overrides.
    cfg: Dict[str, Any] = {}
    repro_path = os.path.join(configs_root, "shared", "reproducibility.yaml")
    if os.path.exists(repro_path):
        cfg = load_config(repro_path)
    cfg = deep_merge(cfg, load_config(exp_path))

    if model:
        model_path = os.path.join(configs_root, "mimic", "models", f"{model}.yaml")
        if not os.path.exists(model_path):
            models_dir = os.path.join(configs_root, "mimic", "models")
            available = sorted(
                f[:-5] for f in os.listdir(models_dir) if f.endswith(".yaml")
            )
            raise FileNotFoundError(
                f"Unknown model '{model}'. "
                f"Expected file: {model_path}\nAvailable: {available}"
            )
        cfg = deep_merge(cfg, load_config(model_path))
        # Re-apply experiment-leaf model keys so they override the model overlay.
        if exp_model_leaf:
            cfg.setdefault("model", {}).update(copy.deepcopy(exp_model_leaf))
        if exp_training_leaf:
            cfg.setdefault("training", {}).update(copy.deepcopy(exp_training_leaf))

    if task:
        task_path = os.path.join(configs_root, "mimic", "tasks", f"{task}.yaml")
        if not os.path.exists(task_path):
            tasks_dir = os.path.join(configs_root, "mimic", "tasks")
            available = sorted(
                f[:-5] for f in os.listdir(tasks_dir) if f.endswith(".yaml")
            )
            raise FileNotFoundError(
                f"Unknown task '{task}'. "
                f"Expected file: {task_path}\nAvailable: {available}"
            )
        cfg = deep_merge(cfg, load_config(task_path))

    # Instruct experiments use the chat/instruct checkpoint, not the base LM.
    INSTRUCT_EXPERIMENTS = {"sft", "prompt"}
    if experiment in INSTRUCT_EXPERIMENTS:
        model_cfg = cfg.setdefault("model", {})
        instruct = model_cfg.get("instruct_model_name")
        if instruct:
            model_cfg["model_name"] = instruct
        if not model_cfg.get("keep_pretrained_checkpoint", False):
            model_cfg["pretrained_checkpoint"] = None

    # Clear domain-pretrain checkpoint for experiments that start from a fresh base.
    # (Model overlays always carry a pretrained_checkpoint path for Exp 1 / 5b.)
    if experiment in {
        "pretrain",
        "ablation_no_pretrain_lora",
        "ablation_frozen_nopretrain",
        # Lexical surface-form ablations: fair comparison without text CPT mismatch.
        "surface_raw",
        "surface_scramble_hash",
        "surface_scramble_shuffle",
    }:
        cfg.setdefault("model", {})["pretrained_checkpoint"] = None

    # Stamp identity for output naming / wandb
    cfg.setdefault("_meta", {})
    cfg["_meta"]["experiment"] = experiment
    cfg["_meta"]["site"] = site_meta.get("name", "default")
    if model:
        cfg["_meta"]["model"] = model
    if task:
        cfg["_meta"]["task"] = task

    # Shared length-sorted index cache: labels_cache/{task}/
    # Approximate length order is shared across models (good enough for packing);
    # exact token lengths differ slightly by tokenizer but indices stay valid.
    # Surface-form ablations change string lengths substantially → separate cache.
    labels_cache_root = site_meta["labels_cache_root"]
    task_name = (cfg.get("task") or {}).get("name") or task
    if task_name:
        training = cfg.setdefault("training", {})
        surface = (cfg.get("data") or {}).get("surface_form", "text")
        if surface and surface != "text":
            cache_subdir = os.path.join(str(task_name), f"surface_{surface}")
        else:
            cache_subdir = str(task_name)
        training.setdefault(
            "data_cache_dir",
            os.path.join(labels_cache_root, cache_subdir),
        )

    # Resolve {experiment} / {model} / {task} placeholders in string fields.
    tag = {
        "experiment": experiment,
        "model": model or "na",
        "task": task or "na",
    }
    cfg = _format_placeholders(cfg, tag)

    # Site-specific filesystem remaps (ag619 paths → Isambard, etc.).
    cfg = _rewrite_paths(cfg, site_meta.get("path_rewrite") or [])

    # Optional site overrides (VRAM / hyperparams), applied after remaps.
    overlay = site_meta.get("overlay") or {}
    if overlay:
        cfg = deep_merge(cfg, overlay)

    return cfg


def _format_placeholders(obj: Any, tag: Dict[str, str]) -> Any:
    """Replace ``{experiment}`` / ``{model}`` / ``{task}`` in string leaves."""
    if isinstance(obj, dict):
        return {k: _format_placeholders(v, tag) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_format_placeholders(v, tag) for v in obj]
    if isinstance(obj, str):
        try:
            return obj.format(**tag)
        except (KeyError, IndexError, ValueError):
            return obj
    return obj


def save_config(config: Dict[str, Any], path: str) -> str:
    """Write a resolved config to disk (used by the SLURM dispatcher)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# CLI: python -m src.training.config_utils build --experiment X [--model Y] [--task Z]
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="MIMIC config builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Build and optionally write a resolved config")
    build.add_argument("--experiment", required=True)
    build.add_argument("--model", default=None)
    build.add_argument("--task", default=None, help="Downstream task (default: icu_mortality)")
    build.add_argument(
        "--site",
        default=None,
        help="Cluster site remaps (e.g. isambard). Default = ag619 paths.",
    )
    build.add_argument("--out", default=None, help="Write resolved YAML here")
    build.add_argument("--print", action="store_true", dest="do_print")

    load = sub.add_parser("load", help="Load a YAML path with extends resolution")
    load.add_argument("path")
    load.add_argument("--print", action="store_true", dest="do_print")

    args = parser.parse_args()

    if args.cmd == "build":
        cfg = build_mimic_config(
            args.experiment,
            model=args.model,
            task=args.task,
            site=args.site,
        )
        if args.out:
            save_config(cfg, args.out)
            print(args.out)
        if args.do_print or not args.out:
            pprint.pprint(cfg)
    elif args.cmd == "load":
        cfg = load_config(args.path)
        if args.do_print:
            pprint.pprint(cfg)
        else:
            pprint.pprint(cfg)


if __name__ == "__main__":
    _cli()
