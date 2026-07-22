#!/usr/bin/env python3
"""
Preview MIMIC surface forms (text / raw / scramble_*) on a few examples.

No GPU required. Useful to sanity-check lexical ablations before launching jobs.

Example:
  python -m src.analysis.preview_surface_forms \\
    --config_filepath src/configs/mimic/_resolved_isambard/classifier-mistral-icu_mortality.yaml \\
    --n 3 --forms text raw scramble_hash scramble_shuffle
"""
from __future__ import annotations

import argparse
import os
from typing import List

from src.training.config_utils import load_config
from src.data.dataset_factory import get_dataset


def _preview(data_cfg: dict, split: str, forms: List[str], n: int) -> None:
    for form in forms:
        cfg = dict(data_cfg)
        cfg["surface_form"] = form
        ds = get_dataset(cfg.get("dataset", "MIMIC"), cfg, split)
        print("\n" + "=" * 80)
        print(f"surface_form={form}  split={split}  n={min(n, len(ds))}")
        print("=" * 80)
        for i in range(min(n, len(ds))):
            item = ds[i]
            text = item.get("text", "")
            sid = item.get("subject_id")
            label = item.get("label")
            label_v = int(label.item()) if hasattr(label, "item") else int(label)
            snippet = text if len(text) <= 800 else text[:800] + " …"
            print(f"\n--- example {i} subject={sid} label={label_v} chars={len(text)} ---")
            print(snippet)


def main():
    p = argparse.ArgumentParser(description="Preview MIMIC surface-form narratives")
    p.add_argument("--config_filepath", required=True, help="Any resolved MIMIC YAML")
    p.add_argument("--split", default="held_out")
    p.add_argument("--n", type=int, default=2)
    p.add_argument(
        "--forms",
        nargs="+",
        default=["text", "raw", "scramble_hash", "scramble_shuffle"],
    )
    args = p.parse_args()

    cfg = load_config(os.path.abspath(args.config_filepath))
    data_cfg = cfg["data"]
    _preview(data_cfg, args.split, args.forms, args.n)


if __name__ == "__main__":
    main()
