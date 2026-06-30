import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_baseline_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "data" not in config:
        raise ValueError("Config missing required section: 'data'")
    return config


def setup_output_dir(output_dir: str, overwrite: bool = False) -> str:
    if overwrite and os.path.exists(output_dir):
        print("output_dir exists and overwrite=True; reusing path: %s", output_dir)
    for subdir in ["plots", "models", "results"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return output_dir