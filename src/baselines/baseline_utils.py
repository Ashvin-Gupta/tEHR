import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_baseline_config(config_path: str) -> Dict[str, Any]:
    from src.training.config_utils import load_config
    config = load_config(config_path)
    if "data" not in config:
        raise ValueError("Config missing required section: 'data'")
    return config


def setup_output_dir(output_dir: str, overwrite: bool = False) -> str:
    if overwrite and os.path.exists(output_dir):
        print("output_dir exists and overwrite=True; reusing path: %s", output_dir)
    for subdir in ["plots", "models", "results"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return output_dir