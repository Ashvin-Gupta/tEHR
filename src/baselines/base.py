"""
Shared scaffolding for comparison methods (zero-shot, XGBoost, ClinicalBERT).

Not an experiment itself. Subclasses implement ``load_model`` and ``predict``.
"""
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from src.data.dataset_factory import get_dataset
from src.evaluations.evaluations import write_split_artifacts
from src.training.training_utils import init_wandb


def setup_output_dir(output_dir: str, overwrite: bool = False) -> str:
    if overwrite and os.path.exists(output_dir):
        print(f"output_dir exists and overwrite=True; reusing path: {output_dir}")
    for subdir in ["plots", "models", "results"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    return output_dir


class BaseBaseline(ABC):
    """
    Shared scaffolding for comparison methods (zero-shot, XGBoost, ClinicalBERT).

    Subclasses implement ``load_model`` and ``predict``.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.data_config = config["data"]
        self.training_config = config.get("training", {})
        self.output_dir = (
            config.get("output_dir")
            or self.training_config.get("output_dir")
            or "./outputs"
        )
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        setup_output_dir(self.output_dir)

    def setup_wandb(self) -> str:
        run_name, _ = init_wandb(
            self.config, local_rank=self.local_rank, default_project="tEHR-baselines"
        )
        return run_name

    def load_datasets(self, splits: List[str] = ["train", "tuning", "held_out"]):
        dataset_name = self.data_config.get("dataset", "CPRD")
        return {split: get_dataset(dataset_name, self.data_config, split) for split in splits}

    @abstractmethod
    def load_model(self): ...

    @abstractmethod
    def predict(self, dataset) -> np.ndarray:
        """Run inference and return predicted probabilities for the positive class."""

    def evaluate_split(self, dataset, split_name: str) -> Dict[str, float]:
        probs = self.predict(dataset)
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            if item is not None:
                label_value = item["label"].item()
                labels.append(0 if label_value == 0 else 1)
        return write_split_artifacts(self.output_dir, split_name, np.array(labels), probs)

    def run(self, resume_from_checkpoint: str | None = None):
        """Train then score tuning / held_out. ``resume_from_checkpoint`` is unused."""
        self.setup_wandb()
        self.load_model()
        datasets = self.load_datasets()
        val_metrics = self.evaluate_split(datasets["tuning"], "tuning")
        test_metrics = self.evaluate_split(datasets["held_out"], "held_out")
        return val_metrics, test_metrics