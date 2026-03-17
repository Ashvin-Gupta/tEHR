import os
import logging
import yaml
import wandb
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from src.baselines.baseline_utils import setup_output_dir, load_baseline_config
from src.data.dataset_factory import get_dataset
from src.evaluations.evaluations import compute_baseline_metrics, plot_all_curves, save_results, print_results

logger = logging.getLogger(__name__)


class BaseBaseline(ABC):
    """
    Shared scaffolding for all baseline models.

    Subclasses must implement:
      - load_model()
      - predict(dataset) -> np.ndarray of probabilities
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.data_config = config["data"]
        self.output_dir = config.get("output_dir", "./outputs")
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        setup_output_dir(self.output_dir)

    def setup_wandb(self) -> str:
        wandb_cfg = self.config.get("wandb", {})
        run_name = wandb_cfg.get("run_name", self.config.get("name", "baseline-run"))
        if wandb_cfg.get("enabled", False) and self.local_rank == 0:
            wandb.init(
                project=wandb_cfg.get("project", "ehr-baselines"),
                name=run_name,
                config=self.config,
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
                labels.append(item["label"].item())
        labels = np.array(labels)

        metrics = compute_baseline_metrics(labels, probs)
        print_results(metrics, split_name)
        plot_all_curves(labels, probs, os.path.join(self.output_dir, "plots", split_name))
        save_results(
            {"metrics": metrics, "labels": labels.tolist(), "probs": probs.tolist()},
            os.path.join(self.output_dir, "results"),
            f"{split_name}_results.json",
        )
        return metrics

    def run(self):
        self.setup_wandb()
        self.load_model()
        datasets = self.load_datasets()
        val_metrics = self.evaluate_split(datasets["tuning"], "tuning")
        test_metrics = self.evaluate_split(datasets["held_out"], "held_out")
        return val_metrics, test_metrics