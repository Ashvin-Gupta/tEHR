import torch
import numpy as np
from abc import abstractmethod
from typing import Dict, Any

from transformers import TrainingArguments, Trainer
from src.baselines.base_baselines import BaseBaseline
from src.data.classification_collator import ClassificationCollator
from src.models.llm_classifier import LLMClassifier
from src.training.metrics import compute_classification_metrics
from src.evaluations.evaluations import compute_baseline_metrics, plot_all_curves, save_results, print_results
import os


class BaseLLMClassifierBaseline(BaseBaseline):
    """
    Shared training loop for all LLM + classification-head baselines.

    This includes:
    - Frozen LLM
    - Pretrained LLM
    - LoRA
    - ClinicalBERT

    Subclasses only need to implement load_model() to set self.base_model
    and self.tokenizer. Everything else (classifier creation, training,
    evaluation) is handled here.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.training_config = config.get("training", {})
        self.base_model = None
        self.tokenizer = None
        self.classifier_model = None
        self.trainer = None

    def create_classifier(self):
        """Wrap base model with classification head. Override for custom head configs."""
        self.classifier_model = LLMClassifier(
            base_model=self.base_model,
            hidden_size=self.model_config["hidden_size"],
            num_labels=self.model_config.get("num_labels", 2),
            freeze_base=self.model_config.get("freeze_llm", True),
            trainable_param_keywords=self._get_trainable_keywords(),
            multi_label=self.training_config.get("multi_label", False),
            tokenizer=self.tokenizer,
            head_type=self.model_config.get("head_type", "linear"),
            head_hidden_size=self.model_config.get("head_hidden_size", 512),
            head_dropout=self.model_config.get("head_dropout", 0.1),
        )

    def _get_trainable_keywords(self):
        """Override in subclasses to allow different parameters to be trained."""
        return []

    def _build_training_args(self, run_name: str) -> TrainingArguments:
        """Build TrainingArguments from config. Shared across all LLM classifier baselines."""
        return TrainingArguments(
            output_dir=os.path.join(self.output_dir, "checkpoints"),
            run_name=run_name,
            overwrite_output_dir=self.training_config.get("overwrite_output_dir", True),
            num_train_epochs=int(self.training_config.get("epochs", 3)),
            per_device_train_batch_size=int(self.training_config.get("batch_size", 2)),
            per_device_eval_batch_size=int(self.training_config.get("eval_batch_size", 1)),
            learning_rate=float(self.training_config.get("learning_rate", 1e-4)),
            weight_decay=float(self.training_config.get("weight_decay", 0.01)),
            warmup_steps=int(self.training_config.get("warmup_steps", 100)),
            gradient_accumulation_steps=int(self.training_config.get("gradient_accumulation_steps", 2)),
            fp16=bool(self.training_config.get("fp16", False)),
            bf16=bool(self.training_config.get("bf16", True)),
            logging_steps=int(self.training_config.get("logging_steps", 50)),
            eval_strategy="steps",
            eval_steps=int(self.training_config.get("eval_steps", 250)),
            save_strategy="steps",
            save_steps=int(self.training_config.get("save_steps", 500)),
            save_total_limit=int(self.training_config.get("save_total_limit", 2)),
            load_best_model_at_end=True,
            metric_for_best_model="auroc",
            greater_is_better=True,
            dataloader_num_workers=int(self.training_config.get("dataloader_num_workers", 0)),
            report_to="wandb" if self.config.get("wandb", {}).get("enabled", False) else "none",
            remove_unused_columns=False,
        )

    def create_trainer(self, train_dataset, val_dataset, run_name: str):
        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_config.get("max_length"),
            binary_classification=not self.training_config.get("multi_label", False),
            truncation=False,
        )
        self.trainer = Trainer(
            model=self.classifier_model,
            args=self._build_training_args(run_name),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_classification_metrics,
        )

    def predict(self, dataset) -> np.ndarray:
        pred_output = self.trainer.predict(dataset)
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    def run(self):
        run_name = self.setup_wandb()
        self.load_model()
        self.create_classifier()
        datasets = self.load_datasets()
        self.create_trainer(datasets["train"], datasets["tuning"], run_name)
        self.trainer.train()
        val_metrics = self.evaluate_split(datasets["tuning"], "tuning")
        test_metrics = self.evaluate_split(datasets["held_out"], "held_out")
        if self.training_config.get("save_model", False):
            self.trainer.save_model(os.path.join(self.output_dir, "final_model"))
        return val_metrics, test_metrics