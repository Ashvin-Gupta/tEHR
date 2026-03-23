"""
Baseline: Qwen LLM with fresh LoRA + linear classification head (single-stage).

Loads the base Qwen model directly (no pretraining checkpoint required), applies
fresh LoRA adapters, and trains end-to-end with a logistic regression head in a
single supervised learning stage.

This is the direct-supervised counterpart to the two-stage pipeline
(pretrain → finetune_llm_classifier).
"""
import os
import torch
import numpy as np
from typing import Dict, Any

from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
import wandb

from src.baselines.base_baselines import BaseBaseline
from src.baselines.baseline_utils import setup_output_dir
from src.models.llm_classifier import LLMClassifier
from src.data.classification_collator import ClassificationCollator
from src.training.metrics import compute_classification_metrics
from src.evaluations.evaluations import save_results


class QwenLLMLinearBaseline(BaseBaseline):
    """
    Single-stage baseline: base Qwen LLM + fresh LoRA adapters + linear head.

    The full model (LoRA adapters + linear head) is trained jointly on the
    supervised classification objective from scratch — no continued pretraining
    step is needed.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.training_config = config.get("training", {})
        self.wandb_config = config.get("wandb", {})
        self.lora_config = config.get("lora", {})
        self.output_dir = self.training_config.get("output_dir", self.output_dir)

        self.base_model = None
        self.classifier_model = None
        self.tokenizer = None
        self.trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load base Qwen model and apply fresh LoRA adapters."""
        print("\n" + "=" * 80)
        print("Loading Qwen LLM base model...")
        print("=" * 80)

        model_name = self.model_config["model_name"]
        max_length = self.model_config.get("max_length", 8192)
        load_in_4bit = self.training_config.get("load_in_4bit", True)
        gradient_checkpointing = self.training_config.get("gradient_checkpointing", "unsloth")

        print(f"  - Model:      {model_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - 4-bit:      {load_in_4bit}")

        if torch.cuda.is_available():
            print(f"  - CUDA:       {torch.cuda.get_device_name(0)}")

        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map={"": self.local_rank},
        )
        print(f"  - Loaded base model (vocab size: {len(self.tokenizer)})")

        # Apply fresh LoRA adapters
        r = self.lora_config.get("r", 16)
        lora_alpha = self.lora_config.get("lora_alpha", 32)
        lora_dropout = self.lora_config.get("lora_dropout", 0.05)
        target_modules = self.lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])
        print(f"  - Applying fresh LoRA: r={r}, alpha={lora_alpha}, "
              f"dropout={lora_dropout}, modules={target_modules}")

        self.base_model = FastLanguageModel.get_peft_model(
            self.base_model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=self.lora_config.get("bias", "none"),
            use_gradient_checkpointing=gradient_checkpointing,
            random_state=42,
            use_rslora=self.lora_config.get("use_rslora", True),
            loftq_config=self.lora_config.get("loftq_config", None),
        )
        print("  - Fresh LoRA adapters applied")

        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        pct = 100.0 * trainable_params / total_params if total_params else 0.0
        print(f"  - Trainable: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    # ------------------------------------------------------------------
    # Classifier wrapper
    # ------------------------------------------------------------------

    def create_classifier(self):
        """Wrap the base model (with LoRA) in a classification head."""
        print("\n" + "=" * 80)
        print("Creating linear classification head...")
        print("=" * 80)

        hidden_size = self.model_config["hidden_size"]
        num_labels = self.model_config.get("num_labels", 2)
        multi_label = self.training_config.get("multi_label", False)

        self.classifier_model = LLMClassifier(
            base_model=self.base_model,
            hidden_size=hidden_size,
            num_labels=num_labels,
            freeze_base=True,                   # base weights already frozen by Unsloth/PEFT
            trainable_param_keywords=["lora_"], # keep LoRA adapters trainable
            multi_label=multi_label,
            tokenizer=self.tokenizer,
            head_type=self.model_config.get("head_type", "linear"),
            head_hidden_size=self.model_config.get("head_hidden_size", 512),
            head_dropout=self.model_config.get("head_dropout", 0.1),
        )
        print(f"  - Head: {self.model_config.get('head_type', 'linear')}  "
              f"({hidden_size} → {num_labels})")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------

    def create_trainer(self, train_dataset, val_dataset, run_name: str):
        """Create HuggingFace Trainer for classification."""
        print("\n" + "=" * 80)
        print("Setting up trainer...")
        print("=" * 80)

        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_config.get("max_length"),
            binary_classification=not self.training_config.get("multi_label", False),
            truncation=False,
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "checkpoints"),
            run_name=run_name,
            overwrite_output_dir=bool(self.training_config.get("overwrite_output_dir", True)),
            num_train_epochs=int(self.training_config.get("epochs", 10)),
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
            report_to="wandb" if self.wandb_config.get("enabled", False) else "none",
            remove_unused_columns=False,
        )

        self.trainer = Trainer(
            model=self.classifier_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_classification_metrics,
        )
        print("  - Trainer created")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, dataset) -> np.ndarray:
        """Return predicted probabilities for the positive class."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialised — run `run()` first.")
        pred_output = self.trainer.predict(dataset)
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1].astype(np.float32)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self):
        """Run the single-stage supervised classification pipeline."""
        setup_output_dir(self.output_dir)
        run_name = self.setup_wandb()

        self.load_model()
        self.create_classifier()

        datasets = self.load_datasets(splits=["train", "tuning", "held_out"])
        self.create_trainer(datasets["train"], datasets["tuning"], run_name)

        print("\n" + "=" * 80)
        print("Training...")
        print("=" * 80)
        self.trainer.train()

        if self.training_config.get("save_model", False):
            self.trainer.save_model(os.path.join(self.output_dir, "final_model"))

        val_metrics = self.evaluate_split(datasets["tuning"], "tuning")
        test_metrics = self.evaluate_split(datasets["held_out"], "held_out")

        summary = {"validation": val_metrics, "test": test_metrics}
        save_results(summary, self.output_dir, "summary.json")

        print("\n" + "=" * 80)
        print("Qwen LLM Linear Baseline complete!")
        print("=" * 80)

        return val_metrics, test_metrics
