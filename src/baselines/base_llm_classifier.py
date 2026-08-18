import torch
import numpy as np
import logging
from abc import abstractmethod
from typing import Dict, Any

from transformers import TrainingArguments, Trainer
from src.baselines.base_baselines import BaseBaseline
from src.data.classification_collator import ClassificationCollator
from src.models.llm_classifier import LLMClassifier
from src.training.metrics import compute_classification_metrics
from src.training.eval_schedule import (
    EvalAtStepsCallback,
    append_max_runtime_callback,
    eval_optimizer_steps_from_fractions,
    get_eval_epoch_fractions,
)
from src.evaluations.evaluations import compute_baseline_metrics, plot_all_curves, save_results, print_results
import os

logger = logging.getLogger(__name__)


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
            pooling=self.model_config.get("pooling", "last_token"),
        )

    def _get_trainable_keywords(self):
        """Override in subclasses to allow different parameters to be trained."""
        return []

    def _build_training_args(self, run_name: str, eval_steps: int, save_steps: int) -> TrainingArguments:
        """Build TrainingArguments from config. Shared across all LLM classifier baselines."""
        device_pref = str(self.training_config.get("device", "auto")).lower()
        if device_pref not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unsupported training.device='{device_pref}'. Use one of: auto, cpu, cuda.")

        has_cuda = torch.cuda.is_available()
        require_cuda = bool(self.training_config.get("require_cuda", False))

        use_no_cuda = False
        if device_pref == "cpu":
            use_no_cuda = True
        elif device_pref == "cuda":
            if not has_cuda:
                if require_cuda:
                    raise RuntimeError("training.device='cuda' but CUDA is not available on this node.")
                logger.warning("training.device='cuda' requested but CUDA unavailable; falling back to CPU.")
                use_no_cuda = True
        # device_pref == "auto" keeps HF default (GPU if available, else CPU).

        use_fp16 = bool(self.training_config.get("fp16", False))
        use_bf16 = bool(self.training_config.get("bf16", True))

        bf16_supported = bool(has_cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())

        # Guard precision settings to avoid runtime crashes on unsupported hardware.
        if use_bf16 and not bf16_supported:
            print("bf16 requested but not supported on this setup; disabling bf16.")
            use_bf16 = False
        if use_fp16 and not has_cuda:
            print("fp16 requested but CUDA is unavailable; disabling fp16.")
            use_fp16 = False

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
            fp16=use_fp16,
            bf16=use_bf16,
            no_cuda=use_no_cuda,
            logging_steps=int(self.training_config.get("logging_steps", 50)),
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
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

        per_device_batch = int(self.training_config.get("batch_size", 2))
        grad_accum = int(self.training_config.get("gradient_accumulation_steps", 2))
        raw_fractions = self.training_config.get("eval_epoch_fractions", "default")
        use_fraction_schedule = raw_fractions not in (None, False, "steps", "legacy")
        callbacks = []
        if use_fraction_schedule:
            fractions = get_eval_epoch_fractions(self.training_config)
            eval_at_steps = eval_optimizer_steps_from_fractions(
                len(train_dataset),
                fractions,
                per_device_batch=per_device_batch,
                grad_accum=grad_accum,
            )
            eval_steps = 10**9
            save_steps = 10**9
            callbacks.append(EvalAtStepsCallback(eval_at_steps, fractions=fractions))
            print(f"  - Eval epoch fractions: {fractions}")
            print(f"  - Eval optimizer steps: {eval_at_steps}")
        else:
            eval_steps = int(self.training_config.get("eval_steps", 250))
            save_steps = int(self.training_config.get("save_steps", 500))

        append_max_runtime_callback(callbacks, self.training_config)

        self.trainer = Trainer(
            model=self.classifier_model,
            args=self._build_training_args(run_name, eval_steps=eval_steps, save_steps=save_steps),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_classification_metrics,
            callbacks=callbacks,
        )

    def predict(self, dataset) -> np.ndarray:
        pred_output = self.trainer.predict(dataset)
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    def _restore_weights_from_checkpoint(self, checkpoint_path: str):
        """
        Load LoRA + classifier-head weights from a Trainer checkpoint into
        ``self.classifier_model``, without touching optimizer / step counter.

        Head-only runs (``train_lora: false``) legitimately have no LoRA keys;
        those just restore ``classifier.*``.
        """
        print("\n" + "=" * 80)
        print("Loading weights from checkpoint (eval-only)...")
        print("=" * 80)
        print(f"  - Source checkpoint: {checkpoint_path}")

        weights_file = os.path.join(checkpoint_path, "model.safetensors")
        if not os.path.exists(weights_file):
            weights_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"No model weights found in checkpoint directory: {checkpoint_path}"
            )

        try:
            from safetensors.torch import load_file as _st_load
            ckpt_state = _st_load(weights_file)
        except Exception:
            ckpt_state = torch.load(weights_file, map_location="cpu", weights_only=True)

        lora_keywords = (
            "lora_A", "lora_B", "lora_magnitude_vector",
            "lora_embedding_A", "lora_embedding_B",
        )
        transfer_state = {
            k: v for k, v in ckpt_state.items()
            if any(kw in k for kw in lora_keywords) or k.startswith("classifier.")
        }
        lora_keys = [k for k in transfer_state if any(kw in k for kw in lora_keywords)]
        head_keys = [k for k in transfer_state if k.startswith("classifier.")]

        if not transfer_state:
            raise RuntimeError(
                f"No LoRA or classifier.* keys found in {weights_file}."
            )

        missing, unexpected = self.classifier_model.load_state_dict(
            transfer_state, strict=False
        )
        print(f"  - Loaded {len(lora_keys)} LoRA tensors, {len(head_keys)} head tensors.")
        if unexpected:
            print(
                f"  - WARNING: {len(unexpected)} keys did not match the current model "
                f"and were dropped: {unexpected[:5]}"
            )
        unexpected_missing = [
            k for k in missing
            if any(kw in k for kw in lora_keywords) or k.startswith("classifier.")
        ]
        if unexpected_missing:
            print(
                f"  - WARNING: {len(unexpected_missing)} LoRA/classifier keys missing "
                f"from checkpoint: {unexpected_missing[:5]}"
            )

    def run_eval_only(self, checkpoint_path: str):
        """
        Score an existing checkpoint on held_out only, with no training.

        Same role as ``EHRClassificationTrainer.run_eval_only`` for the
        plain-python baseline path (``ablation_frozen_nopretrain`` / clinicalbert).
        """
        print("\n" + "=" * 80)
        print("Evaluation-only run (held_out, no training)")
        print(f"  - Checkpoint: {checkpoint_path}")
        print("=" * 80)

        self.training_config["overwrite_output_dir"] = False

        # Prefer a single visible GPU, same as run().
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if "," in visible:
            first = visible.split(",")[0].strip() or "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = first

        self.load_model()
        self.create_classifier()
        self._restore_weights_from_checkpoint(checkpoint_path)

        datasets = self.load_datasets(splits=["held_out"])
        # create_trainer needs a train slot for the fraction schedule math; the
        # test split stands in — no training loop runs.
        self.create_trainer(datasets["held_out"], datasets["held_out"], "eval-only")

        test_metrics = self.evaluate_split(datasets["held_out"], "held_out")

        if self.local_rank == 0:
            import json
            import time

            step = None
            base = os.path.basename(os.path.normpath(checkpoint_path))
            if base.startswith("checkpoint-"):
                try:
                    step = int(base.split("-")[-1])
                except ValueError:
                    pass
            payload = {
                "evaluated_checkpoint": checkpoint_path,
                "checkpoint_step": step,
                "produced_by": "src.pipelines.run_baseline --eval_only",
                "note": (
                    "Metrics come from the checkpoint above (test/held_out only), "
                    "not from a training run that reached its final step."
                ),
                "n_test": len(datasets["held_out"]),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            results_dir = os.path.join(self.output_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, "eval_provenance.json")
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"  - Provenance written to {path}")

        return None, test_metrics

    def run(self):
        # Prefer a single visible GPU for this plain-python path. Must run
        # before model load; run.sh also pins CUDA_VISIBLE_DEVICES for safety.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if "," in visible:
            first = visible.split(",")[0].strip() or "0"
            os.environ["CUDA_VISIBLE_DEVICES"] = first
            logger.warning(
                "Multiple GPUs in CUDA_VISIBLE_DEVICES (%s); pinning to %s "
                "for single-process 4-bit LLM baseline training.",
                visible,
                first,
            )

        run_name = self.setup_wandb()
        self.load_model()
        self.create_classifier()
        datasets = self.load_datasets()
        self.create_trainer(datasets["train"], datasets["tuning"], run_name)

        resume_ckpt = self.training_config.get("resume_from_checkpoint")
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")
            self.trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            self.trainer.train()

        # ── FLOPs reporting ─────────────────────────────────────────────────
        try:
            from src.training.flops_counter import report_flops_from_trainer
            report_flops_from_trainer(
                trainer=self.trainer,
                model=self.classifier_model,
                seq_length=int(self.model_config.get("max_length", 8192)),
                output_dir=self.output_dir,
                local_rank=getattr(self, "local_rank", 0),
            )
        except Exception as _flops_err:
            logger.warning(f"FLOPs reporting failed (non-fatal): {_flops_err}")

        val_metrics = self.evaluate_split(datasets["tuning"], "tuning")
        test_metrics = self.evaluate_split(datasets["held_out"], "held_out")
        if self.training_config.get("save_model", False):
            self.trainer.save_model(os.path.join(self.output_dir, "final_model"))
        return val_metrics, test_metrics