"""
Baseline: Qwen3-Embedding-8B with two-stage training.

Stage 1 (optional):
    - LoRA adapters + Qwen3-Embedding
    - Masked language modeling (packed EHR trajectories, no per-patient truncation)
    - Full stream of patients, separated by EOS/separator tokens; multiple patients
      can appear in a single sequence and a single patient can span multiple sequences.

Stage 2:
    - Frozen (or LoRA-only) Qwen3-Embedding encoder
    - CLS-pooled embedding → linear head (logistic regression-style) for binary
      classification.
"""
import os
import torch
import numpy as np
from typing import Dict, Any
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from datasets import Dataset

from src.baselines.utils import load_baseline_config, setup_output_dir, load_datasets
from src.models.qwen3_embedding_classifier import Qwen3EmbeddingClassifier
from src.models.qwen3_embedding_mlm import Qwen3EmbeddingMLMHead
from src.data.classification_collator import ClassificationCollator
from src.data.qwen3_mlm_collator import Qwen3MLMCollator
from src.training.metrics import compute_classification_metrics
from src.evaluations.baseline_metrics import compute_baseline_metrics, plot_all_curves, save_results, print_results
from src.data.preprocessing import extract_text
from src.training.trainer import pack_and_chunk_texts


class Qwen3EmbeddingBaseline:
    """
    Qwen3-Embedding baseline for EHR classification.
    
    Uses Qwen3-Embedding-8B as encoder with last-token pooling and a linear head.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model', {})
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        self.wandb_config = config.get('wandb', {})
        self.pretrain_config = config.get('pretrain', {})  # optional stage-1 config
        self.output_dir = self.training_config.get('output_dir', './outputs/qwen3_embedding')

        # Shared components
        self.base_model = None
        self.model = None
        self.mlm_model = None
        self.tokenizer = None
        self.trainer = None
        self.mlm_trainer = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    def setup_wandb(self) -> str:
        """Setup WandB logging."""
        if self.wandb_config.get('enabled', False):
            run_name = self.wandb_config.get("run_name") or f"qwen3_embedding_{self.config.get('name', 'default')}"
            if self.local_rank == 0:
                wandb.init(
                    project=self.wandb_config.get("project", "llm-classification"),
                    name=run_name,
                    config=self.config
                )
                print(f"\nWandB enabled - Project: {self.wandb_config.get('project')}, Run: {run_name}")
            return run_name
        return "qwen3-embedding-run"

    def load_base_model(self):
        """
        Load Qwen3-Embedding base model and tokenizer, optionally with LoRA.

        This prepares `self.base_model` and `self.tokenizer` that can then be
        wrapped by either:
          - `Qwen3EmbeddingMLMHead` for stage-1 MLM
          - `Qwen3EmbeddingClassifier` for stage-2 classification
        """
        print("\n" + "=" * 80)
        print("Loading Qwen3-Embedding base model...")
        print("=" * 80)

        model_name = self.model_config.get('model_name', 'Qwen/Qwen3-Embedding-8B')
        hidden_size = self.model_config.get('hidden_size', 4096)
        num_labels = self.model_config.get('num_labels', 2)
        head_dropout = self.model_config.get('head_dropout', 0.1)
        max_length = self.data_config.get('max_length', 8192)

        print(f"  - Model: {model_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Num labels: {num_labels}")

        # Load tokenizer - use left padding as recommended for Qwen3-Embedding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Stage-1 (optional) uses MLM masking; Qwen tokenizers typically don't ship a mask token.
        # If enabled, we add one so masking uses [MASK] instead of random-token-only replacement.
        if self.pretrain_config.get("enabled", False) and self.pretrain_config.get("add_mask_token", True):
            if getattr(self.tokenizer, "mask_token", None) is None:
                self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
                print("  - Added [MASK] token to tokenizer for stage-1 MLM")

        # Load base embedding model
        base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2" # Optional: drastically reduces memory for long sequences
        )

        # If we added tokens (e.g. [MASK]), resize the embedding matrix.
        if hasattr(base_model, "resize_token_embeddings"):
            base_model.resize_token_embeddings(len(self.tokenizer))

        # Two options only:
        # 1) use_lora: false → frozen embedding + trainable linear head
        # 2) use_lora: true  → frozen embedding + trainable LoRA adapters + trainable linear head
        use_lora = self.model_config.get("use_lora", False)
        if use_lora:
            lora_config = self.model_config.get("lora", {})
            target_modules = lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ])
            r = lora_config.get("r", 16)
            lora_alpha = lora_config.get("lora_alpha", 16)
            lora_dropout = lora_config.get("lora_dropout", 0.05)
            bias = lora_config.get("bias", "none")
            print(f"  - Mode: frozen embedding + trainable LoRA + trainable linear head")
            print(f"  - LoRA: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, target_modules={target_modules}")
            peft_config = LoraConfig(
                r=r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
            )
            base_model = get_peft_model(base_model, peft_config)
            print("  - LoRA adapters applied")
            trainable_param_keywords = ["lora_"]
        else:
            print(f"  - Mode: frozen embedding + trainable linear head only")
            trainable_param_keywords = []

        # Freeze base weights always; keep LoRA params trainable if enabled.
        for p in base_model.parameters():
            p.requires_grad = False
        if use_lora:
            for name, p in base_model.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

        # Enable gradient checkpointing to reduce memory (saves activation memory when training LoRA)
        # if self.training_config.get("gradient_checkpointing", True) and hasattr(base_model, "gradient_checkpointing_enable"):
        #     base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        #     print("  - Enabled gradient checkpointing for base model")

        # Store base model and metadata; heads are created per-stage
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.head_dropout = head_dropout
        self.trainable_param_keywords = trainable_param_keywords

        # Enable gradient checkpointing to reduce memory (saves activation memory)
        # if self.training_config.get("gradient_checkpointing", True):
        #     if hasattr(self.base_model, "gradient_checkpointing_enable"):
        #         self.base_model.gradient_checkpointing_enable(
        #             gradient_checkpointing_kwargs={"use_reentrant": False}
        #         )

        print(f"  - Base model loaded successfully")
        print(f"  - Vocab size: {len(self.tokenizer)}")

        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        pct = 100.0 * trainable_params / total_params if total_params else 0
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    def build_classifier_model(self):
        """
        Wrap the shared `base_model` with a classification head.
        """
        if self.base_model is None:
            self.load_base_model()

        self.model = Qwen3EmbeddingClassifier(
            base_model=self.base_model,
            hidden_size=self.hidden_size,
            num_labels=self.num_labels,
            head_dropout=self.head_dropout,
            freeze_base=True,
            trainable_param_keywords=self.trainable_param_keywords,
        )

    def build_mlm_model(self):
        """
        Wrap the shared `base_model` with an MLM head for stage-1 pretraining.
        """
        if self.base_model is None:
            self.load_base_model()

        vocab_size = len(self.tokenizer)
        self.mlm_model = Qwen3EmbeddingMLMHead(
            base_model=self.base_model,
            hidden_size=self.hidden_size,
            vocab_size=vocab_size,
            freeze_base=True,
            trainable_param_keywords=self.trainable_param_keywords,
        )

    def prepare_mlm_datasets(self, datasets: Dict[str, Any]) -> Dict[str, Dataset]:
        """
        Prepare packed MLM datasets from patient-level text.

        This mirrors the `EHRPretrainer` behaviour:
          - Extract full patient trajectories as text.
          - Tokenize & concatenate all patients into one long stream with EOS
            separators between patients.
          - Slice into fixed-length chunks (no per-patient truncation).
        """
        print("\n" + "=" * 80)
        print("Preparing packed MLM datasets (no per-patient padding/truncation)...")
        print("=" * 80)

        # IMPORTANT: keep stage-1 context length separate from classification max_length to avoid OOM
        max_length = int(self.pretrain_config.get("max_length", 2048))

        mlm_datasets = {}
        for split_name in ["train", "tuning"]:
            base_dataset = datasets[split_name]
            print(f"\n  - Extracting text for split: {split_name}")
            text_list = extract_text(base_dataset, self.tokenizer)
            print(f"    • Patients in {split_name}: {len(text_list)}")

            print(f"    • Packing to context length: {max_length}")
            chunk_ids = pack_and_chunk_texts(text_list, self.tokenizer, max_length)

            print(f"    • Created {len(chunk_ids)} packed sequences for {split_name}")
            if len(chunk_ids) > 0:
                lengths = [len(seq) for seq in chunk_ids]
                print(
                    f"    • Sequence lengths (min/mean/max): "
                    f"{min(lengths)}/{sum(lengths)/len(lengths):.1f}/{max(lengths)}"
                )
                # Debug: show first two sequences' first/last few tokens
                for i in range(min(2, len(chunk_ids))):
                    seq = chunk_ids[i]
                    print(f"    • Example seq {i}: len={len(seq)}, "
                          f"head={seq[:10]}, tail={seq[-10:]}")

            mlm_datasets[split_name] = Dataset.from_dict({"input_ids": chunk_ids})

        print("\nPacked MLM datasets ready.\n" + "=" * 80)
        return mlm_datasets

    def create_trainer(self, train_dataset, val_dataset, collator, run_name: str):
        """Create HuggingFace Trainer."""
        print("\n" + "=" * 80)
        print("Setting up training...")
        print("=" * 80)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, 'checkpoints'),
            run_name=run_name,
            overwrite_output_dir=self.training_config.get('overwrite_output_dir', True),
            num_train_epochs=int(self.training_config.get('epochs', 3)),
            per_device_train_batch_size=int(self.training_config.get('batch_size', 4)),
            per_device_eval_batch_size=int(self.training_config.get('eval_batch_size', 8)),
            learning_rate=float(self.training_config.get('learning_rate', 2e-5)),
            weight_decay=float(self.training_config.get('weight_decay', 0.01)),
            warmup_steps=int(self.training_config.get('warmup_steps', 100)),
            gradient_accumulation_steps=int(self.training_config.get('gradient_accumulation_steps', 2)),
            fp16=bool(self.training_config.get('fp16', False)),
            bf16=bool(self.training_config.get('bf16', True)),
            gradient_checkpointing=bool(self.training_config.get('gradient_checkpointing', True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=int(self.training_config.get('logging_steps', 50)),
            eval_steps=int(self.training_config.get('eval_steps', 250)),
            save_steps=int(self.training_config.get('save_steps', 500)),
            save_total_limit=int(self.training_config.get('save_total_limit', 2)),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            dataloader_num_workers=int(self.training_config.get('dataloader_num_workers', 4)),
            report_to="wandb" if self.wandb_config.get('enabled', False) else "none",
            remove_unused_columns=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=compute_classification_metrics,
        )

        print("  - Trainer created successfully")

        # Debug: print shapes of the first two training batches
        print("\n" + "=" * 80)
        print("DEBUG: First two classification batches (shapes)")
        print("=" * 80)
        try:
            train_dl = self.trainer.get_train_dataloader()
            for i, batch in enumerate(train_dl):
                input_shape = tuple(batch["input_ids"].shape)
                mask_shape = tuple(batch["attention_mask"].shape)
                labels_shape = tuple(batch["labels"].shape)
                print(
                    f"  - Batch {i}: "
                    f"input_ids {input_shape}, "
                    f"attention_mask {mask_shape}, "
                    f"labels {labels_shape}"
                )
                if i >= 1:
                    break
        except Exception as e:
            print(f"  ⚠️  Error inspecting classification dataloader: {e}")
        print("=" * 80)

    def create_mlm_trainer(self, train_dataset, val_dataset, run_name: str):
        """
        Create a Trainer for masked language modeling (stage-1).
        """
        print("\n" + "=" * 80)
        print("Setting up MLM pretraining...")
        print("=" * 80)

        pretrain_cfg = self.pretrain_config or {}

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, "mlm_checkpoints"),
            run_name=f"{run_name}-mlm",
            overwrite_output_dir=pretrain_cfg.get("overwrite_output_dir", True),
            num_train_epochs=int(pretrain_cfg.get("epochs", self.training_config.get("epochs", 1))),
            per_device_train_batch_size=int(pretrain_cfg.get("batch_size", self.training_config.get("batch_size", 1))),
            per_device_eval_batch_size=int(pretrain_cfg.get("eval_batch_size", self.training_config.get("eval_batch_size", 1))),
            learning_rate=float(pretrain_cfg.get("learning_rate", 2e-5)),
            weight_decay=float(pretrain_cfg.get("weight_decay", 0.01)),
            warmup_steps=int(pretrain_cfg.get("warmup_steps", 100)),
            gradient_accumulation_steps=int(pretrain_cfg.get("gradient_accumulation_steps", 1)),
            fp16=bool(pretrain_cfg.get("fp16", self.training_config.get("fp16", False))),
            bf16=bool(pretrain_cfg.get("bf16", self.training_config.get("bf16", True))),
            gradient_checkpointing=bool(
                pretrain_cfg.get(
                    "gradient_checkpointing",
                    self.training_config.get("gradient_checkpointing", True),
                )
            ),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=int(pretrain_cfg.get("logging_steps", self.training_config.get("logging_steps", 50))),
            eval_steps=int(pretrain_cfg.get("eval_steps", self.training_config.get("eval_steps", 250))),
            save_steps=int(pretrain_cfg.get("save_steps", self.training_config.get("save_steps", 500))),
            save_total_limit=int(pretrain_cfg.get("save_total_limit", self.training_config.get("save_total_limit", 2))),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            dataloader_num_workers=int(pretrain_cfg.get("dataloader_num_workers", 4)),
            report_to="wandb" if self.wandb_config.get("enabled", False) else "none",
            remove_unused_columns=False,
        )

        mlm_collator = Qwen3MLMCollator(
            tokenizer=self.tokenizer,
            mlm_probability=float(pretrain_cfg.get("mlm_probability", 0.15)),
        )

        self.mlm_trainer = Trainer(
            model=self.mlm_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=mlm_collator,
        )

        print("  - MLM Trainer created successfully")

        # Debug: print shapes of the first two MLM batches
        print("\n" + "=" * 80)
        print("DEBUG: First two MLM batches (shapes)")
        print("=" * 80)
        try:
            train_dl = self.mlm_trainer.get_train_dataloader()
            for i, batch in enumerate(train_dl):
                input_shape = tuple(batch["input_ids"].shape)
                mask_shape = tuple(batch["attention_mask"].shape)
                labels_shape = tuple(batch["labels"].shape)
                print(
                    f"  - MLM Batch {i}: "
                    f"input_ids {input_shape}, "
                    f"attention_mask {mask_shape}, "
                    f"labels {labels_shape}"
                )
                if i >= 1:
                    break
        except Exception as e:
            print(f"  ⚠️  Error inspecting MLM dataloader: {e}")
        print("=" * 80)

    def train_mlm(self):
        """Run the stage-1 MLM pretraining loop."""
        print("\n" + "=" * 80)
        print("Stage 1: MLM pretraining (Qwen3-Embedding + LoRA)")
        print("=" * 80)
        self.mlm_trainer.train()
        print("\nStage 1 (MLM) completed!")

    def train(self):
        """Train the model."""
        print("\n" + "=" * 80)
        print("Training Qwen3-Embedding classifier...")
        print("=" * 80)
        self.trainer.train()
        print("\nTraining completed!")

    def evaluate(self, dataset, split_name: str = "validation") -> Dict[str, float]:
        """Evaluate model on a dataset."""
        print(f"\n" + "=" * 80)
        print(f"Evaluating on {split_name} set...")
        print("=" * 80)

        pred_output = self.trainer.predict(dataset)
        logits = pred_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        labels = pred_output.label_ids

        metrics = compute_baseline_metrics(labels, probs)
        print_results(metrics, split_name)

        plot_dir = os.path.join(self.output_dir, 'plots', split_name)
        plot_all_curves(labels, probs, plot_dir)

        results = {
            'metrics': metrics,
            'labels': labels.tolist(),
            'probs': probs.tolist()
        }
        save_results(results, os.path.join(self.output_dir, 'results'), f'{split_name}_results.json')

        return metrics

    def save_model(self):
        """Save the trained model (custom nn.Module - save state_dict + tokenizer)."""
        final_model_path = os.path.join(self.output_dir, 'final_model')
        os.makedirs(final_model_path, exist_ok=True)
        # Save full model state (base + classifier)
        torch.save(self.model.state_dict(), os.path.join(final_model_path, 'pytorch_model.bin'))
        self.tokenizer.save_pretrained(final_model_path)
        # Save config for loading
        import json
        with open(os.path.join(final_model_path, 'config.json'), 'w') as f:
            json.dump({
                'model_name': self.model_config.get('model_name'),
                'hidden_size': self.model_config.get('hidden_size', 4096),
                'num_labels': self.model_config.get('num_labels', 2),
                'head_dropout': self.model_config.get('head_dropout', 0.1),
            }, f, indent=2)
        print(f"\n  - Model saved to: {final_model_path}")

    def run(self):
        """Run the complete baseline pipeline (optional two-stage training)."""
        setup_output_dir(self.output_dir, overwrite=self.training_config.get('overwrite_output_dir', False))

        run_name = self.setup_wandb()
        self.load_base_model()

        # Load datasets once (text format for classification & for MLM packing)
        datasets = load_datasets(
            self.data_config,
            splits=['train', 'tuning', 'held_out'],
            format='text',
            tokenizer=self.tokenizer
        )

        # Optional Stage 1: MLM pretraining with LoRA adapters only
        if self.pretrain_config.get("enabled", False):
            print("\n" + "=" * 80)
            print("Two-stage training ENABLED: running MLM pretraining first.")
            print("=" * 80)
            # Prepare packed MLM datasets (no per-patient truncation)
            mlm_datasets = self.prepare_mlm_datasets(datasets)
            self.build_mlm_model()
            self.create_mlm_trainer(mlm_datasets['train'], mlm_datasets['tuning'], run_name)
            self.train_mlm()
        else:
            print("\nTwo-stage training disabled (no MLM pretraining). Proceeding directly to classification.")

        # Stage 2: supervised classification with logistic regression head
        self.build_classifier_model()

        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.data_config.get('max_length', 8192),
            truncation=True,
        )

        self.create_trainer(datasets['train'], datasets['tuning'], collator, run_name)
        self.train()
        self.save_model()

        val_metrics = self.evaluate(datasets['tuning'], 'validation')
        test_metrics = self.evaluate(datasets['held_out'], 'test')

        summary = {
            'validation': val_metrics,
            'test': test_metrics
        }
        save_results(summary, self.output_dir, 'summary.json')

        print("\n" + "=" * 80)
        print("Qwen3-Embedding Baseline Complete!")
        print("=" * 80)

        return val_metrics, test_metrics
