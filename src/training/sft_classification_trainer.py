"""
SFT Classification Trainer.

Fine-tunes a causal LLM (via LoRA + SFTTrainer) to generate "0" or "1" given a
patient EHR prompt.  Evaluation uses logit-based probability extraction rather than
parsing generated text: the model is run in forward-pass mode and the next-token
logits for the literal tokens "0" and "1" are extracted.  If both logits are below
`model.logit_low_threshold` the prediction defaults to 0.5.
"""
import os
import json
import random
import time
import torch
import numpy as np
import wandb
from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from src.data.dataset_factory import get_dataset
from src.training.eval_schedule import append_max_runtime_callback
from src.evaluations.evaluations import (
    compute_baseline_metrics,
    plot_all_curves,
    save_results,
    print_results,
)
from src.training.training_utils import resolve_binary_label_token_ids

import logging
logger = logging.getLogger(__name__)


class _LogitClassificationCallback(TrainerCallback):
    """
    SFTTrainer's built-in eval only reports next-token CE loss.

    After each Trainer evaluate(), run logit-based P(y=1) extraction on the
    classification val set and log AUROC / AUPR / ECE / accuracy / F1 like the
    logistic classifier.
    """

    def __init__(
        self,
        outer: "SFTClassificationTrainer",
        eval_dataset: "_SFTDataset",
        batch_size: int,
    ):
        self.outer = outer
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.outer.local_rank != 0:
            return
        clf_metrics = self.outer.compute_logit_metrics(
            self.eval_dataset,
            batch_size=self.batch_size,
        )
        print_results(clf_metrics, "eval (logit)")
        log_dict = {f"eval_{k}": float(v) for k, v in clf_metrics.items()}
        if metrics is not None:
            metrics.update(log_dict)
        if self.outer.wandb_config.get("enabled", False):
            wandb.log({**log_dict, "train/global_step": state.global_step}, step=state.global_step)


# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------

def _load_prompt_template(prompt_cfg: Dict[str, Any]) -> str:
    """Load the prompt template from the path given in config."""
    path = prompt_cfg.get("filepath", "src/resources/sft_classifier_prompt.txt")
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# HF Dataset wrapper
# ---------------------------------------------------------------------------

class _SFTDataset:
    """
    Wraps the MIMIC dataset to produce SFT training examples.

    Each example is a prompt/completion pair: the prompt ends at "Answer:"
    and the completion is the label token (" 0" or " 1").  TRL's
    completion_only_loss masks the prompt during training.

    EHR truncation
    --------------
    Long EHR texts are truncated BEFORE the prompt is assembled so that
    the final sequence always fits within max_length.  Truncation removes
    the OLDEST clinical events (those at the start of the clinical body,
    just after the `<HEADER_SPLIT>` marker) while always preserving:
      1. The full demographics header (age, sex, etc.)
      2. The most recent events (end of the EHR text)
      3. The prompt prefix and the `Answer:` suffix

    This ensures the tokenizer never needs to truncate from the right, which
    would otherwise silently remove the answer token and produce zero training
    loss for that sample.

    Prediction-time cutoff is applied upstream by MIMICDataset
    (_apply_prediction_time_cutoff) before the text reaches this class.
    """

    # Reserved tokens for prompt prefix + response template + answer token + EOS.
    # This is a conservative upper-bound; the actual overhead is measured below.
    _ANSWER_OVERHEAD = 4

    def __init__(
        self,
        base_dataset,
        prompt_template: str,
        task_description: str,
        tokenizer=None,
        max_length: int = 8192,
    ):
        self.prompt_template = prompt_template
        self.task_description = task_description
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-measure the prompt overhead (everything except the EHR body).
        # We use an empty EHR string so we can subtract it later.
        self._prompt_overhead_tokens = self._measure_prompt_overhead()

        # Materialise: each item needs both 'text' and 'label'.
        # This is the slow step after shard loading — each sample translates
        # tokens→text and may re-tokenise the EHR for length truncation.
        n = len(base_dataset)
        print(
            f"  Materialising {n:,} SFT samples (token→text + EHR truncation); "
            "this can take several minutes...",
            flush=True,
        )
        self.samples: List[Dict[str, Any]] = []
        n_truncated = 0
        for i in tqdm(range(n), desc="Materialising SFT samples", mininterval=5.0):
            item = base_dataset[i]
            if item is None:
                continue
            text = item.get("text")
            label = item.get("label")
            if text is None or label is None:
                continue
            binary_label = 0 if int(label) == 0 else 1
            truncated = self._truncate_ehr(text)
            if truncated != text:
                n_truncated += 1
            self.samples.append({"text": truncated, "label": binary_label})
        print(
            f"  Materialised {len(self.samples):,} samples "
            f"({n_truncated:,} needed EHR truncation)",
            flush=True,
        )

    def _measure_prompt_overhead(self) -> int:
        """
        Count the number of tokens used by the prompt structure itself
        (prefix + empty EHR + response template + answer + EOS).
        """
        if self.tokenizer is None:
            # Rough fallback: assume 50 tokens for prompt overhead
            return 50
        dummy_prompt = self.prompt_template.format(task=self.task_description, ehr_text="")
        dummy_full = dummy_prompt + " 0"
        ids = self.tokenizer.encode(dummy_full, add_special_tokens=True)
        return len(ids) + self._ANSWER_OVERHEAD

    def _truncate_ehr(self, ehr_text: str) -> str:
        """
        Truncate EHR text so it fits in the token budget.

        Strategy:
          - Split on '<HEADER_SPLIT>' to separate demographics from clinical body.
          - Always keep the full demographics header (it is short).
          - Drop the OLDEST part of the clinical body (beginning of the body string).
          - Keep the most recent events (end of the body string).

        If no tokenizer is available we skip truncation and rely on the
        tokenizer's own truncation (which may be incorrect — always provide
        a tokenizer in production).
        """
        if self.tokenizer is None:
            return ehr_text

        budget = self.max_length - self._prompt_overhead_tokens
        if budget <= 0:
            return ehr_text  # degenerate config — don't silently corrupt

        # Split into demographics header and clinical body
        header_split_marker = " <HEADER_SPLIT> "
        if header_split_marker in ehr_text:
            header, body = ehr_text.split(header_split_marker, 1)
        else:
            header = ""
            body = ehr_text

        # Measure the header's token cost
        header_tokens = len(self.tokenizer.encode(header, add_special_tokens=False)) if header else 0
        body_budget = budget - header_tokens

        if body_budget <= 0:
            # Extremely long header — return just header (edge case)
            return header

        # Measure body tokens
        body_ids = self.tokenizer.encode(body, add_special_tokens=False)
        if len(body_ids) <= body_budget:
            # Body already fits; no truncation needed
            return ehr_text

        # Keep only the last `body_budget` tokens of the body (most recent events)
        truncated_body_ids = body_ids[-body_budget:]
        truncated_body = self.tokenizer.decode(truncated_body_ids, skip_special_tokens=True)

        if header:
            return header + header_split_marker + truncated_body
        return truncated_body

    def _format_prompt(self, ehr_text: str) -> str:
        return self.prompt_template.format(
            task=self.task_description,
            ehr_text=ehr_text,
        )

    def as_hf_dataset(self) -> HFDataset:
        """
        Return an HF Dataset with separate prompt/completion columns.
        Prefer ``as_tokenized_hf_dataset`` when using Unsloth's SFTTrainer,
        which does not accept the prompt/completion column format.
        """
        rows = []
        for s in self.samples:
            prompt = self._format_prompt(s["text"])
            completion = " " + str(s["label"])
            rows.append({"prompt": prompt, "completion": completion})
        return HFDataset.from_list(rows)

    def as_tokenized_hf_dataset(self, tokenizer, max_length: int) -> HFDataset:
        """
        Pre-tokenise prompt+completion with completion-only labels.

        Unsloth's patched SFTTrainer does not accept ``prompt``/``completion``
        columns (it requires ``text`` or a ``formatting_func``, which is
        incompatible with ``completion_only_loss``).  Pre-tokenising ourselves
        and providing ``input_ids`` / ``attention_mask`` / ``labels`` lets
        Unsloth skip its dataset prep while still masking the prompt in the loss
        (prompt tokens get label ``-100``).
        """
        print(
            f"  Tokenising {len(self.samples):,} samples "
            f"(prompt masked, loss on answer token only)...",
            flush=True,
        )
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        rows = []
        for s in tqdm(self.samples, desc="Tokenising SFT samples", mininterval=5.0):
            prompt = self._format_prompt(s["text"])
            completion = " " + str(s["label"])

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)

            bos = [bos_id] if bos_id is not None else []
            eos = [eos_id] if eos_id is not None else []

            # Safety net: EHR was already truncated in __init__, but if the
            # assembled sequence is still too long, drop oldest prompt tokens.
            overhead = len(bos) + len(completion_ids) + len(eos)
            budget = max_length - overhead
            if budget <= 0:
                continue
            if len(prompt_ids) > budget:
                prompt_ids = prompt_ids[-budget:]

            input_ids = bos + prompt_ids + completion_ids + eos
            prompt_len = len(bos) + len(prompt_ids)
            labels = ([-100] * prompt_len) + list(completion_ids) + list(eos)

            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

        print(f"  Tokenised {len(rows):,} samples.", flush=True)
        return HFDataset.from_list(rows)

    def prompts_and_labels(self) -> Tuple[List[str], List[int]]:
        """
        Return (prompts_without_answer, binary_labels) for evaluation.
        Prompts are already truncated so the response template always
        falls at the final token position.
        """
        prompts = [self._format_prompt(s["text"]) for s in self.samples]
        labels = [s["label"] for s in self.samples]
        return prompts, labels

    def subset(self, max_samples: int, seed: int = 42) -> "_SFTDataset":
        """
        Return a shallow copy with at most ``max_samples`` examples (for cheap
        mid-training eval). Final val/test should keep the full set.
        """
        n = len(self.samples)
        if max_samples is None or max_samples <= 0 or max_samples >= n:
            return self
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(n), int(max_samples)))
        out = object.__new__(_SFTDataset)
        out.prompt_template = self.prompt_template
        out.task_description = self.task_description
        out.tokenizer = self.tokenizer
        out.max_length = self.max_length
        out._prompt_overhead_tokens = self._prompt_overhead_tokens
        out.samples = [self.samples[i] for i in indices]
        return out


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class SFTClassificationTrainer:
    """
    Orchestrates SFT-based classification:

    1. load_model()           — Unsloth FastLanguageModel + LoRA
    2. prepare_datasets()     — wrap MIMIC text datasets into SFT format
    3. train()                — SFTTrainer (loss only on answer token)
    4. evaluate_split()       — logit-based probability extraction
    5. run_full_pipeline()    — run all steps, save results
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config["model"]
        self.data_config = config["data"]
        self.training_config = config["training"]
        self.lora_config = config.get("lora", {})
        self.wandb_config = config.get("wandb", {})
        self.prompt_config = config.get("prompt", {})
        self.task_config = config.get("task", {})

        task_desc = self.task_config.get("description", "")
        if not task_desc or task_desc.startswith("REQUIRED"):
            raise ValueError(
                "task.description must be explicitly set in the config (e.g. "
                "'die in the ICU' or 'be readmitted to hospital')."
            )
        self.task_description = task_desc

        self.logit_low_threshold = float(
            self.model_config.get("logit_low_threshold", -10.0)
        )

        self.prompt_template = _load_prompt_template(self.prompt_config)
        self.output_dir = self.training_config.get("output_dir", "./outputs/sft_classifier")

        self.model = None
        self.tokenizer = None
        self._token_id_0: int | None = None
        self._token_id_1: int | None = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------

    def setup_wandb(self):
        if self.wandb_config.get("enabled", False) and self.local_rank == 0:
            wandb.init(
                project=self.wandb_config.get("project", "MIMIC-IV-SFT-classification"),
                name=self.wandb_config.get("run_name", "sft-classifier"),
                config=self.config,
            )
            logger.info(
                f"WandB enabled — project: {self.wandb_config['project']}, "
                f"run: {self.wandb_config.get('run_name')}"
            )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        """
        Load model and apply LoRA for SFT classification.
        Set ``model.use_unsloth: false`` in config to use the HF+BitsAndBytes
        path (required for MedGemma and other non-Unsloth models).
        """
        logger.info("=" * 80)
        logger.info("Loading model for SFT classification...")
        logger.info("=" * 80)

        pretrained_checkpoint = self.model_config.get("pretrained_checkpoint")
        model_name = self.model_config.get("model_name", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
        max_length = int(self.model_config.get("max_length", 8192))
        load_in_4bit = bool(self.model_config.get("load_in_4bit", True))
        use_unsloth = bool(self.model_config.get("use_unsloth", True))

        source = pretrained_checkpoint if pretrained_checkpoint else model_name
        logger.info(f"  - Source: {source}  (use_unsloth={use_unsloth})")

        if use_unsloth:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=source,
                max_seq_length=max_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
                device_map={"": self.local_rank},
            )

            # Apply (or re-apply) LoRA for SFT
            target_modules = self.lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            # Under DDP, Unsloth's "unsloth" (reentrant) checkpointing marks LoRA
            # params ready twice and crashes on the first backward. Use standard
            # non-reentrant checkpointing instead (same as pretrain).
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=int(self.lora_config.get("r", 16)),
                target_modules=target_modules,
                lora_alpha=int(self.lora_config.get("lora_alpha", 32)),
                lora_dropout=float(self.lora_config.get("lora_dropout", 0.05)),
                bias="none",
                use_gradient_checkpointing=True,
                random_state=42,
            )
        else:
            from src.training.model_loader import load_hf_model, load_pretrained_lora_model_hf, apply_lora_peft
            if pretrained_checkpoint:
                self.model, self.tokenizer = load_pretrained_lora_model_hf(
                    pretrained_checkpoint=pretrained_checkpoint,
                    max_seq_length=max_length,
                    load_in_4bit=load_in_4bit,
                    local_rank=self.local_rank,
                )
            else:
                self.model, self.tokenizer = load_hf_model(
                    model_name=model_name,
                    max_seq_length=max_length,
                    load_in_4bit=load_in_4bit,
                    local_rank=self.local_rank,
                )
            self.model = apply_lora_peft(self.model, self.lora_config)

        # Ensure a pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache vocab IDs for answer tokens "0" / "1" (see resolve_binary_label_token_ids).
        self._token_id_0, self._token_id_1 = resolve_binary_label_token_ids(
            self.tokenizer
        )
        logger.info(
            f"  - Token IDs: '0' → {self._token_id_0}, '1' → {self._token_id_1}"
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"  - Total parameters:     {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")

    def load_model_from_checkpoint(self, checkpoint_path: str):
        """
        Load base model + LoRA adapters from a Trainer ``checkpoint-*`` directory.
        Used by eval-only runs (no training).
        """
        logger.info("=" * 80)
        logger.info("Loading SFT model from checkpoint...")
        logger.info("=" * 80)
        logger.info(f"  - Checkpoint: {checkpoint_path}")

        max_length = int(self.model_config.get("max_length", 8192))
        load_in_4bit = bool(self.model_config.get("load_in_4bit", True))
        use_unsloth = bool(self.model_config.get("use_unsloth", True))

        if use_unsloth:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=max_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
                device_map={"": self.local_rank},
            )
            FastLanguageModel.for_inference(self.model)
        else:
            from src.training.model_loader import load_pretrained_lora_model_hf

            self.model, self.tokenizer = load_pretrained_lora_model_hf(
                pretrained_checkpoint=checkpoint_path,
                max_seq_length=max_length,
                load_in_4bit=load_in_4bit,
                local_rank=self.local_rank,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._token_id_0, self._token_id_1 = resolve_binary_label_token_ids(
            self.tokenizer
        )
        logger.info(
            f"  - Token IDs: '0' → {self._token_id_0}, '1' → {self._token_id_1}"
        )

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _load_split(
        self,
        split: str,
        *,
        file_fraction: float | None = None,
        file_fraction_label: str | None = None,
    ) -> "_SFTDataset":
        data_cfg = dict(self.data_config)
        data_cfg["format"] = "text"
        data_cfg["task"] = "prediction"
        frac = file_fraction
        if frac is None and split == "train":
            train_frac = data_cfg.get("train_file_fraction")
            if train_frac is not None and float(train_frac) < 1.0:
                frac = float(train_frac)
                file_fraction_label = file_fraction_label or "train"
        if frac is not None and float(frac) < 1.0:
            data_cfg["file_fraction"] = float(frac)
            data_cfg["file_fraction_seed"] = int(
                self.training_config.get("seed", 42)
            )
            label = file_fraction_label or split
            print(
                f"  Using {float(frac) * 100:.0f}% of {label} pickle shards",
                flush=True,
            )
        base = get_dataset(data_cfg.get("dataset", "MIMIC"), data_cfg, split)
        max_length = int(self.model_config.get("max_length", 8192))
        return _SFTDataset(
            base,
            self.prompt_template,
            self.task_description,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

    def prepare_datasets(self) -> Dict[str, "_SFTDataset"]:
        print("=" * 80, flush=True)
        print("Preparing SFT datasets...", flush=True)
        print("=" * 80, flush=True)
        max_length = int(self.model_config.get("max_length", 8192))
        print(
            f"  EHR texts will be pre-truncated to fit within {max_length} tokens "
            f"(oldest events dropped, demographics header + recent events kept)",
            flush=True,
        )
        splits = {}
        for split in ("train", "tuning", "held_out"):
            print(f"  - Loading {split} shards + building prediction examples...", flush=True)
            splits[split] = self._load_split(split)
            print(f"    {len(splits[split].samples):,} samples ready", flush=True)
        return splits

    def prepare_eval_splits(self) -> Dict[str, "_SFTDataset"]:
        """Load tuning + held_out only (skips train) for eval-only runs."""
        print("=" * 80, flush=True)
        print("Preparing SFT eval datasets (tuning + held_out)...", flush=True)
        print("=" * 80, flush=True)
        splits = {}
        for split in ("tuning", "held_out"):
            print(f"  - Loading {split} shards + building prediction examples...", flush=True)
            splits[split] = self._load_split(split)
            print(f"    {len(splits[split].samples):,} samples ready", flush=True)
        return splits

    def _mid_train_eval_dataset(self, full_tuning: "_SFTDataset") -> "_SFTDataset":
        """
        Build a cheap mid-training eval set.

        Preference order:
          1. ``data.eval_file_fraction`` — reload tuning from a shard subset
          2. ``training.eval_max_samples`` — subsample already-loaded tuning
          3. else use the full tuning set

        Final ``evaluate_split`` on tuning/held_out always uses the full sets.
        """
        eval_frac = self.data_config.get("eval_file_fraction")
        if eval_frac is not None and float(eval_frac) < 1.0:
            print(
                f"  Mid-train eval: reloading tuning with "
                f"eval_file_fraction={float(eval_frac)}",
                flush=True,
            )
            return self._load_split(
                "tuning",
                file_fraction=float(eval_frac),
                file_fraction_label="mid-train tuning",
            )

        max_samples = self.training_config.get("eval_max_samples")
        if max_samples is not None:
            n = int(max_samples)
            subset = full_tuning.subset(
                n, seed=int(self.training_config.get("seed", 42))
            )
            if subset is not full_tuning:
                print(
                    f"  Mid-train eval: subsampled {len(subset.samples):,} / "
                    f"{len(full_tuning.samples):,} tuning samples "
                    f"(eval_max_samples={n})",
                    flush=True,
                )
            return subset

        return full_tuning

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, splits: Dict[str, "_SFTDataset"], resume_from_checkpoint: str | None = None):
        """Run SFT via TRL/Unsloth SFTTrainer (loss computed only on the answer token)."""
        print("=" * 80, flush=True)
        print("Training with SFTTrainer...", flush=True)
        print("=" * 80, flush=True)

        max_length = int(self.model_config.get("max_length", 8192))
        mid_eval = self._mid_train_eval_dataset(splits["tuning"])
        print(
            f"  Building pre-tokenised datasets "
            f"(train={len(splits['train'].samples):,}, "
            f"mid_eval={len(mid_eval.samples):,} / "
            f"full_tuning={len(splits['tuning'].samples):,})...",
            flush=True,
        )
        # Pre-tokenise with prompt labels masked to -100.  Unsloth's SFTTrainer
        # does not accept prompt/completion columns (requires formatting_func,
        # which is incompatible with completion_only_loss), so we prepare the
        # tokenised form ourselves and let Unsloth skip dataset prep.
        train_hf = splits["train"].as_tokenized_hf_dataset(self.tokenizer, max_length)
        val_hf = mid_eval.as_tokenized_hf_dataset(self.tokenizer, max_length)
        print("  Tokenised datasets ready.", flush=True)

        report_to = "wandb" if self.wandb_config.get("enabled") else "none"
        eval_steps = int(self.training_config.get("eval_steps", 2000))
        save_steps = int(self.training_config.get("save_steps", eval_steps))

        print(
            f"  Eval schedule: every {eval_steps} steps on "
            f"{len(mid_eval.samples):,} samples (CE loss + logit AUROC)",
            flush=True,
        )
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            overwrite_output_dir=bool(self.training_config.get("overwrite_output_dir", False)),
            num_train_epochs=int(self.training_config.get("epochs", 1)),
            per_device_train_batch_size=int(self.training_config.get("batch_size", 4)),
            per_device_eval_batch_size=int(self.training_config.get("eval_batch_size", 4)),
            gradient_accumulation_steps=int(
                self.training_config.get("gradient_accumulation_steps", 2)
            ),
            learning_rate=float(self.training_config.get("learning_rate", 2e-5)),
            weight_decay=float(self.training_config.get("weight_decay", 0.01)),
            warmup_steps=int(self.training_config.get("warmup_steps", 100)),
            fp16=bool(self.training_config.get("fp16", False)),
            bf16=bool(self.training_config.get("bf16", True)),
            logging_steps=int(self.training_config.get("logging_steps", 10)),
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=int(self.training_config.get("save_total_limit", 2)),
            load_best_model_at_end=False,
            report_to=report_to,
            max_length=max_length,
            # Labels already mask the prompt (-100); do not ask TRL/Unsloth to
            # re-derive completion-only loss from prompt/completion columns.
            completion_only_loss=False,
            packing=False,
            dataloader_num_workers=int(self.training_config.get("dataloader_num_workers", 0)),
            remove_unused_columns=False,
            # DDP + checkpointing: non-reentrant required (see isa-mimic-run-5975278).
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
        )

        print("  Constructing SFTTrainer...", flush=True)
        sft_callbacks: List[Any] = [
            _LogitClassificationCallback(
                self,
                mid_eval,
                batch_size=int(self.training_config.get("eval_batch_size", 4)),
            )
        ]
        append_max_runtime_callback(sft_callbacks, self.training_config)
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            args=sft_config,
            callbacks=sft_callbacks,
        )
        print("  SFTTrainer ready — starting training loop.", flush=True)

        if resume_from_checkpoint:
            print(f"  Resuming from checkpoint: {resume_from_checkpoint}", flush=True)
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
        self.trainer = trainer

        # ── FLOPs reporting ─────────────────────────────────────────────────
        try:
            from src.training.flops_counter import report_flops_from_trainer
            report_flops_from_trainer(
                trainer=self.trainer,
                model=self.model,
                seq_length=max_length,
                output_dir=self.training_config.get("output_dir"),
                local_rank=self.local_rank,
            )
        except Exception as _flops_err:
            logger.warning(f"FLOPs reporting failed (non-fatal): {_flops_err}")

    # ------------------------------------------------------------------
    # Logit-based evaluation
    # ------------------------------------------------------------------

    def _logits_to_prob(self, logit_0: float, logit_1: float) -> float:
        """Convert next-token logits for '0' and '1' to positive-class probability."""
        if max(logit_0, logit_1) < self.logit_low_threshold:
            return 0.5
        pair = torch.tensor([logit_0, logit_1], dtype=torch.float32)
        return float(torch.softmax(pair, dim=0)[1].item())

    def _predict_probs(
        self,
        sft_dataset: "_SFTDataset",
        batch_size: int = 4,
        split_name: str = "eval",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward-pass logit extraction → P(y=1) for every sample."""
        prompts, true_labels = sft_dataset.prompts_and_labels()

        was_training = self.model.training
        self.model.eval()
        probs = []
        low_logit_fallbacks = 0
        max_length = int(self.model_config.get("max_length", 8192))

        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Eval [{split_name}]"):
                batch_prompts = prompts[i: i + batch_size]

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.model.device)

                # Avoid default fp16 autocast with bf16 4-bit MedGemma (NaN logits).
                outputs = self.model(**inputs)

                # Last non-pad token (correct for left- and right-padding).
                last_logits = outputs.logits.float()  # (B, T, V)
                last_indices = inputs["attention_mask"].long().cumsum(dim=1).argmax(dim=1)
                batch_sz = last_logits.size(0)
                next_logits = last_logits[range(batch_sz), last_indices, :]

                logit_0 = next_logits[:, self._token_id_0].cpu().float()
                logit_1 = next_logits[:, self._token_id_1].cpu().float()

                for j in range(batch_sz):
                    p = self._logits_to_prob(logit_0[j].item(), logit_1[j].item())
                    if p == 0.5 and max(logit_0[j].item(), logit_1[j].item()) < self.logit_low_threshold:
                        low_logit_fallbacks += 1
                    probs.append(p)

        if was_training:
            self.model.train()

        if len(prompts) > 0:
            fallback_rate = 100.0 * low_logit_fallbacks / len(prompts)
            print(
                f"  Logit fallbacks: {low_logit_fallbacks}/{len(prompts)} "
                f"({fallback_rate:.1f}%) defaulted to 0.5",
                flush=True,
            )

        return (
            np.array(true_labels, dtype=np.int32),
            np.array(probs, dtype=np.float32),
        )

    def compute_logit_metrics(
        self,
        sft_dataset: "_SFTDataset",
        batch_size: int = 4,
        split_name: str = "eval",
    ) -> Dict[str, float]:
        """AUROC / AUPR / ECE / accuracy / F1 via logit-based P(y=1)."""
        labels_arr, probs_arr = self._predict_probs(
            sft_dataset, batch_size=batch_size, split_name=split_name
        )
        return compute_baseline_metrics(labels_arr, probs_arr)

    def evaluate_split(
        self,
        sft_dataset: "_SFTDataset",
        split_name: str,
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        Evaluate on one split using logit-based probability extraction.

        Args:
            sft_dataset: The _SFTDataset for this split.
            split_name:  Name for logging / file output.
            batch_size:  Inference batch size.

        Returns:
            Dict of metric names to values.
        """
        print(
            f"\nEvaluating [{split_name}] ({len(sft_dataset.samples)} samples)...",
            flush=True,
        )

        labels_arr, probs_arr = self._predict_probs(
            sft_dataset, batch_size=batch_size, split_name=split_name
        )
        metrics = compute_baseline_metrics(labels_arr, probs_arr)

        plot_dir = os.path.join(self.output_dir, "plots", split_name)
        plot_all_curves(labels_arr, probs_arr, plot_dir)

        results_dir = os.path.join(self.output_dir, "results")
        save_results(
            {"metrics": metrics, "labels": labels_arr.tolist(), "probs": probs_arr.tolist()},
            results_dir,
            f"{split_name}_results.json",
        )

        print_results(metrics, split_name)
        if self.wandb_config.get("enabled", False) and self.local_rank == 0:
            wandb.log({f"{split_name}_{k}": float(v) for k, v in metrics.items()})

        return metrics

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(self, resume_from_checkpoint: str | None = None):
        """Run training then evaluate on tuning and held_out splits."""
        os.makedirs(self.output_dir, exist_ok=True)

        self.setup_wandb()
        self.load_model()
        splits = self.prepare_datasets()
        self.train(splits, resume_from_checkpoint=resume_from_checkpoint)

        # Switch to inference mode (Unsloth optimisation — skip for HF backend)
        if self.model_config.get("use_unsloth", True):
            FastLanguageModel.for_inference(self.model)

        eval_batch_size = int(self.training_config.get("eval_batch_size", 4))
        val_metrics = self.evaluate_split(splits["tuning"], "tuning", batch_size=eval_batch_size)
        test_metrics = self.evaluate_split(splits["held_out"], "held_out", batch_size=eval_batch_size)

        print(
            f"\nFinal metrics — "
            f"val AUROC={val_metrics['auroc']:.4f}, "
            f"val AUPR={val_metrics['aupr']:.4f}, "
            f"test AUROC={test_metrics['auroc']:.4f}, "
            f"test AUPR={test_metrics['aupr']:.4f}",
            flush=True,
        )

        summary = {"validation": val_metrics, "test": test_metrics}
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}", flush=True)

        if self.wandb_config.get("enabled") and self.local_rank == 0:
            wandb.summary.update(
                {f"final_tuning_{k}": v for k, v in val_metrics.items()}
                | {f"final_held_out_{k}": v for k, v in test_metrics.items()}
            )
            wandb.finish()

        return val_metrics, test_metrics

    def _write_eval_provenance(
        self,
        checkpoint_path: str,
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
    ):
        if self.local_rank != 0:
            return

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
            "produced_by": "src.pipelines.evaluate_checkpoint (sft)",
            "note": (
                "Metrics come from the checkpoint above (tuning + held_out), "
                "not from a training run that reached its final step."
            ),
            "validation_auroc": val_metrics.get("auroc"),
            "test_auroc": test_metrics.get("auroc"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "eval_provenance.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  - Provenance written to {path}", flush=True)

    def run_eval_only(self, checkpoint_path: str):
        """
        Score an existing SFT checkpoint on tuning (validation) and held_out
        (test), with no training.

        Recovers metrics for runs killed after their last checkpoint but before
        the final evaluation. The train split is never loaded.
        """
        if self.local_rank != 0:
            return None, None

        os.makedirs(self.output_dir, exist_ok=True)

        print("\n" + "=" * 80, flush=True)
        print("SFT evaluation-only run (tuning + held_out, no training)", flush=True)
        print(f"  - Checkpoint: {checkpoint_path}", flush=True)
        print("=" * 80, flush=True)

        self.training_config["overwrite_output_dir"] = False

        self.setup_wandb()
        self.load_model_from_checkpoint(checkpoint_path)
        splits = self.prepare_eval_splits()

        eval_batch_size = int(self.training_config.get("eval_batch_size", 4))
        val_metrics = self.evaluate_split(
            splits["tuning"], "tuning", batch_size=eval_batch_size
        )
        test_metrics = self.evaluate_split(
            splits["held_out"], "held_out", batch_size=eval_batch_size
        )

        print(
            f"\nFinal metrics — "
            f"val AUROC={val_metrics['auroc']:.4f}, "
            f"val AUPR={val_metrics['aupr']:.4f}, "
            f"test AUROC={test_metrics['auroc']:.4f}, "
            f"test AUPR={test_metrics['aupr']:.4f}",
            flush=True,
        )

        summary = {"validation": val_metrics, "test": test_metrics}
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}", flush=True)

        self._write_eval_provenance(checkpoint_path, val_metrics, test_metrics)

        if self.wandb_config.get("enabled", False):
            wandb.summary.update(
                {f"final_tuning_{k}": v for k, v in val_metrics.items()}
                | {f"final_held_out_{k}": v for k, v in test_metrics.items()}
            )
            wandb.finish()

        return val_metrics, test_metrics
