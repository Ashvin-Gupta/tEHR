"""
Event-token Mistral: one vocabulary id per clinical event, no text translation.

Stage 1 continued-pretrains Mistral-7B on those ids. Stage 2 is the usual
LoRA classifier. Each patient contributes the events whose tEHR text survives
the 12000-token tail window, capped so the event sequence fits in 2048.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.dataset_factory import get_dataset
from src.data.event_token_align import is_event_code, recent_events
from src.training.classification_trainer import TehrClassifierTrainer
from src.training.pretrainer_trainer import TehrStage1Trainer

import logging

logger = logging.getLogger(__name__)


def _wait_for(path: str, poll_s: float = 5.0) -> None:
    while not os.path.exists(path):
        logger.info("Waiting for %s", path)
        time.sleep(poll_s)


def _adapter_weight_keys(checkpoint: str) -> List[str]:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = os.path.join(checkpoint, name)
        if not os.path.exists(path):
            continue
        if name.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt") as handle:
                return list(handle.keys())
        state = torch.load(path, map_location="cpu", weights_only=True)
        return list(state.keys())
    return []


def _require_trained_embeddings(checkpoint: str) -> None:
    keys = _adapter_weight_keys(checkpoint)
    if not any("embed_tokens" in key or "lm_head" in key for key in keys):
        raise RuntimeError(
            f"Stage-1 checkpoint {checkpoint} has no embed_tokens/lm_head "
            "weights. Event tokens would stay at their initial values."
        )


def _token_config(data_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(data_config)
    cfg["format"] = "tokens"
    cfg["max_events"] = None
    return cfg


def _string_codes(dataset, item: Dict[str, Any]) -> List[str]:
    return [str(dataset.id_to_token.get(int(token_id), "")) for token_id in item["tokens"]]


def _label_of(item: Dict[str, Any]) -> int:
    label = item["label"]
    if hasattr(label, "item"):
        label = label.item()
    return int(label)


def _pack_ids(sequences: List[List[int]], chunk_size: int) -> List[List[int]]:
    flat: List[int] = []
    for ids in sequences:
        flat.extend(ids)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [
        flat[start:start + chunk_size]
        for start in range(0, len(flat) - chunk_size + 1, chunk_size)
    ]


class _EventRows(TorchDataset):
    """Stage-2 rows already converted to event-token ids."""

    def __init__(self, table: Dataset):
        self.table = table
        self.examples = [
            {"subject_id": table[i]["subject_id"]} for i in range(len(table))
        ]

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.table[idx]
        return {
            "input_ids": list(row["input_ids"]),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject_id": row["subject_id"],
        }


class EventTokenStage1Trainer(TehrStage1Trainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_name = str((config.get("_meta") or {}).get("model") or "")
        if model_name != "mistral":
            raise ValueError(
                "event_token_stage1 is Mistral-7B only "
                f"(got {model_name or 'no model'})."
            )
        if "event_seq_length" not in self.model_config:
            raise ValueError("model.event_seq_length is required (2048).")
        self.model_config["max_length"] = int(self.model_config["event_seq_length"])
        self._cache_dir = os.path.join(
            self.training_config["output_dir"],
            f"event_token_cache_cap{self.model_config['max_length']}",
        )

    def load_model(self):
        super().load_model()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self._build_event_cache()
        vocab_path = os.path.join(self._cache_dir, "vocab.json")
        if self.local_rank != 0:
            _wait_for(os.path.join(self._cache_dir, "READY"))
        with open(vocab_path) as handle:
            event_tokens = json.load(handle)
        if not event_tokens:
            raise RuntimeError(f"No event tokens in {vocab_path}")
        logger.info("Adding %s event tokens to the Mistral vocabulary", f"{len(event_tokens):,}")
        print(f"Adding {len(event_tokens):,} event tokens to the Mistral vocabulary", flush=True)
        from unsloth_zoo.tokenizer_utils import add_new_tokens
        add_new_tokens(self.model, self.tokenizer, event_tokens)
        unk_id = self.tokenizer.unk_token_id
        missing = [
            token for token in event_tokens
            if self.tokenizer.convert_tokens_to_ids(token) in (None, unk_id)
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing):,} event codes did not become single vocabulary "
                f"ids. Examples: {missing[:5]}"
            )

    def apply_lora(self):
        from unsloth import FastLanguageModel

        logger.info("Applying LoRA and training the event-token embeddings")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_config.get("r", 16),
            target_modules=self.lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            lora_alpha=self.lora_config.get("lora_alpha", 16),
            lora_dropout=self.lora_config.get("lora_dropout", 0.05),
            bias=self.lora_config.get("bias", "none"),
            modules_to_save=["embed_tokens", "lm_head"],
            ensure_weight_tying=True,
            use_gradient_checkpointing=self.training_config.get("gradient_checkpointing", "unsloth"),
            random_state=42,
            use_rslora=self.lora_config.get("use_rslora", True),
            loftq_config=self.lora_config.get("loftq_config", None),
        )
        embedding = self.model.get_input_embeddings()
        weight = embedding.weight
        saved = getattr(embedding, "modules_to_save", None)
        if saved is not None:
            weight = saved.default.weight if hasattr(saved, "default") else saved["default"].weight
        if not weight.requires_grad:
            raise RuntimeError(
                "Event-token embeddings are frozen after LoRA setup. "
                "Stage 1 would not learn the new tokens."
            )
        if int(weight.shape[0]) != len(self.tokenizer):
            raise RuntimeError(
                f"Embedding rows {tuple(weight.shape)} do not match tokenizer "
                f"size {len(self.tokenizer)}."
            )
        print(
            f"  - Event embeddings trainable: {tuple(weight.shape)} "
            f"requires_grad={weight.requires_grad}",
            flush=True,
        )

    def _build_event_cache(self) -> None:
        ready = os.path.join(self._cache_dir, "READY")
        if self.local_rank != 0:
            return
        if os.path.exists(ready):
            logger.info("Event-token cache exists: %s", self._cache_dir)
            return
        os.makedirs(self._cache_dir, exist_ok=True)
        cap = int(self.model_config["max_length"])
        print(
            f"Building event-token cache (most recent {cap} events, full splits)",
            flush=True,
        )
        vocab = set()
        token_cfg = _token_config(self.data_config)
        for split, store in (("train", True), ("tuning", True), ("held_out", False)):
            dataset = get_dataset(self.data_config["dataset"], token_cfg, split)
            windows: List[List[str]] = []
            n_capped = 0
            for index in tqdm(range(len(dataset)), desc=f"events {split}"):
                item = dataset[index]
                if not item:
                    continue
                codes = _string_codes(dataset, item)
                for code in codes:
                    if is_event_code(code):
                        vocab.add(code)
                if store:
                    events = recent_events(codes, cap)
                    if len(events) == cap:
                        n_capped += 1
                    windows.append(events)
            if store:
                Dataset.from_dict({"events": windows}).save_to_disk(
                    os.path.join(self._cache_dir, split)
                )
                lengths = [len(row) for row in windows]
                mean = sum(lengths) / max(1, len(lengths))
                print(
                    f"  - {split}: {len(windows):,} timelines, "
                    f"mean events {mean:.1f}, "
                    f"max {max(lengths) if lengths else 0}, "
                    f"at cap {n_capped:,}",
                    flush=True,
                )
                del windows
            del dataset
            gc.collect()
        tokens = sorted(vocab)
        vocab_path = os.path.join(self._cache_dir, "vocab.json")
        tmp = vocab_path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(tokens, handle)
        os.replace(tmp, vocab_path)
        print(f"  - Vocabulary: {len(tokens):,} event codes", flush=True)
        with open(ready, "w") as handle:
            handle.write("ok\n")

    def prepare_datasets(self):
        max_length = int(self.model_config["max_length"])
        cache_root = self.training_config.get("data_cache_dir") or self.training_config["output_dir"]
        cache_dir = os.path.join(cache_root, "data_cache")
        train_cache = os.path.join(cache_dir, f"train_packed_len{max_length}")
        val_cache = os.path.join(cache_dir, f"val_packed_len{max_length}")
        train_info = os.path.join(train_cache, "dataset_info.json")

        if self.local_rank == 0 and not os.path.exists(train_info):
            os.makedirs(cache_dir, exist_ok=True)
            self._pack_split("train", train_cache, max_length)
            self._pack_split("tuning", val_cache, max_length)
        if self.local_rank != 0:
            _wait_for(train_info)

        train_dataset = Dataset.load_from_disk(train_cache)
        val_dataset = Dataset.load_from_disk(val_cache) if os.path.exists(val_cache) else None
        if len(train_dataset) == 0:
            raise RuntimeError(f"No packed training sequences in {train_cache}")
        observed = len(train_dataset[0]["input_ids"])
        if observed != max_length:
            raise RuntimeError(
                f"Packed length {observed} != max_length {max_length} at {train_cache}"
            )
        return train_dataset, val_dataset

    def _pack_split(self, split: str, cache_path: str, max_length: int) -> None:
        table = Dataset.load_from_disk(os.path.join(self._cache_dir, split))
        eos_id = self.tokenizer.eos_token_id
        unk_id = self.tokenizer.unk_token_id
        sequences: List[List[int]] = []
        lengths = []
        for row in tqdm(table, desc=f"pack {split}"):
            ids: List[int] = []
            for code in row["events"]:
                token_id = self.tokenizer.convert_tokens_to_ids(code)
                if token_id is None or token_id == unk_id:
                    continue
                ids.append(int(token_id))
            if eos_id is not None:
                ids.append(int(eos_id))
            if not ids:
                continue
            sequences.append(ids)
            lengths.append(len(ids))
        packed = _pack_ids(sequences, max_length)
        if split == "train" and not packed:
            raise RuntimeError(f"Packing {split} produced no {max_length}-token chunks")
        if packed:
            Dataset.from_dict({"input_ids": packed}).save_to_disk(cache_path)
        mean = sum(lengths) / max(1, len(lengths))
        print(
            f"  - {split}: {len(sequences):,} timelines, mean length {mean:.1f}, "
            f"packed chunks {len(packed):,}",
            flush=True,
        )

    def save_model(self):
        super().save_model()
        final_dir = os.path.join(
            self.training_config["output_dir"],
            self.training_config.get("final_subdir", "final_model"),
        )
        vocab_src = os.path.join(self._cache_dir, "vocab.json")
        if os.path.exists(vocab_src):
            shutil.copy(vocab_src, os.path.join(final_dir, "event_vocab.json"))
        if self.local_rank == 0:
            _require_trained_embeddings(final_dir)
            print(f"  - Saved event vocabulary and embedding weights to {final_dir}", flush=True)


class EventTokenStage2Trainer(TehrClassifierTrainer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_name = str((config.get("_meta") or {}).get("model") or "")
        if model_name != "mistral":
            raise ValueError(
                "event_token_stage2 is Mistral-7B only "
                f"(got {model_name or 'no model'})."
            )
        if "event_seq_length" not in self.model_config:
            raise ValueError("model.event_seq_length is required (2048).")
        self.model_config["max_length"] = int(self.model_config["event_seq_length"])

    def load_pretrained_model(self):
        checkpoint = self.model_config.get("pretrained_checkpoint")
        if not checkpoint:
            raise ValueError("event_token_stage2 requires the stage-1 checkpoint.")
        _require_trained_embeddings(checkpoint)
        n_vocab = len(AutoTokenizer.from_pretrained(checkpoint))
        print(
            f"Loading event-token stage 1 from {checkpoint} "
            f"(vocab {n_vocab:,}, max_length {self.model_config['max_length']})",
            flush=True,
        )
        from unsloth import FastLanguageModel
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=self.model_config["max_length"],
            dtype=None,
            load_in_4bit=self.training_config.get("load_in_4bit", True),
            device_map={"": self.local_rank},
            resize_model_vocab=n_vocab,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        rows = int(self.base_model.get_input_embeddings().weight.shape[0])
        if rows != len(self.tokenizer):
            raise RuntimeError(
                f"Loaded embedding rows {rows} != tokenizer size {len(self.tokenizer)}. "
                "The stage-1 event vocabulary did not resize onto the base model."
            )
        print(f"  - Tokenizer vocab {len(self.tokenizer):,}, embedding rows {rows:,}", flush=True)

    def _row_cache(self, split: str) -> str:
        added = self.tokenizer.get_added_vocab()
        blob = "\n".join(f"{token}\t{index}" for token, index in sorted(added.items()))
        digest = hashlib.sha1(blob.encode()).hexdigest()[:12]
        root = self.training_config.get("data_cache_dir") or self.training_config["output_dir"]
        return os.path.join(
            root,
            f"event_rows_{digest}_cap{self.model_config['max_length']}",
            split,
        )

    def prepare_datasets(self):
        print("\n" + "=" * 80)
        print("Building event-token classification rows")
        print("=" * 80)
        splits = {}
        for split in ("train", "tuning", "held_out"):
            cache = self._row_cache(split)
            info = os.path.join(cache, "dataset_info.json")
            if self.local_rank == 0 and not os.path.exists(info):
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                dataset = get_dataset(
                    self.data_config["dataset"],
                    _token_config(self.data_config),
                    split,
                )
                self._write_rows(dataset, cache, split)
            if self.local_rank != 0:
                _wait_for(info)
            splits[split] = _EventRows(Dataset.load_from_disk(cache))
            self._log_class_distribution(splits[split], split)
        return splits["train"], splits["tuning"], splits["held_out"]

    def _write_rows(self, dataset, cache: str, split: str) -> None:
        max_length = int(self.model_config["max_length"])
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.eos_token_id
        unk_id = self.tokenizer.unk_token_id
        n_special = int(bos_id is not None) + int(eos_id is not None)
        event_cap = max(1, max_length - n_special)
        input_ids: List[List[int]] = []
        labels: List[int] = []
        subject_ids: List[str] = []
        n_events: List[int] = []
        n_capped = 0
        for index in tqdm(range(len(dataset)), desc=f"rows {split}"):
            item = dataset[index]
            if not item:
                continue
            events = recent_events(_string_codes(dataset, item), event_cap)
            if len(events) == event_cap:
                n_capped += 1
            ids: List[int] = []
            if bos_id is not None:
                ids.append(int(bos_id))
            for code in events:
                token_id = self.tokenizer.convert_tokens_to_ids(code)
                if token_id is None or token_id == unk_id:
                    continue
                ids.append(int(token_id))
            if eos_id is not None:
                ids.append(int(eos_id))
            input_ids.append(ids)
            labels.append(_label_of(item))
            subject_ids.append(str(item.get("subject_id", index)))
            n_events.append(len(events))
        Dataset.from_dict({
            "input_ids": input_ids,
            "label": labels,
            "subject_id": subject_ids,
            "n_events": n_events,
        }).save_to_disk(cache)
        mean_events = sum(n_events) / max(1, len(n_events))
        mean_len = sum(len(row) for row in input_ids) / max(1, len(input_ids))
        print(
            f"  - {split}: {len(input_ids):,} rows, "
            f"mean events {mean_events:.1f}, "
            f"mean sequence length {mean_len:.1f}, at cap {n_capped:,}",
            flush=True,
        )
