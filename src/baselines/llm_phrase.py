"""
Experiment: llm_phrase — frozen phrase embeddings projected into an LLM.

One clinical event is one embedding. The most recent 2048 events are
projected with a trainable LayerNorm + Linear into the LLM hidden size and
passed as ``inputs_embeds``. The LLM (existing stage-1 checkpoint, LoRA, and
a linear head) is the transformer. There is no pretraining stage in this
experiment: training is supervised only.

The frozen cache is not a parameter. Gradients update the projection, the
LoRA adapters, and the classification head.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.base_dataset import tokens_to_phrases
from src.models.llm_classifier import LLMClassifier
from src.training.classification_trainer import TehrClassifierTrainer


class PhraseLLMClassifier(LLMClassifier):
    """LLM classifier whose inputs are projected frozen event embeddings."""

    def __init__(self, base_model, hidden_size: int, embed_dim: int, **kwargs):
        super().__init__(base_model, hidden_size, **kwargs)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_size)
        print(
            f"  - Event projection: LayerNorm + Linear({embed_dim} → {hidden_size}), "
            "trainable"
        )

    def _embed_dtype(self) -> torch.dtype:
        dtype = getattr(self.base_model, "dtype", torch.bfloat16)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return torch.bfloat16
        return dtype

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        del input_ids
        if not self._lora_verified:
            lora_params = [
                (n, p) for n, p in self.base_model.named_parameters() if "lora_A" in n
            ]
            if lora_params:
                name, param = lora_params[0]
                print(
                    f"  [LoRA check] Found {len(lora_params)} lora_A parameters. "
                    f"Example: '{name}' shape={tuple(param.shape)} "
                    f"requires_grad={param.requires_grad}"
                )
            self._lora_verified = True

        hidden = embeddings.to(dtype=self.input_norm.weight.dtype)
        projected = self.input_proj(self.input_norm(hidden)).to(dtype=self._embed_dtype())
        outputs = self._forward_backbone(
            attention_mask=attention_mask,
            inputs_embeds=projected,
        )
        states = getattr(outputs, "last_hidden_state", None)
        if states is None:
            states = outputs.hidden_states[-1]
        logits = self.classifier(self._pool(states, attention_mask))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


class _PhraseEventDataset(Dataset):
    """Token timelines → last ``max_events`` frozen phrase rows."""

    def __init__(
        self,
        base_dataset,
        phrase_to_row: Dict[str, int],
        max_events: int,
    ):
        source = base_dataset
        while isinstance(source, torch.utils.data.Subset):
            source = source.dataset
        id_to_token = getattr(source, "id_to_token", None)
        translator = getattr(source, "translator", None)
        if id_to_token is None or translator is None:
            raise ValueError(
                "Dataset must expose id_to_token and translator for phrase lookup."
            )
        self.phrase_to_row = phrase_to_row
        self.max_events = int(max_events)
        self.samples: List[Dict[str, Any]] = []
        self.examples: List[Dict[str, Any]] = []
        n_phrases = 0
        n_hit = 0
        n_empty = 0

        for i in tqdm(range(len(base_dataset)), desc="Phrase events"):
            item = base_dataset[i]
            if item is None or item.get("tokens") is None or item.get("label") is None:
                continue
            tokens = item["tokens"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            codes = [str(id_to_token.get(int(tid), "")) for tid in tokens]
            rows: List[int] = []
            for phrase in tokens_to_phrases(
                codes, translator, include_header_split=False, drop_time=True
            ):
                cleaned = phrase.strip()
                if not cleaned:
                    continue
                n_phrases += 1
                row = phrase_to_row.get(cleaned)
                if row is None:
                    continue
                n_hit += 1
                rows.append(row)
            if len(rows) > self.max_events:
                rows = rows[-self.max_events :]
            if not rows:
                n_empty += 1
            label = item["label"]
            if hasattr(label, "item"):
                label = int(label.item())
            else:
                label = int(label)
            subject_id = item.get("subject_id", i)
            self.samples.append({"rows": rows, "label": 0 if label == 0 else 1})
            self.examples.append({"subject_id": subject_id})

        hit = (n_hit / n_phrases) if n_phrases else 1.0
        print(
            f"  - Phrase lookup: {n_hit:,}/{n_phrases:,} ({hit:.1%}); "
            f"{n_empty:,} sequences had no embedding; "
            f"cap {self.max_events}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "rows": sample["rows"],
            "label": sample["label"],
            "subject_id": self.examples[idx]["subject_id"],
        }


def _collate_phrase_batch(
    batch: List[Dict[str, Any]],
    embedding_matrix: torch.Tensor,
    max_events: int,
) -> Dict[str, torch.Tensor]:
    batch = [item for item in batch if item is not None and "label" in item]
    dim = int(embedding_matrix.shape[1])
    sequences = []
    for item in batch:
        rows = list(item["rows"])[-max_events:]
        if rows:
            sequences.append(embedding_matrix[torch.tensor(rows, dtype=torch.long)].float())
        else:
            sequences.append(torch.zeros(1, dim))
    width = max(seq.shape[0] for seq in sequences)
    embeddings = torch.zeros(len(sequences), width, dim)
    attention_mask = torch.zeros(len(sequences), width, dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        embeddings[i, :length] = seq
        attention_mask[i, :length] = 1
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    return {
        "embeddings": embeddings,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class LLMPhraseTrainer(TehrClassifierTrainer):
    """Supervised LLM on projected frozen event embeddings. No stage-1 training."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cap = self.model_config.get("event_seq_length")
        if cap is None:
            raise ValueError("model.event_seq_length is required (2048).")
        self.model_config["max_length"] = int(cap)
        self.embedding_matrix: Optional[torch.Tensor] = None
        self.phrase_to_row: Optional[Dict[str, int]] = None
        self.embed_dim: Optional[int] = None

    def _load_phrase_cache(self) -> None:
        path = str((self.config.get("embedder") or {}).get("precomputed_embeddings_path") or "")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Frozen phrase embeddings not found at: {path or '<unset>'}\n"
                "Run src/pipelines/precompute_token_embeddings.py first."
            )
        print(f"  - Loading frozen phrase embeddings: {path}")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if "phrase_to_row" not in cache:
            raise ValueError(
                f"Embeddings at {path} have no 'phrase_to_row'. "
                "Use the phrase-level cache."
            )
        self.embedding_matrix = cache["embeddings"].float()
        self.phrase_to_row = {str(k): int(v) for k, v in cache["phrase_to_row"].items()}
        self.embed_dim = int(cache["dim"])
        print(
            f"  - Frozen matrix {tuple(self.embedding_matrix.shape)} "
            f"from {cache.get('model_name', 'unknown')}"
        )

    def load_pretrained_model(self):
        self.model_config["max_length"] = int(self.model_config["event_seq_length"])
        super().load_pretrained_model()
        self._load_phrase_cache()
        hidden = int(self.model_config["hidden_size"])
        print(
            f"  - Supervised only: project {self.embed_dim} → {hidden}, "
            f"then the LLM over {self.model_config['max_length']} events"
        )

    def create_classifier(self):
        print("\n" + "=" * 80)
        print("Creating phrase-projected LLM classifier...")
        print("=" * 80)
        if self.embed_dim is None:
            raise RuntimeError("Load the phrase cache before creating the classifier.")
        train_lora = self.model_config.get("train_lora", False)
        freeze_llm = self.model_config.get("freeze_llm", True)
        trainable_keywords = ["lora_"] if train_lora else None
        if train_lora:
            print("  - Training mode: projection + LoRA + classification head")
        else:
            print("  - Training mode: projection + classification head")
        self.classifier_model = PhraseLLMClassifier(
            base_model=self.base_model,
            hidden_size=int(self.model_config["hidden_size"]),
            embed_dim=int(self.embed_dim),
            num_labels=int(self.model_config.get("num_labels", 2)),
            freeze_base=freeze_llm,
            trainable_param_keywords=trainable_keywords,
            pooling=self.model_config.get("pooling", "last_token"),
        )
        proj = sum(p.numel() for p in self.classifier_model.input_proj.parameters())
        proj += sum(p.numel() for p in self.classifier_model.input_norm.parameters())
        trainable = sum(
            p.numel() for p in self.classifier_model.parameters() if p.requires_grad
        )
        print(f"  - Projection parameters: {proj:,}")
        print(f"  - Trainable parameters:  {trainable:,}")
        print("  - Frozen phrase vectors stay fixed (they are batch inputs)")

    def _token_data_config(self) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.data_config)
        cfg["format"] = "tokens"
        cfg["max_events"] = None
        cfg["sort_by_length"] = False
        return cfg

    def _wrap(self, base_dataset) -> _PhraseEventDataset:
        if self.phrase_to_row is None:
            raise RuntimeError("Phrase cache is not loaded.")
        return _PhraseEventDataset(
            base_dataset,
            self.phrase_to_row,
            max_events=int(self.model_config["max_length"]),
        )

    def _load_token_split(self, split_name: str, label: str):
        saved = self.data_config
        self.data_config = self._token_data_config()
        try:
            base = super()._prediction_dataset(split_name, label)
        finally:
            self.data_config = saved
        return self._wrap(base)

    def prepare_datasets(self):
        print("\n" + "=" * 80)
        print("Loading event-embedding datasets...")
        print("=" * 80)
        train_dataset = self._load_token_split("train", "train")
        val_dataset = self._load_token_split("tuning", "validation")
        test_dataset = self._load_token_split("held_out", "test")
        print(f"  - Train dataset: {len(train_dataset)} patients")
        print(f"  - Validation dataset: {len(val_dataset)} patients")
        print(f"  - Test dataset: {len(test_dataset)} patients")
        return train_dataset, val_dataset, test_dataset

    def _prediction_dataset(self, split_name: str, label: str):
        return self._load_token_split(split_name, label)

    def create_collator(self):
        matrix = self.embedding_matrix
        max_events = int(self.model_config["max_length"])

        def _collate(batch):
            return _collate_phrase_batch(batch, matrix, max_events)

        return _collate
