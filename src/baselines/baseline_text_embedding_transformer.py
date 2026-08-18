"""
Baseline: Text-Embedding Transformer

Each patient trajectory is converted to natural-language *phrases* (same pairing
as the LLM path: measurable concept + value + optional unit), looked up in a
precomputed Qwen3-Embedding cache (last-token pooled), then fed to a transformer
encoder whose capacity matches the discrete encoder ablation:

  - hidden_dim=512 (Linear projection from embed_dim + LayerNorm)
  - pooling=mean (configurable)
  - head_type=linear (CrossEntropy)
  - 1 supervised epoch with matched eval_epoch_fractions

Precompute cache: see ``src/pipelines/precompute_token_embeddings.py``.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.baselines.baseline_utils import setup_output_dir
from src.data.base_dataset import tokens_to_phrases
from src.data.dataset_factory import get_dataset
from src.evaluations.evaluations import (
    compute_baseline_metrics,
    plot_all_curves,
    print_results,
    save_results,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _Head(nn.Module):
    """Classification head factory (matches build_head in encoder transformer)."""

    def __init__(
        self,
        head_type: str,
        hidden_dim: int,
        num_labels: int,
        head_hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if head_type == "linear":
            self.net = nn.Linear(hidden_dim, num_labels)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_size, num_labels),
            )
        elif head_type == "logistic":
            self.net = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(
                f"Unknown head_type: '{head_type}'. "
                "Options: 'linear', 'mlp', 'logistic'"
            )
        self.head_type = head_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextEmbeddingTransformer(nn.Module):
    """
    Transformer encoder on frozen phrase embeddings.

    Input:  (B, seq_len, embed_dim) precomputed vectors (typically unit-norm).
    Process:
        1. LayerNorm + Linear projection → hidden_dim
        2. Add small-init positional encodings
        3. TransformerEncoder
        4. Mean or CLS pooling → classification head
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        num_labels: int = 2,
        head_type: str = "linear",
        head_hidden_size: int = 256,
        pooling: str = "mean",
        pos_init_std: float = 0.02,
    ):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.head_type = head_type
        self.pooling = pooling

        # Adapt frozen (often unit-norm) embedder features into transformer scale.
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_proj = (
            nn.Identity()
            if embed_dim == hidden_dim
            else nn.Linear(embed_dim, hidden_dim)
        )

        # Small init so position does not drown frozen semantic features.
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_length + 1, hidden_dim) * pos_init_std
        )

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * pos_init_std)
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = _Head(head_type, hidden_dim, num_labels, head_hidden_size, dropout)

    def forward(
        self,
        embeddings: torch.Tensor,        # (B, seq_len, D)
        attention_mask: torch.Tensor,    # (B, seq_len)  1=attend, 0=pad
    ) -> torch.Tensor:
        batch_size, seq_len, _ = embeddings.shape

        if seq_len > self.max_seq_length:
            embeddings = embeddings[:, -self.max_seq_length :, :]
            attention_mask = attention_mask[:, -self.max_seq_length :]
            seq_len = self.max_seq_length

        x = self.input_proj(self.input_norm(embeddings))
        x = self.dropout(x)

        if self.pooling == "cls":
            # Tokens get positions 1..seq_len; CLS gets position 0.
            x = x + self.pos_encoding[1 : seq_len + 1, :].unsqueeze(0)
            cls = self.cls_token.expand(batch_size, -1, -1)
            cls = cls + self.pos_encoding[0:1, :].unsqueeze(0)
            x = torch.cat([cls, x], dim=1)

            cls_mask = torch.ones(
                batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device
            )
            full_mask = torch.cat([cls_mask, attention_mask], dim=1)
            src_key_padding_mask = full_mask == 0

            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            pooled = x[:, 0, :]
        else:
            x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
            src_key_padding_mask = attention_mask == 0
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            mask_expanded = attention_mask.float().unsqueeze(-1)
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Dataset with phrase embedding lookup
# ---------------------------------------------------------------------------

class _EmbeddingLookupDataset(Dataset):
    """
    Wrap a MIMIC/CPRD token dataset: token IDs → phrases → frozen embeddings.
    """

    def __init__(
        self,
        base_dataset,
        embedding_matrix: torch.Tensor,      # (N_phrases, D)
        phrase_to_row: Dict[str, int],
        max_seq_length: int = 512,
    ):
        self.embedding_matrix = embedding_matrix
        self.phrase_to_row = phrase_to_row
        self.max_seq_length = max_seq_length

        id_to_token = getattr(base_dataset, "id_to_token", None)
        translator = getattr(base_dataset, "translator", None)
        if id_to_token is None or translator is None:
            raise ValueError(
                "Base dataset must expose id_to_token and translator for phrase lookup."
            )
        self.id_to_token = id_to_token
        self.translator = translator

        self.samples: List[Dict[str, Any]] = []
        self.n_phrases_total = 0
        self.n_phrases_hit = 0
        self.n_empty = 0

        for i in range(len(base_dataset)):
            item = base_dataset[i]
            if item is None:
                continue
            tokens = item.get("tokens")
            label = item.get("label")
            if tokens is None or label is None:
                continue
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            binary_label = 0 if int(label) == 0 else 1
            self.samples.append({"tokens": list(tokens), "label": binary_label})

    def __len__(self) -> int:
        return len(self.samples)

    def _tokens_to_embedding_rows(self, token_ids: List[int]) -> List[int]:
        string_codes = [self.id_to_token.get(int(tid), "") for tid in token_ids]
        phrases = tokens_to_phrases(
            string_codes, self.translator, include_header_split=False
        )
        rows: List[int] = []
        for phrase in phrases:
            cleaned = phrase.strip()
            if not cleaned:
                continue
            self.n_phrases_total += 1
            row = self.phrase_to_row.get(cleaned)
            if row is not None:
                self.n_phrases_hit += 1
                rows.append(row)
        return rows

    def coverage(self) -> float:
        if self.n_phrases_total == 0:
            return 1.0
        return self.n_phrases_hit / self.n_phrases_total

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        rows = self._tokens_to_embedding_rows(s["tokens"])

        if not rows:
            self.n_empty += 1
            D = self.embedding_matrix.shape[1]
            emb = torch.zeros(1, D, dtype=torch.float32)
        else:
            rows = rows[-self.max_seq_length :]
            row_tensor = torch.tensor(rows, dtype=torch.long)
            emb = self.embedding_matrix[row_tensor].float()

        return {"embeddings": emb, "label": s["label"]}


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def _collate_fn(
    batch: List[Dict[str, Any]],
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    embeddings_list = [item["embeddings"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    D = embeddings_list[0].shape[1]
    lengths = [min(e.shape[0], max_seq_length) for e in embeddings_list]
    max_len = max(lengths)

    padded = torch.zeros(len(batch), max_len, D, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (emb, length) in enumerate(zip(embeddings_list, lengths)):
        padded[i, :length, :] = emb[-length:, :]
        attention_mask[i, :length] = 1

    return {"embeddings": padded, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Main baseline class
# ---------------------------------------------------------------------------

class TextEmbeddingTransformerBaseline:
    """
    Text-Embedding Transformer baseline.

    Run order:
        1. run() calls load_model() + _load_datasets()
        2. _train() performs supervised training on the train split
        3. _evaluate() runs logit extraction on tuning and held_out splits
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.embedder_config = config.get("embedder", {})
        self.data_config = config["data"]
        self.training_config = config.get("training", {})
        self.output_dir = self.training_config.get(
            "output_dir",
            "./outputs/text_embedding_transformer",
        )

        self.model: Optional[TextEmbeddingTransformer] = None
        self.embedding_matrix: Optional[torch.Tensor] = None
        self.phrase_to_row: Optional[Dict[str, int]] = None
        self.embed_dim: Optional[int] = None
        self.head_type: str = self.model_config.get("head_type", "linear")
        self.pooling: str = self.model_config.get("pooling", "mean")

    # ------------------------------------------------------------------
    # Model / embedding loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load precomputed phrase embeddings and build the transformer model."""
        print("\n" + "=" * 80)
        print("Loading precomputed embeddings and building model...")
        print("=" * 80)

        emb_path = self.embedder_config.get("precomputed_embeddings_path")
        if not emb_path or not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Precomputed embeddings not found at: {emb_path}\n"
                "Run src/pipelines/precompute_token_embeddings.py first "
                "(phrase-level, last-token pooling)."
            )

        print(f"  - Loading embeddings from: {emb_path}")
        cache = torch.load(emb_path, map_location="cpu", weights_only=False)

        if "phrase_to_row" not in cache:
            raise ValueError(
                f"Embeddings at {emb_path} look like the old token-level cache "
                "(missing 'phrase_to_row'). Delete it and re-run "
                "precompute_token_embeddings.py to regenerate with last-token "
                "pooling over phrases."
            )

        self.embedding_matrix = cache["embeddings"]
        self.phrase_to_row = {str(k): int(v) for k, v in cache["phrase_to_row"].items()}
        self.embed_dim = int(cache["dim"])
        print(
            f"  - Embeddings: {tuple(self.embedding_matrix.shape)}, "
            f"dim={self.embed_dim}, unit={cache.get('unit', '?')}, "
            f"pooling={cache.get('pooling', '?')}, "
            f"source={cache.get('model_name', 'unknown')}"
        )

        hidden_dim = int(self.model_config.get("hidden_dim", 512))
        num_layers = int(self.model_config.get("num_layers", 6))
        num_heads = int(self.model_config.get("num_heads", 8))
        ff_dim = int(self.model_config.get("ff_dim", 2048))
        dropout = float(self.model_config.get("dropout", 0.1))
        max_seq_length = int(self.model_config.get("max_seq_length", 512))
        num_labels = int(self.model_config.get("num_labels", 2))
        head_type = self.head_type
        head_hidden_size = int(self.model_config.get("head_hidden_size", 256))
        pooling = self.pooling
        pos_init_std = float(self.model_config.get("pos_init_std", 0.02))

        print(
            f"  - Model: hidden={hidden_dim} / {num_layers}L / {num_heads}H / "
            f"ff={ff_dim} / max_seq={max_seq_length} / pool={pooling} / head={head_type}"
        )

        self.model = TextEmbeddingTransformer(
            embed_dim=self.embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_labels=num_labels,
            head_type=head_type,
            head_hidden_size=head_hidden_size,
            pooling=pooling,
            pos_init_std=pos_init_std,
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  - Total parameters:     {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _wrap_dataset(self, base_dataset) -> "_EmbeddingLookupDataset":
        max_seq_length = int(self.model_config.get("max_seq_length", 512))
        return _EmbeddingLookupDataset(
            base_dataset,
            self.embedding_matrix,
            self.phrase_to_row,
            max_seq_length=max_seq_length,
        )

    def _load_datasets(self) -> Dict[str, "_EmbeddingLookupDataset"]:
        print("\n" + "=" * 80)
        print("Loading datasets...")
        print("=" * 80)
        dataset_name = self.data_config.get("dataset", "MIMIC")
        data_cfg = copy.deepcopy(self.data_config)
        data_cfg["format"] = "tokens"
        data_cfg["task"] = "prediction"

        wrapped = {}
        for split in ("train", "tuning", "held_out"):
            print(f"  - {split}...")
            base = get_dataset(dataset_name, data_cfg, split)
            wrapped[split] = self._wrap_dataset(base)
            print(f"    {len(wrapped[split]):,} samples")
        return wrapped

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _get_criterion(self) -> nn.Module:
        if self.head_type == "logistic":
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.head_type == "logistic":
            return torch.sigmoid(logits).squeeze(1)
        return torch.softmax(logits, dim=1)[:, 1]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_collate(self, max_seq_length: int):
        def _collate(batch):
            return _collate_fn(batch, max_seq_length)
        return _collate

    def _train_epoch_with_fraction_evals(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        *,
        n_train: int,
        fractions: Sequence[float],
        best_val_auroc: float,
        convergence_path: str,
    ) -> Tuple[float, float]:
        from src.training.eval_schedule import next_fraction_targets, save_convergence_point

        self.model.train()
        total_loss, n_batches = 0.0, 0
        examples_seen = 0
        optimizer_steps = 0
        pending = next_fraction_targets(fractions)

        for batch in tqdm(train_loader, desc="Training"):
            emb = batch["embeddings"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = self.model(emb, mask)

            if self.head_type == "logistic":
                loss = criterion(logits.squeeze(1), labels.float())
            else:
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            optimizer_steps += 1
            examples_seen += int(labels.shape[0])
            progress = min(1.0, examples_seen / max(1, n_train))

            while pending and progress + 1e-12 >= pending[0]:
                frac = pending.pop(0)
                print(
                    f"\n  [Matched eval] epoch_fraction={frac:.2f} "
                    f"(examples_seen={examples_seen}/{n_train}, step={optimizer_steps})"
                )
                val_metrics = self._eval_loader(
                    val_loader, device, "validation", save_artifacts=False
                )
                save_convergence_point(
                    convergence_path,
                    source="text_embedding",
                    epoch_fraction=frac,
                    examples_seen=examples_seen,
                    n_train=n_train,
                    step=optimizer_steps,
                    metrics=val_metrics,
                )
                print(
                    f"  - Val AUROC={val_metrics['auroc']:.4f} "
                    f"AUPR={val_metrics['aupr']:.4f}"
                )
                if val_metrics["auroc"] > best_val_auroc:
                    best_val_auroc = val_metrics["auroc"]
                    os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.output_dir, "models", "best_model.pt"),
                    )
                    print(f"  - Saved best model (val AUROC {best_val_auroc:.4f})")
                self.model.train()

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, best_val_auroc

    def _train(self, datasets: Dict[str, "_EmbeddingLookupDataset"], device: torch.device):
        print("\n" + "=" * 80)
        print("Supervised training...")
        print("=" * 80)

        # Report phrase coverage (forces one pass over train for accurate stats)
        train_ds = datasets["train"]
        # Cheap coverage estimate: sample up to 512 patients
        sample_n = min(512, len(train_ds))
        for i in range(sample_n):
            _ = train_ds[i]
        print(
            f"  Phrase embedding coverage (first {sample_n} train samples): "
            f"{100.0 * train_ds.coverage():.2f}% "
            f"({train_ds.n_phrases_hit}/{train_ds.n_phrases_total})"
        )
        if train_ds.n_empty:
            print(f"  Warning: {train_ds.n_empty} samples had zero known phrases")

        max_seq_length = int(self.model_config.get("max_seq_length", 512))
        batch_size = int(self.training_config.get("batch_size", 128))
        num_epochs = int(self.training_config.get("epochs", 1))
        lr = float(self.training_config.get("learning_rate", 1e-4))
        wd = float(self.training_config.get("weight_decay", 0.01))
        num_workers = int(self.training_config.get("dataloader_num_workers", 2))

        from src.training.eval_schedule import get_eval_epoch_fractions
        raw_fractions = self.training_config.get("eval_epoch_fractions", "default")
        use_fraction_schedule = raw_fractions not in (None, False, "steps", "legacy")
        fractions = get_eval_epoch_fractions(self.training_config) if use_fraction_schedule else []
        n_train = len(datasets["train"])
        convergence_path = os.path.join(self.output_dir, "results", "convergence.jsonl")
        if use_fraction_schedule and os.path.exists(convergence_path):
            os.remove(convergence_path)

        collate = self._build_collate(max_seq_length)

        print(
            f"  Building DataLoaders "
            f"(batch_size={batch_size}, num_workers={num_workers}, "
            f"train={n_train:,})...",
            flush=True,
        )
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            datasets["tuning"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
        )
        print(
            f"  DataLoaders ready — {len(train_loader)} train batches / epoch. "
            "First batch can be slow (worker spin-up + CUDA warmup).",
            flush=True,
        )
        if use_fraction_schedule:
            print(f"  Matched eval fractions: {fractions}")
            print(f"  Convergence log → {convergence_path}")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        criterion = self._get_criterion()

        best_val_auroc = 0.0
        best_state = None

        for epoch in range(num_epochs):
            print(f"\n[Classification] Epoch {epoch + 1}/{num_epochs}")
            if use_fraction_schedule:
                avg_loss, best_val_auroc = self._train_epoch_with_fraction_evals(
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    device,
                    n_train=n_train,
                    fractions=fractions,
                    best_val_auroc=best_val_auroc,
                    convergence_path=convergence_path,
                )
                print(f"  Epoch {epoch + 1} train loss: {avg_loss:.4f}")
                best_path = os.path.join(self.output_dir, "models", "best_model.pt")
                if os.path.exists(best_path):
                    best_state = torch.load(best_path, map_location="cpu")
            else:
                self.model.train()
                total_loss, n_batches = 0.0, 0
                for batch in tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{num_epochs}] Train"):
                    emb = batch["embeddings"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    optimizer.zero_grad()
                    logits = self.model(emb, mask)
                    if self.head_type == "logistic":
                        loss = criterion(logits.squeeze(1), labels.float())
                    else:
                        loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                print(f"  Epoch {epoch + 1} train loss: {avg_loss:.4f}")

                val_metrics = self._eval_loader(val_loader, device, "tuning (epoch)")
                if val_metrics["auroc"] > best_val_auroc:
                    best_val_auroc = val_metrics["auroc"]
                    best_state = copy.deepcopy(self.model.state_dict())
                    print(f"  ↑ New best val AUROC: {best_val_auroc:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best model (val AUROC {best_val_auroc:.4f})")
        else:
            # No fractional/epoch checkpoint yet — persist current weights.
            best_state = copy.deepcopy(self.model.state_dict())

        model_dir = os.path.join(self.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, "best_model.pt"))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _eval_loader(
        self,
        loader: DataLoader,
        device: torch.device,
        split_name: str,
        save_artifacts: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval [{split_name}]"):
                emb = batch["embeddings"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = self.model(emb, mask)
                probs = self._logits_to_probs(logits).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        metrics = compute_baseline_metrics(all_labels, all_probs)
        if save_artifacts:
            print_results(metrics, split_name)
        return metrics

    def _evaluate_split(
        self,
        dataset: "_EmbeddingLookupDataset",
        split_name: str,
        device: torch.device,
    ) -> Dict[str, float]:
        max_seq_length = int(self.model_config.get("max_seq_length", 512))
        batch_size = int(self.training_config.get("batch_size", 128))
        num_workers = int(self.training_config.get("dataloader_num_workers", 2))
        collate = self._build_collate(max_seq_length)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
        )

        self.model.eval()
        all_labels, all_probs = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval [{split_name}]"):
                emb = batch["embeddings"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = self.model(emb, mask)
                probs = self._logits_to_probs(logits).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = compute_baseline_metrics(all_labels, all_probs)
        print_results(metrics, split_name)

        plot_all_curves(
            all_labels, all_probs,
            os.path.join(self.output_dir, "plots", split_name),
        )
        save_results(
            {"metrics": metrics, "labels": all_labels.tolist(), "probs": all_probs.tolist()},
            os.path.join(self.output_dir, "results"),
            f"{split_name}_results.json",
        )
        return metrics

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Load, train, evaluate — returns (val_metrics, test_metrics)."""
        setup_output_dir(self.output_dir, overwrite=self.training_config.get("overwrite_output_dir", True))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        self.load_model()
        self.model = self.model.to(device)

        datasets = self._load_datasets()
        self._train(datasets, device)

        val_metrics = self._evaluate_split(datasets["tuning"], "tuning", device)
        test_metrics = self._evaluate_split(datasets["held_out"], "held_out", device)

        summary = {
            "validation": val_metrics,
            "test": test_metrics,
            "eval_epoch_fractions": self.training_config.get("eval_epoch_fractions"),
            "supervised_epochs": int(self.training_config.get("epochs", 1)),
            "pooling": self.pooling,
            "head_type": self.head_type,
            "hidden_dim": int(self.model_config.get("hidden_dim", 512)),
        }
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

        print("\n" + "=" * 80)
        print("Text-Embedding Transformer Baseline Complete!")
        print("=" * 80)

        return val_metrics, test_metrics
