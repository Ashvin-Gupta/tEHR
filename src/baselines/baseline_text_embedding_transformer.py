"""
Baseline: Text-Embedding Transformer

Each MIMIC medical event is translated to a human-readable string and embedded
by a frozen language model (e.g. Qwen3-Embedding-0.6B or Bio_ClinicalBERT).
The per-token embeddings are precomputed once (see
src/pipelines/precompute_token_embeddings.py) and cached to disk.

At training time, for each patient the token IDs are looked up in the
precomputed embedding cache to build a sequence of dense vectors.  A
vanilla transformer encoder (configurable layers / heads) with a learnable
CLS token is trained on top of this sequence with a classification head.

The hidden dimension of the transformer equals the embedder's output dimension
and is set automatically from the precomputed embeddings file.

Supported head types:
  linear   — nn.Linear(D, num_labels), CrossEntropyLoss
  mlp      — two-layer MLP with GELU, CrossEntropyLoss
  logistic — nn.Linear(D, 1), BCEWithLogitsLoss  (binary only)

The baseline follows the same run() → evaluate_split() pattern as other
baselines in this project.
"""
import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.baselines.baseline_utils import setup_output_dir
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
    Transformer encoder operating on precomputed token embeddings.

    Input:  a batch of embedding sequences (B, seq_len, D)
    Process:
        1. Prepend learnable CLS token → (B, seq_len+1, D)
        2. TransformerEncoder (num_layers × num_heads)
        3. Take CLS position → classification head
    Output: logits (B, num_labels) for linear/mlp, or (B, 1) for logistic
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        num_labels: int = 2,
        head_type: str = "logistic",
        head_hidden_size: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.head_type = head_type

        # Learnable CLS token embedding (at position 0)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Learnable positional encoding: position 0 = CLS, 1..max_seq_length = tokens
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length + 1, embed_dim))

        # Projection to allow embed_dim != a multiple of num_heads.
        # If embed_dim is already divisible by num_heads, this is an identity mapping.
        # We always keep the transformer's d_model = embed_dim (no projection needed
        # as long as num_heads divides embed_dim; if not, the user should set
        # num_heads accordingly — we validate here).
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}). "
                "Adjust num_heads in the config."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = _Head(head_type, embed_dim, num_labels, head_hidden_size, dropout)

    def forward(
        self,
        embeddings: torch.Tensor,        # (B, seq_len, D)
        attention_mask: torch.Tensor,    # (B, seq_len)  1=attend, 0=pad
    ) -> torch.Tensor:
        """
        Args:
            embeddings:     Pre-looked-up embedding matrix (B, seq_len, D).
            attention_mask: Mask over token positions (B, seq_len).

        Returns:
            Logits (B, num_labels) or (B, 1) for logistic head.
        """
        batch_size, seq_len, D = embeddings.shape

        # Truncate if needed
        if seq_len > self.max_seq_length:
            embeddings = embeddings[:, : self.max_seq_length, :]
            attention_mask = attention_mask[:, : self.max_seq_length]
            seq_len = self.max_seq_length

        # Add positional encodings to token embeddings (positions 1..seq_len)
        x = embeddings + self.pos_encoding[1: seq_len + 1, :].unsqueeze(0)
        x = self.dropout(x)

        # Prepend CLS token (position 0)
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, seq_len+1, D)

        # Extend attention mask: CLS is always attended
        cls_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, seq_len+1)
        src_key_padding_mask = full_mask == 0  # True = ignore (PyTorch convention)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, seq_len+1, D)

        # Classification using CLS token output
        cls_output = x[:, 0, :]  # (B, D)
        logits = self.classifier(cls_output)
        return logits


# ---------------------------------------------------------------------------
# Dataset with embedding lookup
# ---------------------------------------------------------------------------

class _EmbeddingLookupDataset(Dataset):
    """
    Wraps a MIMIC dataset, returning per-patient embedding sequences.

    For each patient, the token_ids are resolved to precomputed embeddings via
    `token_id_to_row` + the embedding matrix.  Unknown token IDs (not in
    precomputed cache) are silently skipped.
    """

    def __init__(
        self,
        base_dataset,
        embedding_matrix: torch.Tensor,      # (N_vocab, D)
        token_id_to_row: Dict[int, int],
        max_seq_length: int = 512,
    ):
        self.embedding_matrix = embedding_matrix
        self.token_id_to_row = token_id_to_row
        self.max_seq_length = max_seq_length

        self.samples: List[Dict[str, Any]] = []
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        token_ids = s["tokens"]
        label = s["label"]

        rows = [
            self.token_id_to_row[tid]
            for tid in token_ids
            if tid in self.token_id_to_row
        ]

        if not rows:
            # No known tokens — return a zero embedding of length 1 so the
            # collate_fn always receives at least one position.
            D = self.embedding_matrix.shape[1]
            emb = torch.zeros(1, D, dtype=torch.float32)
        else:
            # Truncate before indexing to avoid building very large tensors
            rows = rows[: self.max_seq_length]
            row_tensor = torch.tensor(rows, dtype=torch.long)
            emb = self.embedding_matrix[row_tensor].float()  # (seq_len, D)

        return {"embeddings": emb, "label": label}


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def _collate_fn(
    batch: List[Dict[str, Any]],
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Pad embedding sequences to the same length within a batch."""
    embeddings_list = [item["embeddings"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    D = embeddings_list[0].shape[1]
    lengths = [min(e.shape[0], max_seq_length) for e in embeddings_list]
    max_len = max(lengths)

    padded = torch.zeros(len(batch), max_len, D, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (emb, length) in enumerate(zip(embeddings_list, lengths)):
        padded[i, :length, :] = emb[:length, :]
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
        self.token_id_to_row: Optional[Dict[int, int]] = None
        self.embed_dim: Optional[int] = None
        self.head_type: str = self.model_config.get("head_type", "logistic")
        self._label_conversion_warned = False

    # ------------------------------------------------------------------
    # Model / embedding loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load precomputed embeddings and build the transformer model."""
        print("\n" + "=" * 80)
        print("Loading precomputed embeddings and building model...")
        print("=" * 80)

        emb_path = self.embedder_config.get("precomputed_embeddings_path")
        if not emb_path or not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Precomputed embeddings not found at: {emb_path}\n"
                "Run src/pipelines/precompute_token_embeddings.py first."
            )

        print(f"  - Loading embeddings from: {emb_path}")
        cache = torch.load(emb_path, map_location="cpu")
        self.embedding_matrix = cache["embeddings"]    # (N_vocab, D)
        self.token_id_to_row = cache["token_id_to_row"]
        self.embed_dim = int(cache["dim"])
        print(
            f"  - Embeddings: {tuple(self.embedding_matrix.shape)}, "
            f"dim={self.embed_dim}, "
            f"source={cache.get('model_name', 'unknown')}"
        )

        num_layers = int(self.model_config.get("num_layers", 6))
        num_heads = int(self.model_config.get("num_heads", 8))
        ff_dim = int(self.model_config.get("ff_dim", 2048))
        dropout = float(self.model_config.get("dropout", 0.1))
        max_seq_length = int(self.model_config.get("max_seq_length", 512))
        num_labels = int(self.model_config.get("num_labels", 2))
        head_type = self.head_type
        head_hidden_size = int(self.model_config.get("head_hidden_size", 256))

        print(
            f"  - Model: {num_layers}L / {num_heads}H / ff={ff_dim} / "
            f"max_seq={max_seq_length} / head={head_type}"
        )

        self.model = TextEmbeddingTransformer(
            embed_dim=self.embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_labels=num_labels,
            head_type=head_type,
            head_hidden_size=head_hidden_size,
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
            self.token_id_to_row,
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

    def _train(self, datasets: Dict[str, "_EmbeddingLookupDataset"], device: torch.device):
        print("\n" + "=" * 80)
        print("Supervised training...")
        print("=" * 80)

        max_seq_length = int(self.model_config.get("max_seq_length", 512))
        batch_size = int(self.training_config.get("batch_size", 512))
        num_epochs = int(self.training_config.get("epochs", 10))
        lr = float(self.training_config.get("learning_rate", 1e-4))
        wd = float(self.training_config.get("weight_decay", 0.01))
        num_workers = int(self.training_config.get("dataloader_num_workers", 2))

        collate = self._build_collate(max_seq_length)

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

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        criterion = self._get_criterion()

        best_val_auroc = 0.0
        best_state = None

        for epoch in range(num_epochs):
            # ---- Train ----
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

            # ---- Validate ----
            val_metrics = self._eval_loader(val_loader, device, "tuning (epoch)")
            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                best_state = copy.deepcopy(self.model.state_dict())
                print(f"  ↑ New best val AUROC: {best_val_auroc:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best model (val AUROC {best_val_auroc:.4f})")

        # Persist
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
        print_results(metrics, split_name)
        return metrics

    def _evaluate_split(
        self,
        dataset: "_EmbeddingLookupDataset",
        split_name: str,
        device: torch.device,
    ) -> Dict[str, float]:
        max_seq_length = int(self.model_config.get("max_seq_length", 512))
        batch_size = int(self.training_config.get("batch_size", 512))
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

        summary = {"validation": val_metrics, "test": test_metrics}
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

        print("\n" + "=" * 80)
        print("Text-Embedding Transformer Baseline Complete!")
        print("=" * 80)

        return val_metrics, test_metrics
