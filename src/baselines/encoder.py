"""
Experiments: encoder, encoder_supervised, encoder_frozen.

Transformer encoder on filtered EHR tokens (same token filter as xgboost_bow).
"""
import os
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from collections import Counter

from src.baselines.base import setup_output_dir
from src.evaluations.evaluations import compute_binary_metrics, save_results, write_split_artifacts
from src.data.token_filter import should_include_token
from src.data.dataset_factory import get_dataset, is_aces_dataset


def filter_tokens(token_list: List[str]) -> List[str]:
    """
    Filter out tokens using the same logic as XGBoost BOW baseline.
    Uses should_include_token function to filter tokens.
    """
    filtered = []
    for token in token_list:
        token_str = str(token)
        if should_include_token(token_str):
            filtered.append(token_str)
    return filtered


class FilteredTokenDataset(Dataset):
    """
    Dataset wrapper that filters tokens using the same logic as XGBoost BOW baseline.
    
    Uses should_include_token function to filter out:
    - Time intervals
    - Numeric values
    - Units
    - Lab value categories
    - Quantile values
    - Cancer-related tokens
    
    Keeps only:
    - Medical events (MEDICAL//...)
    - Lab test names (LAB//...) - without values
    - Measurement codes (MEASUREMENT//...)
    - Demographics (AGE, GENDER//..., ETHNICITY//..., REGION//...)
    - Lifestyle (LIFESTYLE//...)
    - Special tokens (<start>, <end>, <unknown>, MEDS_BIRTH)
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.filtered_samples = []
        
        # --- 1. PRE-CALCULATE BLACKLIST ---
        # Get the mapping from the underlying dataset
        id_to_token = getattr(base_dataset, "id_to_token_map", None) or getattr(
            base_dataset, "id_to_token", {}
        )
        exclude_ids = set()
        
        print("Identifying tokens to exclude using XGBoost BOW filtering logic...")
        print("  (Excluding: time, units, numbers, lab values, quantiles, cancer-related tokens)")
        for tid, token_str in id_to_token.items():
            token_str = str(token_str)
            
            # Use the same filtering logic as XGBoost BOW
            # Exclude tokens that don't pass should_include_token check
            # Also exclude cancer-related tokens explicitly
            if not should_include_token(token_str) or 'cancer' in token_str.lower():
                exclude_ids.add(tid)
        
        # --- 2. FILTER SEQUENCES USING BLACKLIST ---
        print(f"Filtering {len(base_dataset)} samples...")
        samples_without_tokens = 0
        samples_filtered_empty = 0
        
        for i in tqdm(range(len(base_dataset)), desc="Filtering"):
            sample = base_dataset[i]
            
            # If sample is None, the dataset couldn't find a label for this patient
            if sample is None:
                continue
                
            if 'tokens' not in sample:
                samples_without_tokens += 1
                if samples_without_tokens <= 5:  # Print first few for debugging
                    print(f"  Warning: Sample {i} doesn't have 'tokens' key. Keys: {sample.keys()}")
                continue
            
            token_ids = sample['tokens']
            
            # Handle both list and tensor formats
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            elif not isinstance(token_ids, list):
                token_ids = list(token_ids)
            
            # Fast filtering using the ID blacklist
            # This keeps only IDs NOT in our exclude set
            filtered_ids = [tid for tid in token_ids if tid not in exclude_ids]
            
            # Re-assign the filtered IDs
            sample['tokens'] = filtered_ids
            
            # Only keep the sample if it still has events left
            if len(filtered_ids) > 0:
                self.filtered_samples.append(sample)
            else:
                samples_filtered_empty += 1
        
        print(f"  - Excluded {len(exclude_ids):,} token IDs from vocabulary")
        print(f"  - Kept {len(self.filtered_samples):,} samples after filtering")
        if samples_without_tokens > 0:
            print(f"  - Warning: {samples_without_tokens} samples didn't have 'tokens' key")
        if samples_filtered_empty > 0:
            print(f"  - Warning: {samples_filtered_empty} samples were filtered to empty sequences")
        
        if len(self.filtered_samples) == 0:
            print("ERROR: All samples were filtered out or skipped!")
            print("  This could mean:")
            print("    1. All tokens were excluded by the filter")
            print("    2. Samples don't have 'tokens' key (check dataset format)")
            print("    3. Dataset is empty or has no valid labels")
    
    def __len__(self):
        return len(self.filtered_samples)
    
    def __getitem__(self, idx):
        return self.filtered_samples[idx]


def build_head(
    head_type: str,
    hidden_dim: int,
    num_labels: int,
    head_hidden_size: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Build a classification head.

    Args:
        head_type: 'linear', 'mlp', or 'logistic'.
            - 'linear': nn.Linear(hidden_dim, num_labels) with CrossEntropyLoss.
            - 'mlp': two-layer MLP with GELU and CrossEntropyLoss.
            - 'logistic': nn.Linear(hidden_dim, 1) with BCEWithLogitsLoss (binary only).
        hidden_dim: Input feature size.
        num_labels: Number of output classes (ignored for 'logistic').
        head_hidden_size: Intermediate size for MLP head.
        dropout: Dropout rate for MLP head.
    """
    if head_type == 'linear':
        return nn.Linear(hidden_dim, num_labels)
    elif head_type == 'mlp':
        return nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, num_labels),
        )
    elif head_type == 'logistic':
        return nn.Linear(hidden_dim, 1)
    else:
        raise ValueError(
            f"Unknown head_type: '{head_type}'. Options: 'linear', 'mlp', 'logistic'"
        )


class EHRTransformerEncoder(nn.Module):
    """
    Transformer encoder for EHR token sequences.

    Supports:
    - pooling: 'mean' (masked mean over all token positions) or 'cls' (learnable
      CLS token prepended; position-0 output used for classification).
    - head_type: 'linear', 'mlp', or 'logistic' (see build_head).
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        num_labels: int = 2,
        head_type: str = 'linear',
        head_hidden_size: int = 256,
        pooling: str = 'mean',
    ):
        """
        Initialize transformer encoder.
        
        Args:
            vocab_size: Size of token vocabulary
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length (for token positions; CLS is extra)
            num_labels: Number of output classes
            head_type: 'linear', 'mlp', or 'logistic'
            head_hidden_size: Intermediate size for MLP head
            pooling: 'mean' or 'cls'
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.pooling = pooling
        self.head_type = head_type
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding. Allocate max_seq_length+1 so that CLS pooling
        # can assign position 0 to the CLS token and 1..seq_len to real tokens.
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_length + 1, hidden_dim))
        
        # Learnable CLS token (used only when pooling='cls')
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = build_head(head_type, hidden_dim, num_labels, head_hidden_size, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode token IDs into contextual representations (no pooling, no CLS, no head).
        Used by EHRTransformerMLMHead for per-token MLM predictions.
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Sequence output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = token_ids.shape
        
        # Truncate if sequence exceeds max_seq_length — keep most recent events
        if seq_len > self.max_seq_length:
            token_ids = token_ids[:, -self.max_seq_length:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_seq_length:]
            seq_len = self.max_seq_length
        
        # Embeddings + positional encoding (positions 0..seq_len-1)
        x = self.token_embedding(token_ids)  # [batch, seq_len, hidden_dim]
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.dropout(x)
        
        # Convert to transformer mask format: True = ignore, False = attend
        if attention_mask is not None:
            mask = (attention_mask == 0)
        else:
            mask = None
        
        x = self.transformer(x, src_key_padding_mask=mask)  # [batch, seq_len, hidden_dim]
        return x
    
    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for classification.
        
        When pooling='cls': prepends a learnable CLS token at position 0, assigns
        positional encodings 1..seq_len to the real tokens, and takes position-0
        output as the sequence representation.

        When pooling='mean': masked mean over all real token positions (original behaviour).
        
        Args:
            token_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Logits [batch_size, num_labels] for linear/mlp, or [batch_size, 1] for logistic
        """
        batch_size, seq_len = token_ids.shape

        # Truncate — keep most recent events
        if seq_len > self.max_seq_length:
            token_ids = token_ids[:, -self.max_seq_length:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_seq_length:]
            seq_len = self.max_seq_length

        if self.pooling == 'cls':
            # Tokens get positions 1..seq_len; CLS occupies position 0.
            x = self.token_embedding(token_ids)
            x = x + self.pos_encoding[1:seq_len + 1, :].unsqueeze(0)
            x = self.dropout(x)

            # Prepend CLS token
            cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
            x = torch.cat([cls, x], dim=1)  # [B, seq_len+1, D]

            # Extend attention mask to cover CLS position (always attended)
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                full_mask = torch.cat([cls_mask, attention_mask], dim=1)
                src_key_padding_mask = (full_mask == 0)
            else:
                src_key_padding_mask = None

            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            pooled = x[:, 0, :]  # CLS token output

        else:  # pooling == 'mean'
            x = self.encode(token_ids, attention_mask)
            if attention_mask is not None:
                mask_expanded = attention_mask.float().unsqueeze(-1)
                x = x * mask_expanded
                pooled = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


class EHRTransformerMLMHead(nn.Module):
    """
    MLM head that wraps an encoder and predicts token IDs.
    """
    
    def __init__(self, encoder: EHRTransformerEncoder, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(encoder.hidden_dim, vocab_size)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] with -100 at non-MLM positions
        
        Returns:
            loss (optional), logits [batch, seq_len, vocab_size]
        """
        sequence_output = self.encoder.encode(token_ids, attention_mask)
        logits = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
            )
        return loss, logits


class EncoderBaseline:
    """
    Encoder transformer baseline for EHR classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the encoder transformer baseline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.data_config = config['data']
        self.training_config = config.get('training', {})
        self.output_dir = self.training_config.get('output_dir', './outputs/encoder_transformer')
        
        self.model = None
        self.mlm_model = None
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None
        # max_seq_length will be set when model is created
        self.max_seq_length = None
        self._label_conversion_warned = False  # Track if we've warned about label conversion
        self.pretrain_config = config.get('pretrain', {})  # Optional stage-1 config
        self._rank = 0
        self._local_rank = 0
        self._world_size = 1

    def _setup_distributed(self) -> torch.device:
        """Init DDP when launched via torchrun; otherwise use a single device."""
        self._world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._rank = int(os.environ.get("RANK", self._local_rank))
        if self._world_size <= 1 or not torch.cuda.is_available():
            self._world_size = 1
            self._rank = 0
            self._local_rank = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"\nUsing device: {device}")
            return device
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self._local_rank)
        device = torch.device("cuda", self._local_rank)
        if self._is_rank0():
            print(f"\nUsing {self._world_size} GPUs (DDP), this rank device: {device}")
        return device

    def _is_rank0(self) -> bool:
        return self._rank == 0

    def _barrier(self) -> None:
        if self._world_size > 1 and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model

    def _wrap_ddp(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        find_unused_parameters: bool = False,
    ) -> nn.Module:
        model = model.to(device)
        if self._world_size <= 1:
            return model
        return DDP(
            model,
            device_ids=[self._local_rank],
            output_device=self._local_rank,
            find_unused_parameters=find_unused_parameters,
        )

    def _make_loader(
        self,
        dataset,
        batch_size: int,
        *,
        shuffle: bool,
        collate_fn,
        num_workers: int,
        drop_last: bool = False,
    ) -> DataLoader:
        sampler = None
        if self._world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )

    def _set_sampler_epoch(self, loader: DataLoader, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

    def _gather_list(self, values: list) -> list:
        if self._world_size <= 1:
            return values
        gathered: list = [None] * self._world_size
        dist.all_gather_object(gathered, values)
        out: list = []
        for part in gathered:
            out.extend(part)
        return out

    def _reduce_mean(self, total: float, count: int, device: torch.device) -> float:
        if self._world_size <= 1:
            return total / max(count, 1)
        t = torch.tensor([total, float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t[0] / t[1].clamp(min=1.0))

    def _save_classifier_state(self, path: str) -> None:
        if not self._is_rank0():
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._unwrap_model(self.model).state_dict(), path)

    def _backbone_label(self) -> str:
        return "encoder"

    def _pretrain_label(self) -> str:
        return "MLM"

    def _experiment_label(self) -> str:
        return "encoder"

    def _stage1_ckpt_basename(self) -> str:
        return "pretrained_encoder_stage1_mlm.pt"

    def _stage1_weight_key(self) -> str:
        return "encoder_state_dict"

    def _load_datasets(self, splits: List[str], task_override: str = None) -> Dict[str, Dataset]:
        """Load datasets through the factory, optionally overriding MIMIC task."""
        dataset_name = self.data_config.get("dataset", "CPRD")
        data_cfg = copy.deepcopy(self.data_config)
        data_cfg["format"] = "tokens"
        if task_override is not None:
            data_cfg["task"] = task_override
        return {split: get_dataset(dataset_name, data_cfg, split) for split in splits}

    def _load_pretrained_checkpoint(self, checkpoint_path: str, device: torch.device):
        """
        Load an encoder checkpoint and restore vocab + weights.

        Accepts either:
          - stage-1 MLM bundle: ``{encoder_state_dict, vocab, ...}``
          - raw ``state_dict`` (e.g. ``best_model.pt`` / ``pytorch_model.bin``)
        """
        print(f"\nLoading pretrained {self._backbone_label()} checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and (
            "encoder_state_dict" in checkpoint
            or "decoder_state_dict" in checkpoint
            or "model_state_dict" in checkpoint
            or "vocab" in checkpoint
        ):
            ckpt_vocab = checkpoint.get("vocab")
            state_dict = (
                checkpoint.get(self._stage1_weight_key())
                or checkpoint.get("encoder_state_dict")
                or checkpoint.get("decoder_state_dict")
                or checkpoint.get("model_state_dict")
            )
        else:
            ckpt_vocab = None
            state_dict = checkpoint

        if ckpt_vocab is not None:
            self.vocab = ckpt_vocab
            self.token_to_id = ckpt_vocab.get("token_to_id", {})
            self.id_to_token = ckpt_vocab.get("id_to_token", {})
            print(f"  - Restored checkpoint vocabulary (size: {self.vocab.get('size', 'unknown')})")
        elif self.vocab is None:
            raise ValueError(
                "Checkpoint missing `vocab`, and no in-memory vocab available. "
                "Use the stage-1 MLM checkpoint (has vocab), or build vocab first."
            )

        if self.model is None:
            self.create_model()
        self.model = self.model.to(device)

        if state_dict is None:
            raise ValueError(
                "Checkpoint missing weights "
                "(`encoder_state_dict` / `decoder_state_dict` / "
                "`model_state_dict` / raw state_dict)."
            )
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"  - Missing keys (ok if reiniting head): {incompatible.missing_keys[:8]}")
        if incompatible.unexpected_keys:
            print(f"  - Unexpected keys ignored: {incompatible.unexpected_keys[:8]}")
        print(f"  - Loaded {self._backbone_label()} weights successfully")

    def _freeze_encoder_train_head_only(self, reinit_classifier: bool = True):
        """
        Freeze token embeddings + transformer; train only the classification head.
        """
        if self.model is None:
            raise RuntimeError("Model must be created/loaded before freezing.")

        if reinit_classifier:
            hidden_dim = int(self.model_config.get("hidden_dim", 512))
            num_labels = int(self.model_config.get("num_labels", 2))
            head_type = self.model_config.get("head_type", "linear")
            head_hidden_size = int(self.model_config.get("head_hidden_size", 256))
            dropout = float(self.model_config.get("dropout", 0.1))
            device = next(self._unwrap_model(self.model).parameters()).device
            self._unwrap_model(self.model).classifier = build_head(
                head_type, hidden_dim, num_labels, head_hidden_size, dropout
            ).to(device)
            print("  - Re-initialized classification head")

        n_frozen = 0
        n_train = 0
        for name, param in self._unwrap_model(self.model).named_parameters():
            train = "classifier" in name
            param.requires_grad = train
            if train:
                n_train += param.numel()
            else:
                n_frozen += param.numel()

        print(
            f"  - Freeze {self._backbone_label()}: trainable={n_train:,} params "
            f"(head only), frozen={n_frozen:,}"
        )    
    def build_vocabulary(self, datasets: Dict[str, Dataset]):
        """
        Build vocabulary from all datasets.
        
        Args:
            datasets: Dictionary of datasets
        """
        print("\n" + "=" * 80)
        print("Building vocabulary from datasets...")
        print("=" * 80)
        
        # Count token frequencies across all datasets
        token_counter: Counter = Counter()
        
        for split_name, dataset in datasets.items():
            print(f"  - Processing {split_name} split...")
            for i in tqdm(range(len(dataset)), desc=f"Collecting tokens from {split_name}"):
                sample = dataset[i]
                if sample is not None and 'tokens' in sample:
                    tokens = sample['tokens']
                    for token in tokens:
                        if isinstance(token, (int, str)):
                            token_str = str(token)
                            token_counter[token_str] += 1
        
        # Report total unique tokens before any cap
        num_unique_tokens = len(token_counter)
        print(f"  - Total unique tokens observed: {num_unique_tokens:,}")
        
        # Configurable vocab cap: vocab_size=null (None) means keep full vocab
        max_vocab_tokens = self.model_config.get('vocab_size', None)
        if max_vocab_tokens is not None:
            most_common_tokens = token_counter.most_common(int(max_vocab_tokens))
            kept_tokens = [tok for tok, _ in most_common_tokens]
            print(f"  - Keeping top {len(kept_tokens):,} most frequent tokens (vocab_size={max_vocab_tokens})")
        else:
            kept_tokens = list(token_counter.keys())
            print(f"  - Using full vocabulary ({len(kept_tokens):,} tokens, no cap)")
        
        # Create vocabulary from kept tokens only
        kept_tokens = sorted(kept_tokens)
        # Reserve special tokens: PAD, UNK, MASK
        vocab_size = len(kept_tokens) + 3
        
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<MASK>': 2}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>', 2: '<MASK>'}
        
        for i, token in enumerate(kept_tokens, start=3):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.vocab = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'size': vocab_size
        }
        
        print(f"  - Final vocabulary size (including PAD/UNK): {vocab_size:,}")
    
    def tokenize_sequence(self, tokens: List) -> torch.Tensor:
        """
        Convert token sequence to IDs.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Tensor of token IDs
        """
        token_ids = []
        for token in tokens:
            token_str = str(token)
            # Only keep tokens that are in the top-K vocabulary.
            # Tokens not in vocab are removed from the stream.
            if token_str in self.token_to_id:
                token_ids.append(self.token_to_id[token_str])
        
        # If all tokens were filtered out for this sequence, return a single PAD token
        # so that downstream code always sees at least one token.
        if len(token_ids) == 0:
            token_ids = [self.token_to_id['<PAD>']]
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def mlm_collate_fn(self, batch):
        """
        Collate function for MLM pretraining.
        
        Returns input_ids, attention_mask, and MLM labels.
        """
        base_batch = self.collate_fn(batch)
        input_ids = base_batch['token_ids']
        attention_mask = base_batch['attention_mask']
        
        labels = input_ids.clone()
        
        # Do not predict on padding tokens
        pad_id = self.token_to_id['<PAD>']
        special_mask = input_ids.eq(pad_id)
        
        mlm_probability = float(self.pretrain_config.get('mlm_probability', 0.15))
        probability_matrix = torch.full(labels.shape, mlm_probability)
        probability_matrix.masked_fill_(special_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100
        
        mask_token_id = self.token_to_id.get('<MASK>', self.token_to_id['<UNK>'])
        
        # 80% of selected tokens → [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id
        
        # 10% → random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.token_to_id), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% → keep original token (already handled by masked_indices but not in indices_replaced/indices_random)
        
        return {
            'token_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of samples
        
        Returns:
            Batched tensors
        """
        # Get max_seq_length (default to 512 if not set)
        max_seq_length = self.max_seq_length if self.max_seq_length is not None else 512
        
        token_ids_list = []
        labels_list = []
        max_len = 0
        
        for sample in batch:
            if sample is not None:
                tokens = sample.get('tokens', [])
                token_ids = self.tokenize_sequence(tokens)
                
                # Keep the most recent events (tail), matching the LLM path
                if len(token_ids) > max_seq_length:
                    token_ids = token_ids[-max_seq_length:]
                
                token_ids_list.append(token_ids)
                max_len = max(max_len, len(token_ids))
                
                label = sample['label']
                if hasattr(label, 'item'):
                    label = label.item()
                
                # Convert multi-class labels to binary: 0 = Control, >0 = Cancer (1)
                # The dataset returns multi-class labels (0-19), but model expects binary (0 or 1)
                binary_label = 0 if label == 0 else 1
                
                # Debug: print first conversion example
                if not self._label_conversion_warned and len(labels_list) == 0:
                    print(f"  Converting labels: multi-class label {label} -> binary label {binary_label}")
                    self._label_conversion_warned = True
                
                labels_list.append(binary_label)
        
        # Ensure max_len doesn't exceed max_seq_length
        max_len = min(max_len, max_seq_length)
        
        # Pad sequences
        batch_size = len(token_ids_list)
        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, token_ids in enumerate(token_ids_list):
            seq_len = min(len(token_ids), max_len)
            padded_ids[i, :seq_len] = token_ids[:seq_len]
            attention_mask[i, :seq_len] = 1
        
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Verify binary labels
        unique_labels = torch.unique(labels)
        if len(unique_labels) > 2 or not all(l.item() in [0, 1] for l in unique_labels):
            print(f"Warning: Expected binary labels [0, 1], got {unique_labels.tolist()}")
        
        return {
            'token_ids': padded_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def create_model(self):
        """Create transformer encoder model."""
        print("\n" + "=" * 80)
        print("Creating transformer encoder model...")
        print("=" * 80)
        
        vocab_size = int(self.vocab['size'])
        hidden_dim = int(self.model_config.get('hidden_dim', 512))
        num_layers = int(self.model_config.get('num_layers', 6))
        num_heads = int(self.model_config.get('num_heads', 8))
        ff_dim = int(self.model_config.get('ff_dim', 2048))
        dropout = float(self.model_config.get('dropout', 0.1))
        max_seq_length = int(self.model_config.get('max_seq_length', 512))
        num_labels = int(self.model_config.get('num_labels', 2))
        head_type = self.model_config.get('head_type', 'linear')
        head_hidden_size = int(self.model_config.get('head_hidden_size', 256))
        pooling = self.model_config.get('pooling', 'mean')
        
        # Store for use in collate_fn, train_epoch, evaluate
        self.max_seq_length = max_seq_length
        self.head_type = head_type
        
        print(f"  - head_type: {head_type}, pooling: {pooling}")
        
        self.model = EHRTransformerEncoder(
            vocab_size=vocab_size,
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
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def create_mlm_model(self):
        """Create MLM head that wraps the encoder."""
        if self.model is None:
            self.create_model()
        
        vocab_size = int(self.vocab['size'])
        self.mlm_model = EHRTransformerMLMHead(self.model, vocab_size=vocab_size)
    
    def train_mlm_epoch(self, train_loader, optimizer, device):
        """Train MLM for one epoch."""
        self.mlm_model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(
            train_loader,
            desc=f"{self._pretrain_label()} Training",
            disable=not self._is_rank0(),
        ):
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            loss, _ = self.mlm_model(token_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return self._reduce_mean(total_loss, num_batches, device)
    
    def evaluate_mlm(self, data_loader, device):
        """Compute average MLM loss on validation set."""
        model = self._unwrap_model(self.mlm_model)
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                desc=f"{self._pretrain_label()} Eval",
                disable=not self._is_rank0(),
            ):
                token_ids = batch['token_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss, _ = model(token_ids, attention_mask, labels)
                total_loss += loss.item()
                num_batches += 1
        
        return self._reduce_mean(total_loss, num_batches, device)
    
    def _get_criterion(self):
        """Return the appropriate loss function for the configured head type."""
        head_type = getattr(self, 'head_type', 'linear')
        if head_type == 'logistic':
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to positive-class probabilities."""
        head_type = getattr(self, 'head_type', 'linear')
        if head_type == 'logistic':
            return torch.sigmoid(logits).squeeze(1)
        return torch.softmax(logits, dim=1)[:, 1]

    def train_epoch(self, train_loader, optimizer, criterion, device):
        """Train for one epoch (no mid-epoch evaluation)."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        head_type = getattr(self, 'head_type', 'linear')
        
        for batch in tqdm(train_loader, desc="Training", disable=not self._is_rank0()):
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = self.model(token_ids, attention_mask)

            if head_type == 'logistic':
                loss = criterion(logits.squeeze(1), labels.float())
            else:
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return self._reduce_mean(total_loss, num_batches, device)

    def train_epoch_with_fraction_evals(
        self,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        *,
        n_train: int,
        fractions: Sequence[float],
        best_val_auroc: float,
        convergence_path: str,
        source: str = "encoder",
    ):
        """
        One supervised epoch with validation at matched epoch fractions
        (examples_seen / n_train), mirroring the LLM classifier schedule.
        """
        from src.training.eval_schedule import next_fraction_targets, save_convergence_point

        self.model.train()
        total_loss = 0.0
        num_batches = 0
        examples_seen = 0
        optimizer_steps = 0
        head_type = getattr(self, "head_type", "linear")
        pending = next_fraction_targets(fractions)

        for batch in tqdm(train_loader, desc="Training", disable=not self._is_rank0()):
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = self.model(token_ids, attention_mask)

            if head_type == "logistic":
                loss = criterion(logits.squeeze(1), labels.float())
            else:
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            optimizer_steps += 1
            examples_seen += int(labels.shape[0]) * self._world_size
            progress = min(1.0, examples_seen / max(1, n_train))

            while pending and progress + 1e-12 >= pending[0]:
                frac = pending.pop(0)
                if self._is_rank0():
                    print(
                        f"\n  [Matched eval] epoch_fraction={frac:.2f} "
                        f"(examples_seen={examples_seen}/{n_train}, step={optimizer_steps})"
                    )
                val_metrics = self.evaluate(
                    val_loader, device, "validation", save_artifacts=False
                )
                if self._is_rank0():
                    save_convergence_point(
                        convergence_path,
                        source=source,
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
                    self._save_classifier_state(
                        os.path.join(self.output_dir, "models", "best_model.pt"),
                    )
                    if self._is_rank0():
                        print(f"  - Saved best model (val AUROC {best_val_auroc:.4f})")
                self.model.train()

        # Ensure fraction 1.0 fired even if rounding left it pending.
        if pending:
            frac = pending[-1]
            if self._is_rank0():
                print(
                    f"\n  [Matched eval] epoch_fraction={frac:.2f} "
                    f"(examples_seen={examples_seen}/{n_train}, step={optimizer_steps})"
                )
            val_metrics = self.evaluate(
                val_loader, device, "validation", save_artifacts=False
            )
            if self._is_rank0():
                save_convergence_point(
                    convergence_path,
                    source=source,
                    epoch_fraction=frac,
                    examples_seen=examples_seen,
                    n_train=n_train,
                    step=optimizer_steps,
                    metrics=val_metrics,
                )
            if val_metrics["auroc"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc"]
                self._save_classifier_state(
                    os.path.join(self.output_dir, "models", "best_model.pt"),
                )
                if self._is_rank0():
                    print(f"  - Saved best model (val AUROC {best_val_auroc:.4f})")

        avg_loss = self._reduce_mean(total_loss, num_batches, device)
        return avg_loss, best_val_auroc
    
    def evaluate(
        self,
        data_loader,
        device,
        split_name: str = "validation",
        save_artifacts: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader instance
            device: Device to run on
            split_name: Name of the split
            save_artifacts: If False, skip plots/JSON (used for mid-epoch matched evals)
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        eval_model = self._unwrap_model(self.model)
        eval_model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                desc=f"Evaluating {split_name}",
                disable=not self._is_rank0(),
            ):
                token_ids = batch['token_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = eval_model(token_ids, attention_mask)
                probs = self._logits_to_probs(logits).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_labels.extend(labels_np.tolist())
                all_probs.extend(np.asarray(probs).reshape(-1).tolist())
        
        all_labels = np.array(self._gather_list(all_labels))
        all_probs = np.array(self._gather_list(all_probs))
        if save_artifacts:
            if self._is_rank0():
                return write_split_artifacts(self.output_dir, split_name, all_labels, all_probs)
            return compute_binary_metrics(all_labels, all_probs)
        return compute_binary_metrics(all_labels, all_probs)
    
    def run(self, resume_from_checkpoint: str | None = None):
        """Run the complete baseline training and evaluation pipeline."""
        # Setup
        setup_output_dir(self.output_dir, overwrite=self.training_config.get('overwrite_output_dir', False))
        device = self._setup_distributed()
        
        # Load datasets (for MIMIC stage-2 we force prediction/cohort mode)
        dataset_name = self.data_config.get("dataset", "CPRD")
        stage2_task = "prediction" if is_aces_dataset(dataset_name) else None
        datasets = self._load_datasets(['train', 'tuning', 'held_out'], task_override=stage2_task)
        
        # Filter tokens
        filtered_datasets = {}
        for split_name, dataset in datasets.items():
            filtered_datasets[split_name] = FilteredTokenDataset(dataset)
        
        checkpoint_path = self.pretrain_config.get("pretrained_checkpoint")
        force_pretrain = bool(self.pretrain_config.get("force_pretrain", False))

        if checkpoint_path and not os.path.exists(checkpoint_path):
            print(
                f"  - pretrain.pretrained_checkpoint not found: {checkpoint_path}\n"
                "  - Will run stage-1 pretraining and save a new checkpoint."
            )
            checkpoint_path = None

        if checkpoint_path:
            # Reuse a previously pretrained stage-1 backbone and skip pretrain by default.
            self._load_pretrained_checkpoint(checkpoint_path, device)
            if self.pretrain_config.get("enabled", False) and not force_pretrain:
                print(
                    f"  - pretrain.enabled=True but pretrained_checkpoint provided; "
                    f"skipping stage-1 {self._pretrain_label()}."
                )
        else:
            # Build vocabulary and create model from scratch when no checkpoint is provided.
            self.build_vocabulary(filtered_datasets)
            self.create_model()
            self.model = self.model.to(device)
        
        # Optional Stage 1: token-level pretraining on the backbone
        if self.pretrain_config.get('enabled', False) and (not checkpoint_path or force_pretrain):
            print("\n" + "=" * 80)
            print(
                f"Stage 1: {self._pretrain_label()} pretraining for "
                f"{self._backbone_label()} transformer"
            )
            print("=" * 80)
            
            # Freeze classification head during token-level pretraining
            for p in self.model.classifier.parameters():
                p.requires_grad = False
            
            self.create_mlm_model()
            self.mlm_model = self._wrap_ddp(
                self.mlm_model, device, find_unused_parameters=True
            )
            
            mlm_batch_size = int(self.pretrain_config.get('batch_size', self.training_config.get('batch_size', 32)))
            mlm_workers = int(self.pretrain_config.get(
                'dataloader_num_workers',
                self.training_config.get('dataloader_num_workers', 4),
            ))
            train_drop = self._world_size > 1

            # Stage-1 must be unlabeled full timelines, not the labeled task cohort.
            # EHRSHOT: train patients only (tuning for MLM val). MIMIC keeps the
            # existing full-corpus option via use_all_splits_for_mimic.
            stage1_train_ds = filtered_datasets['train']
            stage1_val_ds = filtered_datasets['tuning']
            if is_aces_dataset(dataset_name):
                if str(dataset_name).upper() == "EHRSHOT":
                    stage1_raw = self._load_datasets(
                        ['train', 'tuning'],
                        task_override="pretrain",
                    )
                    stage1_train_ds = FilteredTokenDataset(stage1_raw['train'])
                    stage1_val_ds = FilteredTokenDataset(stage1_raw['tuning'])
                    print(
                        "  - EHRSHOT stage-1: unlabeled full timelines on "
                        "train patients; tuning for MLM val (no labels, no cutoff)"
                    )
                elif self.pretrain_config.get("use_all_splits_for_mimic", True):
                    stage1_raw = self._load_datasets(
                        ['train', 'tuning', 'held_out'],
                        task_override="pretrain",
                    )
                    stage1_filtered = [
                        FilteredTokenDataset(stage1_raw[s])
                        for s in ['train', 'tuning', 'held_out']
                    ]
                    stage1_train_ds = ConcatDataset(stage1_filtered)
                    stage1_val_ds = stage1_train_ds
                    print(
                        f"  - {dataset_name} stage-1 uses train+tuning+held_out "
                        "in pretrain mode"
                    )

            mlm_train_loader = self._make_loader(
                stage1_train_ds,
                mlm_batch_size,
                shuffle=True,
                collate_fn=self.mlm_collate_fn,
                num_workers=mlm_workers,
                drop_last=train_drop,
            )
            mlm_val_loader = self._make_loader(
                stage1_val_ds,
                mlm_batch_size,
                shuffle=False,
                collate_fn=self.mlm_collate_fn,
                num_workers=mlm_workers,
                drop_last=False,
            )
            
            mlm_optimizer = torch.optim.AdamW(
                self.mlm_model.parameters(),
                lr=float(self.pretrain_config.get('learning_rate', self.training_config.get('learning_rate', 1e-4))),
                weight_decay=float(self.pretrain_config.get('weight_decay', self.training_config.get('weight_decay', 0.01)))
            )
            mlm_epochs = int(self.pretrain_config.get('epochs', 1))
            
            best_mlm_loss = float('inf')
            ehrshot_mlm = str(dataset_name).upper() == "EHRSHOT"
            for epoch in range(mlm_epochs):
                self._set_sampler_epoch(mlm_train_loader, epoch)
                if self._is_rank0():
                    print(f"\n[{self._pretrain_label()}] Epoch {epoch + 1}/{mlm_epochs}")
                train_mlm_loss = self.train_mlm_epoch(mlm_train_loader, mlm_optimizer, device)
                # EHRSHOT: skip most MLM vals (full-seq, ~vocab 50k). Keep last + every 10th.
                run_mlm_val = (not ehrshot_mlm) or (epoch + 1) == mlm_epochs or (epoch + 1) % 10 == 0
                if run_mlm_val:
                    val_mlm_loss = self.evaluate_mlm(mlm_val_loader, device)
                    if self._is_rank0():
                        print(f"  - {self._pretrain_label()} train loss: {train_mlm_loss:.4f}")
                        print(f"  - {self._pretrain_label()} val loss:   {val_mlm_loss:.4f}")
                    best_mlm_loss = min(best_mlm_loss, val_mlm_loss)
                elif self._is_rank0():
                    print(
                        f"  - {self._pretrain_label()} train loss: {train_mlm_loss:.4f} "
                        "(skipped MLM val)"
                    )
            
            if self._is_rank0():
                print(
                    f"\nFinished {self._pretrain_label()} pretraining. "
                    f"Best val loss: {best_mlm_loss:.4f}"
                )

            # Persist stage-1 backbone so it can be reused independently of stage-2.
            self.mlm_model = self._unwrap_model(self.mlm_model)
            if self.pretrain_config.get("save_model", True) and self._is_rank0():
                configured_ckpt = self.pretrain_config.get("pretrained_checkpoint")
                if configured_ckpt:
                    pretrain_model_path = configured_ckpt
                    pretrain_model_dir = os.path.dirname(pretrain_model_path)
                else:
                    pretrain_model_dir = os.path.join(self.output_dir, "models")
                    pretrain_model_path = os.path.join(
                        pretrain_model_dir, self._stage1_ckpt_basename()
                    )
                os.makedirs(pretrain_model_dir, exist_ok=True)
                torch.save(
                    {
                        self._stage1_weight_key(): self._unwrap_model(self.model).state_dict(),
                        "vocab": self.vocab,
                        "model_config": self.model_config,
                        "pretrain_config": self.pretrain_config,
                        "best_pretrain_val_loss": best_mlm_loss,
                    },
                    pretrain_model_path,
                )
                print(
                    f"  - Saved stage-1 pretrained {self._backbone_label()} to: "
                    f"{pretrain_model_path}"
                )
            self._barrier()
            
            # Re-enable classification head for supervised fine-tuning
            for p in self._unwrap_model(self.model).classifier.parameters():
                p.requires_grad = True
        
        # Optional: freeze encoder backbone, train head only (LLM-5b analogue).
        freeze_encoder = bool(self.training_config.get("freeze_encoder", False))
        if freeze_encoder:
            if not checkpoint_path:
                raise RuntimeError(
                    "training.freeze_encoder=True requires "
                    "pretrain.pretrained_checkpoint to load a frozen backbone "
                    "(use the stage-1 pretrained checkpoint)."
                )
            print("\n" + "=" * 80)
            print(f"Freezing {self._backbone_label()}; training classification head only")
            print("=" * 80)
            self._freeze_encoder_train_head_only(
                reinit_classifier=bool(
                    self.training_config.get("reinit_classifier", True)
                )
            )

        self.model = self._wrap_ddp(self.model, device, find_unused_parameters=False)

        # Create data loaders for supervised training
        batch_size = int(self.training_config.get('batch_size', 32))
        workers = int(self.training_config.get('dataloader_num_workers', 4))
        train_drop = self._world_size > 1
        train_loader = self._make_loader(
            filtered_datasets['train'],
            batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=workers,
            drop_last=train_drop,
        )
        val_loader = self._make_loader(
            filtered_datasets['tuning'],
            batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
            drop_last=False,
        )
        test_loader = self._make_loader(
            filtered_datasets['held_out'],
            batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
            drop_last=False,
        )
        
        # Training setup for supervised classification
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters — check freeze_encoder settings.")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(self.training_config.get('learning_rate', 1e-4)),
            weight_decay=float(self.training_config.get('weight_decay', 0.01))
        )
        criterion = self._get_criterion()
        
        # Train
        num_epochs = int(self.training_config.get('epochs', 1))
        from src.training.eval_schedule import get_eval_epoch_fractions
        raw_fractions = self.training_config.get("eval_epoch_fractions", "default")
        use_fraction_schedule = raw_fractions not in (None, False, "steps", "legacy")
        fractions = get_eval_epoch_fractions(self.training_config) if use_fraction_schedule else []
        n_train = len(filtered_datasets["train"])
        convergence_path = os.path.join(self.output_dir, "results", "convergence.jsonl")
        if use_fraction_schedule and self._is_rank0() and os.path.exists(convergence_path):
            os.remove(convergence_path)

        if self._is_rank0():
            print(f"\n" + "=" * 80)
            stage_label = (
                f"Stage 2: Head-only training ({self._backbone_label()} frozen)"
                if freeze_encoder
                else f"Stage 2: Supervised training for {num_epochs} epoch(s)"
            )
            print(f"{stage_label}...")
            if use_fraction_schedule:
                print(f"  Matched eval fractions: {fractions}")
                print(f"  Convergence log → {convergence_path}")
            print("=" * 80)
        
        best_val_auroc = 0.0
        source = (
            f"{self._experiment_label()}_frozen" if freeze_encoder else self._experiment_label()
        )
        
        for epoch in range(num_epochs):
            self._set_sampler_epoch(train_loader, epoch)
            if self._is_rank0():
                print(f"\n[Classification] Epoch {epoch + 1}/{num_epochs}")
            if use_fraction_schedule:
                train_loss, best_val_auroc = self.train_epoch_with_fraction_evals(
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    device,
                    n_train=n_train,
                    fractions=fractions,
                    best_val_auroc=best_val_auroc,
                    convergence_path=convergence_path,
                    source=source,
                )
                if self._is_rank0():
                    print(f"  - Train loss: {train_loss:.4f}")
            else:
                train_loss = self.train_epoch(train_loader, optimizer, criterion, device)
                if self._is_rank0():
                    print(f"  - Train loss: {train_loss:.4f}")
                
                # Validate (legacy epoch-end only)
                val_metrics = self.evaluate(val_loader, device, 'validation')
                if val_metrics['auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['auroc']
                    self._save_classifier_state(
                        os.path.join(self.output_dir, 'models', 'best_model.pt')
                    )
                    if self._is_rank0():
                        print(f"  - Saved best model (val AUROC {best_val_auroc:.4f})")
        
        # Load best model (by validation AUROC) for final evaluation and saving
        best_path = os.path.join(self.output_dir, 'models', 'best_model.pt')
        if self._is_rank0() and not os.path.exists(best_path):
            # No improvement recorded (e.g. empty schedule) — save current weights.
            self._save_classifier_state(best_path)
            print("  - No best checkpoint yet; saved final weights as best_model.pt")
        self._barrier()
        self._unwrap_model(self.model).load_state_dict(
            torch.load(best_path, map_location=device)
        )
        if self._is_rank0():
            print(f"\nLoaded best model (best val AUROC) for final evaluation.")
        
        # Save best model to final_model for consistency with other baselines
        if self._is_rank0():
            final_model_dir = os.path.join(self.output_dir, 'final_model')
            os.makedirs(final_model_dir, exist_ok=True)
            torch.save(
                self._unwrap_model(self.model).state_dict(),
                os.path.join(final_model_dir, 'pytorch_model.bin'),
            )
            print(f"  - Best model saved to: {final_model_dir}")
        self._barrier()
        
        # Final evaluation (on best model)
        val_metrics = self.evaluate(val_loader, device, 'validation')
        test_metrics = self.evaluate(test_loader, device, 'test')
        
        # Save summary
        if self._is_rank0():
            summary = {
                'validation': val_metrics,
                'test': test_metrics,
                'best_val_auroc': best_val_auroc,
                'eval_epoch_fractions': fractions if use_fraction_schedule else None,
                'supervised_epochs': num_epochs,
            }
            save_results(summary, self.output_dir, 'summary.json')
            
            print("\n" + "=" * 80)
            print(f"{self._backbone_label().title()} Transformer Baseline Complete!")
            print("=" * 80)
        
        return val_metrics, test_metrics

    def run_eval_only(self, checkpoint_path: str):
        """Load a saved encoder (``best_model.pt`` or a run directory) and score."""
        from src.training.training_utils import resolve_weight_path

        setup_output_dir(self.output_dir, overwrite=False)
        device = self._setup_distributed()
        dataset_name = self.data_config.get("dataset", "CPRD")
        stage2_task = "prediction" if is_aces_dataset(dataset_name) else None
        datasets = self._load_datasets(
            ["train", "tuning", "held_out"], task_override=stage2_task
        )
        filtered = {k: FilteredTokenDataset(v) for k, v in datasets.items()}
        self.build_vocabulary(filtered)
        weight = resolve_weight_path(checkpoint_path)
        self.create_model()
        self.model = self.model.to(device)
        state = torch.load(weight, map_location=device)
        if isinstance(state, dict) and (
            "encoder_state_dict" in state
            or "decoder_state_dict" in state
            or "vocab" in state
        ):
            self._load_pretrained_checkpoint(weight, device)
        else:
            self.model.load_state_dict(state, strict=False)

        batch_size = int(self.training_config.get("batch_size", 32))
        workers = int(self.training_config.get("dataloader_num_workers", 4))
        val_loader = self._make_loader(
            filtered["tuning"],
            batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
        )
        test_loader = self._make_loader(
            filtered["held_out"],
            batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
        )
        val_metrics = self.evaluate(val_loader, device, "validation")
        test_metrics = self.evaluate(test_loader, device, "test")
        return val_metrics, test_metrics

