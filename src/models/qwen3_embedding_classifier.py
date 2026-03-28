"""
Qwen3-Embedding based classifier.

Wraps Qwen3-Embedding-8B with a classification head for EHR-based prediction.
Pools the last non-padded position (typically EOS immediately before right-padding)
and L2-normalizes before the linear head.
Supports freezing the base model and training only the head (and optional LoRA adapters).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, List
from torch import Tensor


def last_non_pad_pool(
    last_hidden_states: Tensor,
    sequence_lengths: Tensor,
) -> Tensor:
    """
    Pool hidden states at `sequence_lengths` (last non-pad index per row, 0-based).

    With right-padded batches from `ClassificationCollator`, those indices are
    usually EOS immediately before padding when EOS is the final real token.

    Args:
        last_hidden_states: [batch_size, seq_len, hidden_dim]
        sequence_lengths: [batch_size], each in [0, seq_len - 1]

    Returns:
        pooled: [batch_size, hidden_dim]
    """
    batch_size = last_hidden_states.size(0)
    pooled = last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]
    return pooled


class Qwen3EmbeddingClassifier(nn.Module):
    """
    Wrapper that adds a classification head on top of Qwen3-Embedding.
    
    Architecture:
        Qwen3-Embedding → Last Token Pool → L2 Normalize → Linear → Logits
    
    Per Qwen3-Embedding docs: use last_token_pool and normalize embeddings.
    No instruction prompt needed for document encoding (EHR text).
    
    Args:
        base_model: The Qwen3-Embedding AutoModel.
        hidden_size: Embedding dimension (4096 for Qwen3-Embedding-8B).
        num_labels: Number of output classes (2 for binary).
        head_dropout: Dropout before classification head.
        freeze_base: If True, freeze all base model parameters; only head (and LoRA if any) trainable.
        trainable_param_keywords: Parameter name substrings to keep trainable (e.g. ["lora_"] for LoRA).
        tokenizer: Optional tokenizer for one-time debug: verify pooled index is EOS.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 4096,
        num_labels: int = 2,
        head_dropout: float = 0.1,
        freeze_base: bool = True,
        trainable_param_keywords: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.trainable_param_keywords = trainable_param_keywords or []
        self.tokenizer = tokenizer

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            if self.trainable_param_keywords:
                reenabled = 0
                for name, param in self.base_model.named_parameters():
                    if any(kw in name for kw in self.trainable_param_keywords):
                        param.requires_grad = True
                        reenabled += 1
                if reenabled:
                    print(f"  - Re-enabled {reenabled} parameters matching: {self.trainable_param_keywords}")
            else:
                print("  - Froze all base model parameters (only classification head trainable)")
        
        self.classifier = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self._debug_pooled_once = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model to save memory."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {}
            )
            print("  - Gradient checkpointing enabled for base model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] optional
        
        Returns:
            Dict with 'logits' and 'loss' (if labels provided)
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        sequence_lengths = attention_mask.sum(dim=1).long() - 1
        pooled = last_non_pad_pool(outputs.last_hidden_state, sequence_lengths)

        # Debug print (once) to confirm pooling behaviour and EOS at pooled index
        if not self._debug_pooled_once:
            print("\n[Qwen3EmbeddingClassifier] Using last non-pad pooling (intended: EOS before padding).")
            print(f"  - input_ids shape: {tuple(input_ids.shape)}")
            print(f"  - attention_mask shape: {tuple(attention_mask.shape)}")
            print(f"  - pooled shape: {tuple(pooled.shape)}")
            eos_id = getattr(self.tokenizer, "eos_token_id", None) if self.tokenizer is not None else None
            if eos_id is None:
                print("  - EOS check: no tokenizer or eos_token_id; cannot verify pooled token.")
            else:
                batch_size = input_ids.size(0)
                print(f"  - tokenizer.eos_token_id: {eos_id}")
                for b in range(batch_size):
                    li = int(sequence_lengths[b].item())
                    tid = int(input_ids[b, li].item())
                    is_eos = tid == eos_id
                    dec = ""
                    if self.tokenizer is not None:
                        try:
                            dec = repr(self.tokenizer.decode([tid]))
                        except Exception:
                            dec = "(decode failed)"
                    print(
                        f"  - batch[{b}] last_non_pad_idx={li} token_id={tid} "
                        f"is_eos_token_id={is_eos} decode={dec}"
                    )
            self._debug_pooled_once = True

        # L2 normalize (per Qwen3-Embedding)
        pooled = F.normalize(pooled, p=2, dim=1)
        
        # Classification head
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits}
