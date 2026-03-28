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
from typing import Dict, Optional, List
from torch import Tensor


def last_non_pad_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Pool using the last real token (last index with attention_mask == 1).

    With right-padded batches from `ClassificationCollator`, the EOS appended at
    the end of the trajectory is usually this position — i.e. EOS immediately
    before padding.

    Args:
        last_hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len], 1 = real token, 0 = pad

    Returns:
        pooled: [batch_size, hidden_dim]
    """
    batch_size = last_hidden_states.size(0)
    # Index of last non-padding token per sequence (0-based)
    sequence_lengths = attention_mask.sum(dim=1).long() - 1
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
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 4096,
        num_labels: int = 2,
        head_dropout: float = 0.1,
        freeze_base: bool = True,
        trainable_param_keywords: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.trainable_param_keywords = trainable_param_keywords or []

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

        # Pool: last non-padded token (typically EOS before right-padding)
        pooled = last_non_pad_pool(outputs.last_hidden_state, attention_mask)

        # Debug print (once) to confirm pooling behaviour and shapes
        if not self._debug_pooled_once:
            print("\n[Qwen3EmbeddingClassifier] Using last non-pad pooling (EOS before padding).")
            print(f"  - input_ids shape: {tuple(input_ids.shape)}")
            print(f"  - attention_mask shape: {tuple(attention_mask.shape)}")
            print(f"  - pooled shape: {tuple(pooled.shape)}")
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
