"""
Qwen3-Embedding based classifier.

Wraps Qwen3-Embedding-8B with a classification head for EHR-based prediction.
Uses CLS-token pooling (first non-padding token) and L2-normalized embeddings.
Supports freezing the base model and training only the head (and optional LoRA adapters).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from torch import Tensor


def cls_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Pool embeddings using the CLS token representation.

    We assume sequences are right-padded (as produced by the collator), so
    the CLS token is at position 0 for every sequence. As a safety check,
    we still respect the attention_mask and treat the first non-padding
    position as CLS if needed.

    Args:
        last_hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len], 1 = real token, 0 = pad

    Returns:
        pooled: [batch_size, hidden_dim]
    """
    batch_size, seq_len, _ = last_hidden_states.shape

    # Index of first non-padding token per sequence
    # (argmax over attention_mask along seq dim gives first 1 when pads are 0s)
    first_non_pad = attention_mask.float().argmax(dim=1)
    pooled = last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        first_non_pad,
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

        # Pool: CLS token (first non-padding token)
        pooled = cls_pool(outputs.last_hidden_state, attention_mask)

        # Debug print (once) to confirm CLS pooling behaviour and shapes
        if not self._debug_pooled_once:
            print("\n[Qwen3EmbeddingClassifier] Using CLS pooling for classification head.")
            print(f"  - input_ids shape: {tuple(input_ids.shape)}")
            print(f"  - attention_mask shape: {tuple(attention_mask.shape)}")
            print(f"  - pooled (CLS) shape: {tuple(pooled.shape)}")
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
