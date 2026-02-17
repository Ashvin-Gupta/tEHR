"""
Qwen3-Embedding masked language modeling (MLM) head.

This wraps the Qwen3-Embedding encoder with a token-level prediction head for
masked language modeling style continued pretraining.

Architecture:
    Qwen3-Embedding → last_hidden_state → Linear(vocab_size) → token logits

The base model can be (optionally) frozen apart from LoRA adapters, mirroring
the classification setup.
"""

from typing import Dict, Optional, List

import torch
import torch.nn as nn


class Qwen3EmbeddingMLMHead(nn.Module):
    """
    Wrapper that adds an MLM head on top of Qwen3-Embedding.

    Args:
        base_model: The Qwen3-Embedding `AutoModel`.
        hidden_size: Embedding dimension (4096 for Qwen3-Embedding-8B).
        vocab_size: Size of tokenizer vocabulary.
        freeze_base: If True, freeze all base model parameters; only head
            (and LoRA if any) are trainable.
        trainable_param_keywords: Parameter name substrings to keep trainable
            (e.g. ["lora_"] for LoRA).
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        vocab_size: int,
        freeze_base: bool = True,
        trainable_param_keywords: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
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
                    print(
                        f"  - [MLM] Re-enabled {reenabled} parameters matching: "
                        f"{self.trainable_param_keywords}"
                    )
            else:
                print("  - [MLM] Froze all base model parameters (only MLM head trainable)")

        # Simple token-level head: Linear(hidden_size → vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MLM.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)
            labels: [batch_size, seq_len] with -100 for non-masked tokens.

        Returns:
            Dict with 'logits' (token logits) and 'loss' (if labels provided).
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state  # [B, T, H]
        logits = self.lm_head(sequence_output)       # [B, T, V]

        loss = None
        if labels is not None:
            # Cross-entropy over vocab, ignoring positions with label == -100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
            )

        return {"loss": loss, "logits": logits}

