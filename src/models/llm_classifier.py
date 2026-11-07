"""
Pooled LLM hidden state plus a linear classification head.

Used by tEHR stage 2 and the frozen-LLM linear baseline. Pooling is
``last_token`` (default) or ``mean`` over non-padding tokens.
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LLMClassifier(nn.Module):
    def __init__(
        self,
        base_model,
        hidden_size: int,
        num_labels: int = 2,
        freeze_base: bool = True,
        trainable_param_keywords: Optional[list] = None,
        pooling: str = "last_token",
    ):
        super().__init__()
        if pooling not in ("last_token", "mean"):
            raise ValueError(f"pooling must be 'last_token' or 'mean', got {pooling!r}")
        self.base_model = base_model
        self.num_labels = num_labels
        self.trainable_param_keywords = trainable_param_keywords or []
        self.pooling = pooling
        self._lora_verified = False

        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            print("  - Enabling gradient checkpointing for base model")
            self.base_model.gradient_checkpointing_enable()

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("  - Froze all base model parameters")

        if self.trainable_param_keywords:
            reenabled = 0
            for name, param in self.base_model.named_parameters():
                if any(keyword in name for keyword in self.trainable_param_keywords):
                    param.requires_grad = True
                    reenabled += 1
            print(
                f"  - Re-enabled {reenabled} parameters matching: "
                f"{self.trainable_param_keywords}"
            )

        self.classifier = nn.Linear(hidden_size, num_labels)
        print(f"  - Linear classification head: {hidden_size} -> {num_labels}")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask_expanded = attention_mask.float().unsqueeze(-1)
            summed = (hidden_states * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            return summed / counts
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.size(0)
        return hidden_states[range(batch_size), sequence_lengths]

    def _forward_backbone(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Run the decoder backbone and return hidden states.

        Unsloth's causal-LM wrapper sets ``_has_no_labels = True`` when LM labels
        are omitted (classification). That enables a broken
        ``inputs_embeds *= attention_mask`` path **during training** if
        ``max_seq_length`` is stale. Token-id eval still uses the outer
        CausalLM path (one unwrap) so last-token states match the original
        cluster. Token-id training, and every embedding forward, hits the
        inner decoder and clears the flag.

        Embedding eval cannot use the outer wrapper: it reads ``input_ids.shape``
        and the inner decoder rejects ``input_ids`` together with ``inputs_embeds``.
        """
        backbone = getattr(self.base_model, "model", self.base_model)
        inner = getattr(backbone, "model", None)
        if inner is not None:
            inner._has_no_labels = False
        use_inner = inner is not None and (self.training or inputs_embeds is not None)
        model = inner if use_inner else backbone
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds is required")
        kwargs = {
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        else:
            kwargs["input_ids"] = input_ids
        return model(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self._lora_verified:
            lora_params = [
                (n, p) for n, p in self.base_model.named_parameters() if "lora_A" in n
            ]
            if lora_params:
                name, param = lora_params[0]
                print(
                    f"  [LoRA check] Found {len(lora_params)} lora_A parameters. "
                    f"Example: '{name}' shape={tuple(param.shape)} "
                    f"requires_grad={param.requires_grad} "
                    f"mean={param.float().mean().item():.6f}"
                )
            else:
                logger.warning(
                    "  [LoRA check] No lora_A parameters found in self.base_model."
                )
            self._lora_verified = True

        outputs = self._forward_backbone(input_ids, attention_mask)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs.hidden_states[-1]

        logits = self.classifier(self._pool(hidden_states, attention_mask))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
