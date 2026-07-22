"""
LLM-based classifier model.

Wraps a pretrained LLM with a classification head for EHR-based prediction.
"""
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
from typing import Dict, Optional


class LLMClassifier(nn.Module):
    """
    Wrapper that adds a classification head on top of a (frozen) LLM.
    
    Pooling strategies (pooling arg):
        'last_token': hidden state of the last non-padding token (default, original behaviour).
        'mean':       masked mean over all non-padding token hidden states.

    Head types (head_type arg):
        'linear':   nn.Linear(hidden_size, num_labels).  CrossEntropyLoss.
        'mlp':      Two-layer MLP with GELU.  CrossEntropyLoss.
        'deep_mlp': Three-layer MLP with GELU.  CrossEntropyLoss.
        'logistic': nn.Linear(hidden_size, 1).  BCEWithLogitsLoss (binary only).
    
    Args:
        base_model: The pretrained LLM (with or without LoRA adapters).
        hidden_size: Hidden dimension of the LLM.
        num_labels: Number of output classes (2 for binary classification).
        freeze_base: Whether to freeze the base LLM parameters.
        trainable_param_keywords: Substrings for parameters that should remain 
            trainable (e.g., ["lora_"] to train LoRA adapters).
        multi_label: If True, uses BCE loss (for multi-label tasks).
        tokenizer: Optional tokenizer (for debugging).
        pooling: 'last_token' or 'mean'.
    """
    
    def __init__(
        self,
        base_model,
        hidden_size: int,
        num_labels: int = 2,
        freeze_base: bool = True,
        trainable_param_keywords: Optional[list] = None,
        multi_label: bool = False,
        tokenizer=None,
        head_type: str = 'linear',
        head_hidden_size: int = 512,
        head_dropout: float = 0.1,
        pooling: str = 'last_token',
    ):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.multi_label = multi_label
        self.trainable_param_keywords = trainable_param_keywords or []
        self.tokenizer = tokenizer
        self.head_type = head_type
        self.pooling = pooling
        self._lora_verified = False
        # Enable gradient checkpointing if available
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            print("  - Enabling gradient checkpointing for base model")
            self.base_model.gradient_checkpointing_enable()
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("  - Froze all base model parameters")
        
        # Selectively unfreeze parameters matching keywords (e.g., LoRA adapters)
        if self.trainable_param_keywords:
            reenabled = 0
            for name, param in self.base_model.named_parameters():
                if any(keyword in name for keyword in self.trainable_param_keywords):
                    param.requires_grad = True
                    reenabled += 1
            print(f"  - Re-enabled {reenabled} parameters matching: {self.trainable_param_keywords}")
        
        # Create classification head based on type
        self.classifier = self._create_classifier_head(
            hidden_size, num_labels, head_type, head_hidden_size, head_dropout
        )
    
    def _create_classifier_head(
        self, 
        hidden_size: int, 
        num_labels: int,
        head_type: str,
        head_hidden_size: int,
        head_dropout: float
    ) -> nn.Module:
        """
        Create classification head based on type.
        
        Args:
            hidden_size: Input size from LLM.
            num_labels: Number of output classes.
            head_type: 'linear', 'mlp', 'deep_mlp', or 'logistic'.
                - 'linear':   nn.Linear(hidden_size, num_labels).  CrossEntropyLoss.
                - 'mlp':      Linear -> GELU -> Dropout -> Linear.  CrossEntropyLoss.
                - 'deep_mlp': Two hidden layers.  CrossEntropyLoss.
                - 'logistic': nn.Linear(hidden_size, 1). BCEWithLogitsLoss (binary only).
            head_hidden_size: Hidden size for MLP/deep_mlp heads.
            head_dropout: Dropout rate for MLP/deep_mlp heads.
        
        Returns:
            Classification head module.
        """
        if head_type == 'linear':
            head = nn.Linear(hidden_size, num_labels)
            print(f"  - Added LINEAR classification head: {hidden_size} -> {num_labels}")

        elif head_type == 'mlp':
            head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_size),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_size, num_labels),
            )
            print(
                f"  - Added MLP classification head: "
                f"{hidden_size} -> {head_hidden_size} -> {num_labels}"
            )

        elif head_type == 'deep_mlp':
            head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_size),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_size, head_hidden_size // 2),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_size // 2, num_labels),
            )
            print(
                f"  - Added DEEP_MLP classification head: "
                f"{hidden_size} -> {head_hidden_size} -> {head_hidden_size // 2} -> {num_labels}"
            )

        elif head_type == 'logistic':
            head = nn.Linear(hidden_size, 1)
            print(
                f"  - Added LOGISTIC classification head: {hidden_size} -> 1 "
                f"(BCEWithLogitsLoss, binary)"
            )

        else:
            raise ValueError(
                f"Unknown head_type: '{head_type}'. "
                f"Options: 'linear', 'mlp', 'deep_mlp', 'logistic'"
            )
        
        return head
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate gradient checkpointing to base model."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        """Helper needed by Trainer to verify model compatibility."""
        return self.base_model.get_input_embeddings()

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden states into a single patient-level vector.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            (batch_size, hidden_size)
        """
        if self.pooling == 'mean':
            mask_expanded = attention_mask.float().unsqueeze(-1)
            summed = (hidden_states * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1)
            return summed / counts
        else:  # last_token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            return hidden_states[range(batch_size), sequence_lengths]

    def _forward_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Run the decoder backbone and return hidden states.

        Unsloth's causal-LM wrapper sets ``_has_no_labels = True`` when LM labels
        are omitted (classification). That enables a broken
        ``inputs_embeds *= attention_mask`` path if ``max_seq_length`` is stale
        (often 4096) while sequences are 12000. Hitting the inner model and
        clearing the flag avoids it.

        Still required on Isambard (GH200 / torch 2.12) even with Unsloth 2026.2.x;
        the other cluster (x86 / torch 2.10) can survive the direct backbone path.
        """
        backbone = getattr(self.base_model, "model", self.base_model)
        inner = getattr(backbone, "model", None)
        if inner is not None:
            inner._has_no_labels = False
            return inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    def get_patient_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the patient-level hidden representation used for classification.
        Does NOT apply the classification head.

        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) attention mask.

        Returns:
            Tensor of shape (batch_size, hidden_size) representing each patient.
        """
        outputs = self._forward_backbone(input_ids, attention_mask)

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]

        return self._pool(hidden_states, attention_mask)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) attention mask.
            labels: (batch_size,) ground truth labels.
        
        Returns:
            Dict with 'loss' and 'logits'.
        """
        # One-time check: verify LoRA adapters exist and are in the call graph.
        if not self._lora_verified:
            lora_params = [(n, p) for n, p in self.base_model.named_parameters() if "lora_A" in n]
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
                    "  [LoRA check] *** No lora_A parameters found in self.base_model! "
                    "LoRA may not be active. ***"
                )
            self._lora_verified = True

        outputs = self._forward_backbone(input_ids, attention_mask)
        
        # Extract last layer's hidden states: (batch_size, seq_len, hidden_size)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]
        
        # Pool into patient-level representation
        pooled = self._pool(hidden_states, attention_mask)
        
        # Pass through classification head
        logits = self.classifier(pooled)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if self.head_type == 'logistic':
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.squeeze(1), labels.float())
            elif self.multi_label:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
        }
