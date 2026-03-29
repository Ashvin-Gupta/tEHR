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
    
    The LLM extracts hidden states from the EOS token position,
    and we pass it through a linear classification head.
    
    Architecture:
        LLM (frozen/partially frozen) → Last Hidden State → Linear → Logits
    
    Args:
        base_model: The pretrained LLM (with or without LoRA adapters).
        hidden_size: Hidden dimension of the LLM.
        num_labels: Number of output classes (2 for binary classification).
        freeze_base: Whether to freeze the base LLM parameters.
        trainable_param_keywords: Substrings for parameters that should remain 
            trainable (e.g., ["lora_"] to train LoRA adapters).
        multi_label: If True, uses BCE loss (for multi-label tasks).
        tokenizer: Optional tokenizer (for debugging).
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
        head_dropout: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.multi_label = multi_label
        self.trainable_param_keywords = trainable_param_keywords or []
        self.tokenizer = tokenizer
        self.head_type = head_type
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
            head_type: Type of head ('linear', 'mlp', 'deep_mlp').
            head_hidden_size: Hidden size for MLP heads.
            head_dropout: Dropout rate for MLP heads.
        
        Returns:
            Classification head module.
        """
        if head_type == 'linear':
            # Simple logistic regression
            head = nn.Linear(hidden_size, num_labels)
            print(f"  - Added LINEAR classification head: {hidden_size} -> {num_labels}")
        
        else:
            raise ValueError(
                f"Unknown head_type: '{head_type}'. "
                f"Options: 'linear'"
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

    def get_patient_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the patient-level hidden representation used for classification.

        This mirrors the forward pass up to (and including) the step where we
        extract the last non-padding token's hidden state. It does NOT apply
        the classification head.

        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) attention mask.

        Returns:
            Tensor of shape (batch_size, hidden_size) representing each patient.
        """
        backbone = getattr(self.base_model, "model", self.base_model)

        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.size(0)
        last_hidden_states = hidden_states[range(batch_size), sequence_lengths]

        return last_hidden_states
    
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
        # Get the base model (unwrap if needed).
        # For PEFT/Unsloth models self.base_model is a PeftModel; .model is the
        # underlying MistralForCausalLM.  LoRA is injected at the linear-layer
        # level so it IS applied even when calling the base model directly.
        backbone = getattr(self.base_model, "model", self.base_model)

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

        # Forward pass
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract last layer's hidden states: (batch_size, seq_len, hidden_size)
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]
        
        # Get the last non-padding token's hidden state (intended to be EOS)
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_size = hidden_states.size(0)
        last_hidden_states = hidden_states[range(batch_size), sequence_lengths]
        
        
        # Pass through classification head (logistic regression head by default)
        logits = self.classifier(last_hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if self.multi_label:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
        }
