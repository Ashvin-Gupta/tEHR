"""
LLM-based classifier model.

Wraps a pretrained LLM with a classification head for EHR-based prediction.
"""
import torch
import torch.nn as nn
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
        
        # Flag to track EOS token verification (done once)
        self._eos_verified = False

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
        
        elif head_type == 'mlp':
            # Single hidden layer MLP
            head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_size),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_size, num_labels)
            )
            print(f"  - Added MLP classification head: {hidden_size} -> {head_hidden_size} -> {num_labels}")
        
        elif head_type == 'deep_mlp':
            # Two hidden layer MLP
            head = nn.Sequential(
                nn.Linear(hidden_size, head_hidden_size),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_size, head_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_size // 2, num_labels)
            )
            print(f"  - Added DEEP MLP classification head: {hidden_size} -> {head_hidden_size} -> {head_hidden_size // 2} -> {num_labels}")
        
        else:
            raise ValueError(
                f"Unknown head_type: '{head_type}'. "
                f"Options: 'linear', 'mlp', 'deep_mlp'"
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
    
    def _verify_eos_token(
        self, 
        input_ids: torch.Tensor, 
        sequence_lengths: torch.Tensor, 
        labels: torch.Tensor
    ):
        """
        Verify that the token used for classification is actually the EOS token.
        Checks one positive case and one negative control.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs.
            sequence_lengths: (batch_size,) position of last non-padding token.
            labels: (batch_size,) ground truth labels.
        """
        print("\n" + "="*80)
        print("EOS TOKEN VERIFICATION")
        print("="*80)
        
        # Get EOS token ID
        if hasattr(self.tokenizer, 'eos_token_id'):
            eos_token_id = self.tokenizer.eos_token_id
        else:
            print("⚠️  WARNING: Tokenizer does not have eos_token_id attribute!")
            return
        
        print(f"Expected EOS token ID: {eos_token_id}")
        print(f"EOS token string: '{self.tokenizer.eos_token}'\n")
        
        # Find one case (label=1) and one control (label=0)
        positive_idx = None
        negative_idx = None
        
        for i, label in enumerate(labels):
            if label == 1 and positive_idx is None:
                positive_idx = i
            elif label == 0 and negative_idx is None:
                negative_idx = i
            
            if positive_idx is not None and negative_idx is not None:
                break
        
        # Verify both samples
        for sample_type, idx in [("CASE (label=1)", positive_idx), 
                                   ("CONTROL (label=0)", negative_idx)]:
            if idx is None:
                print(f"⚠️  {sample_type}: Not found in this batch\n")
                continue
            
            seq_len = sequence_lengths[idx].item()
            actual_token_id = input_ids[idx, seq_len].item()
            
            # Get token string representation
            actual_token_str = self.tokenizer.decode([actual_token_id])
            
            # Check if it matches EOS
            is_eos = (actual_token_id == eos_token_id)
            status = "✓ CORRECT" if is_eos else "✗ INCORRECT"
            
            print(f"{sample_type}:")
            print(f"  Position used for classification: {seq_len}")
            print(f"  Actual token ID at that position: {actual_token_id}")
            print(f"  Token string: '{actual_token_str}'")
            print(f"  Status: {status}")
            
            if not is_eos:
                print(f"  ⚠️  WARNING: Expected EOS token (ID={eos_token_id}), got {actual_token_id}!")
                # Show context around this position
                context_start = max(0, seq_len - 3)
                context_end = min(input_ids.size(1), seq_len + 3)
                context_ids = input_ids[idx, context_start:context_end].tolist()
                context_str = self.tokenizer.decode(context_ids)
                print(f"  Context tokens: {context_ids}")
                print(f"  Context string: '{context_str}'")
            print()
        
        print("="*80 + "\n")
    
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
        # Get the base model (unwrap if needed)
        backbone = getattr(self.base_model, "model", self.base_model)

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
        
        # Get the last non-padding token's hidden state (usually EOS token)
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_size = hidden_states.size(0)
        last_hidden_states = hidden_states[range(batch_size), sequence_lengths]
        
        # === VERIFICATION: Check EOS token (run once) ===
        if not self._eos_verified and self.tokenizer is not None and labels is not None:
            self._verify_eos_token(input_ids, sequence_lengths, labels)
            self._eos_verified = True
        
        # Pass through classification head
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
