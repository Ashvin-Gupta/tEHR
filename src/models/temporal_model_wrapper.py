"""
Wrapper model that adds temporal embedding support to any LLM for pretraining.

This wrapper intercepts the forward pass and adds temporal embeddings to token embeddings.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from src.models.temporal_embeddings import TemporalEmbedding


class TemporalModelWrapper(nn.Module):
    """
    Wraps an LLM to add temporal embedding support.
    
    This wrapper:
    1. Accepts delta_times in addition to standard inputs
    2. Gets token embeddings from the base model
    3. Adds temporal embeddings element-wise
    4. Passes combined embeddings through the base model
    
    Args:
        base_model: The LLM to wrap.
        hidden_size: Hidden dimension of the model.
        time_scale: Scaling factor for log scaling (days=86400, hours=3600, seconds=1).
        dropout: Dropout rate for temporal embeddings.
    """
    
    def __init__(
        self,
        base_model,
        hidden_size: int,
        time_scale: float = 86400.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        self.temporal_embedder = TemporalEmbedding(
            hidden_size=hidden_size,
            time_scale=time_scale,
            dropout=dropout
        )
        print(f"  - Wrapped model with temporal embeddings (time_scale={time_scale})")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        delta_times: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with temporal embeddings.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            delta_times: Log-scaled delta times (batch_size, seq_len).
            labels: Labels for language modeling.
            **kwargs: Other arguments passed to base model.
        
        Returns:
            Model outputs.
        """
        # If delta_times provided, use temporal embeddings
        if delta_times is not None:
            # Get token embeddings
            embedding_layer = self.base_model.get_input_embeddings()
            token_embeddings = embedding_layer(input_ids)
            
            # Get temporal embeddings
            temporal_embeds = self.temporal_embedder(delta_times)
            
            # Add temporal embeddings to token embeddings
            combined_embeddings = token_embeddings + temporal_embeds
            
            # Forward pass with combined embeddings
            outputs = self.base_model(
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            # Standard forward pass without temporal embeddings
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        return outputs
    
    def get_input_embeddings(self):
        """Delegate to base model."""
        return self.base_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Delegate to base model."""
        return self.base_model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'get_output_embeddings'):
            return self.base_model.get_output_embeddings()
        return None
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate to base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
    
    def gradient_checkpointing_disable(self):
        """Delegate to base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
    
    def save_pretrained(self, *args, **kwargs):
        """Delegate to base model (temporal embeddings are saved separately)."""
        return self.base_model.save_pretrained(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to base model if not found in wrapper."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
