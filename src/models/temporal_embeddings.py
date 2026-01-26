"""
Temporal embedding module for EHR sequences.

Converts delta times (time between consecutive events) into learned embeddings
that can be added to token embeddings.
"""
import torch
import torch.nn as nn
import math


class TemporalEmbedding(nn.Module):
    """
    Learned temporal embeddings for delta times between events.
    
    Uses log scaling to handle varying time ranges (hours to years):
        embedding = linear(log(1 + delta_time / time_scale))
    
    Args:
        hidden_size: Dimension of embeddings (must match model hidden size).
        time_scale: Scaling factor for delta times.
            - 1.0 for seconds
            - 3600.0 for hours  
            - 86400.0 for days (recommended for clinical data)
        dropout: Dropout rate for regularization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        time_scale: float = 86400.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_scale = time_scale
        
        # Linear projection from log-scaled time to hidden dimension
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        # Initialize with small weights
        nn.init.normal_(self.time_proj[0].weight, std=0.02)
        nn.init.zeros_(self.time_proj[0].bias)
    
    def forward(self, delta_times: torch.Tensor) -> torch.Tensor:
        """
        Convert delta times to embeddings.
        
        Args:
            delta_times: (batch_size, seq_len) tensor of delta times in seconds.
                         Already computed as time_i - time_{i-1}.
        
        Returns:
            Temporal embeddings: (batch_size, seq_len, hidden_size)
        """
        # Ensure non-negative and convert to float
        delta_times = torch.clamp(delta_times, min=0.0).float()
        
        # Apply log scaling: log(1 + delta_time / time_scale)
        # +1 prevents log(0) for first event or same-time events
        scaled_times = torch.log1p(delta_times / self.time_scale)
        
        # Add dimension for linear layer: (batch, seq_len) -> (batch, seq_len, 1)
        scaled_times = scaled_times.unsqueeze(-1)
        
        # Project to hidden dimension
        temporal_embeds = self.time_proj(scaled_times)
        
        return temporal_embeds
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'hidden_size={self.hidden_size}, time_scale={self.time_scale}'
