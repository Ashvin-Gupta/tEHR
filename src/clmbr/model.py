"""Minimal FEMRTransformer forward pass (no xformers)."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file


class RMSNorm(nn.Module):
    """FEMR 0.2.3 / Llama-style RMSNorm (float32 variance, then cast back)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return self.weight * x32.to(dtype)


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    x1 = flat[:, ::2]
    x2 = flat[:, 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape(x.shape)


def _rotary(ages: torch.Tensor, dim: int, dtype: torch.dtype):
    inv_freq = 1.0 / (10000 ** (torch.linspace(0, 2, steps=dim // 2, device=ages.device)))
    inv_freq = inv_freq.reshape(1, 1, dim // 2)
    if ages.dim() == 1:
        ages_f = ages.reshape(ages.shape[0], 1).to(torch.float32)
        t = inv_freq * ages_f
        shape = (ages.shape[0], 1, dim)
    else:
        ages_f = ages.to(torch.float32).unsqueeze(-1)
        t = ages_f * inv_freq
        shape = (ages.shape[0], ages.shape[1], 1, dim)
    sin, cos = torch.sin(t), torch.cos(t)
    sin = torch.stack((sin, sin), dim=-1).reshape(shape).to(dtype)
    cos = torch.stack((cos, cos), dim=-1).reshape(shape).to(dtype)
    return sin, cos


def _apply_rotary(x: torch.Tensor, sincos) -> torch.Tensor:
    sin, cos = sincos
    sin = sin.to(dtype=x.dtype)
    cos = cos.to(dtype=x.dtype)
    if sin.dim() != x.dim():
        sin = sin.reshape((1,) + sin.shape)
        cos = cos.reshape((1,) + cos.shape)
    return (x * cos) + (_rotate_every_two(x) * sin)


class FEMREncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, n_heads=12, use_bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.norm = RMSNorm(hidden_size)
        self.input_proj = nn.Linear(hidden_size, hidden_size * 3 + intermediate_size, bias=use_bias)
        self.output_proj = nn.Linear(hidden_size + intermediate_size, hidden_size, bias=use_bias)

    def forward(self, x, pos_embed, attn_mask):
        # x: (B, L, H)
        x = self.norm(x)
        transformed = self.input_proj(x)
        ff = transformed[..., : -self.hidden_size * 3]
        qkv = transformed[..., -self.hidden_size * 3 :]
        batch, length, _ = x.shape
        head = self.hidden_size // self.n_heads
        qkv = qkv.reshape(batch, length, 3, self.n_heads, head)
        q = _apply_rotary(qkv[:, :, 0], pos_embed)
        k = _apply_rotary(qkv[:, :, 1], pos_embed)
        v = qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.permute(0, 2, 1, 3).reshape(batch, length, self.hidden_size)
        ff = F.gelu(ff)
        return self.output_proj(torch.cat((attn, ff), dim=-1))


class ClmbrTransformer(nn.Module):
    def __init__(self, vocab_size=65536, n_layers=12, hidden_size=768, n_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.in_norm = RMSNorm(hidden_size)
        self.out_norm = RMSNorm(hidden_size)
        self.layers = nn.ModuleList(
            [FEMREncoderLayer(hidden_size=hidden_size, n_heads=n_heads) for _ in range(n_layers)]
        )

    def forward(self, tokens: torch.Tensor, ages: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        # `ages` must be days since birth (FEMR 0.2.3 BatchCreator), not seconds.
        unbatched = tokens.dim() == 1
        if unbatched:
            tokens = tokens.unsqueeze(0)
            ages = ages.unsqueeze(0)
        x = self.embed(tokens)
        x = self.in_norm(x)
        pos_embed = _rotary(ages, self.hidden_size // self.n_heads, x.dtype)
        for layer in self.layers:
            x = x + layer(x, pos_embed, attn_mask)
        x = self.out_norm(x)
        return x[0] if unbatched else x


def load_clmbr_transformer(model_dir: str, device: torch.device) -> ClmbrTransformer:
    weights = load_file(os.path.join(model_dir, "model.safetensors"), device=str(device))
    model = ClmbrTransformer()
    mapped = {}
    for k, v in weights.items():
        if k.startswith("transformer."):
            mapped[k[len("transformer.") :]] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        raise RuntimeError(f"Missing CLMBR weights: {missing[:8]}")
    model.to(device)
    model.eval()
    return model


def causal_local_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    """Additive mask: 0 = keep, -inf = drop. Causal and within `window` tokens."""
    idx = torch.arange(length, device=device)
    # delta[q, k] = q - k. Keep past and self (0 <= q-k < window).
    delta = idx.unsqueeze(1) - idx.unsqueeze(0)
    keep = (delta >= 0) & (delta < window)
    # SDPA wants additive mask or boolean with True=keep depending on version.
    # Use additive -inf for dropped positions, shape (1, 1, L, L).
    additive = torch.zeros((1, 1, length, length), device=device)
    additive.masked_fill_(~keep.unsqueeze(0).unsqueeze(0), float("-inf"))
    return additive


def causal_local_mask_padded(
    lengths: torch.Tensor, max_len: int, window: int, device: torch.device
) -> torch.Tensor:
    """Batched additive mask. Pad positions neither attend nor get attended to."""
    idx = torch.arange(max_len, device=device)
    delta = idx.unsqueeze(1) - idx.unsqueeze(0)
    local = (delta >= 0) & (delta < window)
    valid_q = idx.unsqueeze(0) < lengths.unsqueeze(1)
    valid_k = idx.unsqueeze(0) < lengths.unsqueeze(1)
    keep = local.unsqueeze(0) & valid_q.unsqueeze(2) & valid_k.unsqueeze(1)
    additive = torch.zeros((lengths.shape[0], 1, max_len, max_len), device=device)
    additive.masked_fill_(~keep.unsqueeze(1), float("-inf"))
    return additive


class ClmbrClassifier(nn.Module):
    """CLMBR-T backbone + linear head on the last real token."""

    def __init__(self, transformer: ClmbrTransformer):
        super().__init__()
        self.transformer = transformer
        self.head = nn.Linear(transformer.hidden_size, 1)

    def forward(self, tokens, ages, attn_mask, lengths):
        hidden = self.transformer(tokens, ages, attn_mask)
        last = (lengths - 1).clamp(min=0)
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        return self.head(pooled).squeeze(-1)
