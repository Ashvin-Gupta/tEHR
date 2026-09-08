"""
Experiment: decoder — causal transformer (decoder-only) on filtered EHR tokens.

Same token filter, architecture size, and stage-2 classification head as
``encoder``. Stage 1 is next-token prediction instead of MLM.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.baselines.encoder import (
    EHRTransformerEncoder,
    EncoderBaseline,
)


class EHRTransformerDecoder(EHRTransformerEncoder):
    """
    Decoder-only transformer: same stack as the encoder, with a causal mask
    so each position may only attend to itself and earlier tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Causal attn_mask + padding mask trips the nested-tensor fast path.
        if hasattr(self.transformer, "enable_nested_tensor"):
            self.transformer.enable_nested_tensor = False

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        # Bool subsequent mask (True = blocked) so it matches src_key_padding_mask.
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape

        if seq_len > self.max_seq_length:
            token_ids = token_ids[:, -self.max_seq_length :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_seq_length :]
            seq_len = self.max_seq_length

        x = self.token_embedding(token_ids)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        x = self.dropout(x)

        if attention_mask is not None:
            pad_mask = attention_mask == 0
        else:
            pad_mask = None

        x = self.transformer(
            x,
            mask=self._causal_mask(seq_len, x.device),
            src_key_padding_mask=pad_mask,
        )
        return x

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Classification: same pooling/head as the encoder, causal backbone."""
        batch_size, seq_len = token_ids.shape

        if seq_len > self.max_seq_length:
            token_ids = token_ids[:, -self.max_seq_length :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_seq_length :]
            seq_len = self.max_seq_length

        if self.pooling == "cls":
            # Causal CLS must sit at the end so it can see the whole sequence.
            x = self.token_embedding(token_ids)
            x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
            x = self.dropout(x)
            cls = self.cls_token.expand(batch_size, -1, -1)
            cls = cls + self.pos_encoding[seq_len : seq_len + 1, :].unsqueeze(0)
            x = torch.cat([x, cls], dim=1)
            if attention_mask is not None:
                cls_mask = torch.ones(
                    batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device
                )
                full_mask = torch.cat([attention_mask, cls_mask], dim=1)
                pad_mask = full_mask == 0
            else:
                pad_mask = None
            causal_len = seq_len + 1
            x = self.transformer(
                x,
                mask=self._causal_mask(causal_len, x.device),
                src_key_padding_mask=pad_mask,
            )
            pooled = x[:, -1, :]
        else:
            x = self.encode(token_ids, attention_mask)
            if attention_mask is not None:
                mask_expanded = attention_mask.float().unsqueeze(-1)
                x = x * mask_expanded
                pooled = x.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)

        return self.classifier(pooled)


class EHRTransformerNTPHead(nn.Module):
    """Causal LM head: at position t, predict token t+1."""

    def __init__(self, decoder: EHRTransformerDecoder, vocab_size: int):
        super().__init__()
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(decoder.hidden_dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        del labels  # labels are the input tokens; shift is done here
        sequence_output = self.decoder.encode(token_ids, attention_mask)
        logits = self.lm_head(sequence_output)

        if token_ids.size(1) < 2:
            loss = logits.sum() * 0.0
            return loss, logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = token_ids[:, 1:].contiguous()
        if attention_mask is not None:
            shift_labels = shift_labels.clone()
            shift_labels[attention_mask[:, 1:] == 0] = -100

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        return loss, logits


class DecoderBaseline(EncoderBaseline):
    """Decoder-only transformer: NTP stage 1, then the encoder's stage-2 head."""

    def _backbone_label(self) -> str:
        return "decoder"

    def _pretrain_label(self) -> str:
        return "NTP"

    def _experiment_label(self) -> str:
        return "decoder"

    def _stage1_ckpt_basename(self) -> str:
        return "pretrained_decoder_stage1_ntp.pt"

    def _stage1_weight_key(self) -> str:
        return "decoder_state_dict"

    def mlm_collate_fn(self, batch):
        """NTP uses the raw token stream; the head shifts labels internally."""
        base = self.collate_fn(batch)
        return {
            "token_ids": base["token_ids"],
            "attention_mask": base["attention_mask"],
            "labels": base["token_ids"],
        }

    def create_model(self):
        print("\n" + "=" * 80)
        print("Creating transformer decoder model...")
        print("=" * 80)

        vocab_size = int(self.vocab["size"])
        hidden_dim = int(self.model_config.get("hidden_dim", 512))
        num_layers = int(self.model_config.get("num_layers", 6))
        num_heads = int(self.model_config.get("num_heads", 8))
        ff_dim = int(self.model_config.get("ff_dim", 2048))
        dropout = float(self.model_config.get("dropout", 0.1))
        max_seq_length = int(self.model_config.get("max_seq_length", 512))
        num_labels = int(self.model_config.get("num_labels", 2))
        head_type = self.model_config.get("head_type", "linear")
        head_hidden_size = int(self.model_config.get("head_hidden_size", 256))
        pooling = self.model_config.get("pooling", "mean")

        self.max_seq_length = max_seq_length
        self.head_type = head_type
        print(f"  - head_type: {head_type}, pooling: {pooling}, causal=True")

        self.model = EHRTransformerDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_length=max_seq_length,
            num_labels=num_labels,
            head_type=head_type,
            head_hidden_size=head_hidden_size,
            pooling=pooling,
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

    def create_mlm_model(self):
        if self.model is None:
            self.create_model()
        self.mlm_model = EHRTransformerNTPHead(
            self.model, vocab_size=int(self.vocab["size"])
        )
