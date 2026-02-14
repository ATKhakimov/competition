from __future__ import annotations

import torch
import torch.nn as nn


class TemporalFoundationEncoder(nn.Module):
    def __init__(
        self,
        event_vocab_size: int,
        channel_vocab_size: int,
        hour_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.event_emb = nn.Embedding(event_vocab_size, d_model, padding_idx=0)
        self.channel_emb = nn.Embedding(channel_vocab_size, d_model, padding_idx=0)
        self.hour_emb = nn.Embedding(hour_vocab_size, d_model, padding_idx=0)
        self.delta_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

        self.head_next_event = nn.Linear(d_model, event_vocab_size)
        self.head_next_channel = nn.Linear(d_model, channel_vocab_size)
        self.head_delta = nn.Linear(d_model, 1)

    def forward(
        self,
        event_token: torch.Tensor,
        channel_token: torch.Tensor,
        hour_token: torch.Tensor,
        delta_log: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz, seqlen = event_token.shape
        pos = torch.arange(seqlen, device=event_token.device).unsqueeze(0).expand(bsz, seqlen)
        x = (
            self.event_emb(event_token)
            + self.channel_emb(channel_token)
            + self.hour_emb(hour_token)
            + self.delta_proj(delta_log.unsqueeze(-1))
            + self.pos_emb(pos)
        )
        x = self.dropout(self.norm_in(x))
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        h = self.norm_out(x)
        return {
            "hidden": h,
            "next_event_logits": self.head_next_event(h),
            "next_channel_logits": self.head_next_channel(h),
            "next_delta_pred": self.head_delta(h).squeeze(-1),
        }

