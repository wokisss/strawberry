# -*- coding: utf-8 -*-
"""Transformer-hybrid conditional baseline for AGC multi-step forecasting."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first tensors."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ConditionalTransformerHybridForecaster(nn.Module):
    """Encode history with self-attention and decode future with cross-attention."""

    def __init__(
        self,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        ff_dim: int = 192,
        max_past_len: int = 512,
        max_future_len: int = 128,
    ):
        super().__init__()
        self.past_proj = nn.Linear(past_dim, hidden_dim)
        self.future_proj = nn.Linear(weather_dim + control_dim, hidden_dim)
        self.pos_past = PositionalEncoding(hidden_dim, max_len=max_past_len)
        self.pos_future = PositionalEncoding(hidden_dim, max_len=max_future_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        past_tokens = self.pos_past(self.past_proj(x_past))
        memory = self.encoder(past_tokens)
        future_known = torch.cat([w_future, u_future], dim=-1)
        future_tokens = self.pos_future(self.future_proj(future_known))
        decoded = self.decoder(future_tokens, memory)
        global_context = memory.mean(dim=1, keepdim=True).expand(-1, decoded.size(1), -1)
        fused = self.context_gate(torch.cat([decoded, global_context], dim=-1))
        return self.head(decoded + fused)
