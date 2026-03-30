# -*- coding: utf-8 -*-
"""Inverted-transformer-style conditional forecaster for AGC multi-step forecasting."""

import math

import torch
import torch.nn as nn


class ConditionalITransformerForecaster(nn.Module):
    """Model cross-variate relations by treating variables as tokens."""

    def __init__(
        self,
        seq_len: int,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        ff_dim: int = 192,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.variable_proj = nn.Linear(seq_len, hidden_dim)
        self.variable_pos = nn.Parameter(torch.zeros(1, past_dim, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.future_proj = nn.Sequential(
            nn.Linear(weather_dim + control_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        var_tokens = self.variable_proj(x_past.transpose(1, 2)) + self.variable_pos
        memory = self.encoder(var_tokens)
        future_tokens = self.future_proj(torch.cat([w_future, u_future], dim=-1))
        attn_scores = torch.matmul(future_tokens, memory.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        variable_context = torch.matmul(attn_weights, memory)
        fused = self.fuse(torch.cat([future_tokens, variable_context], dim=-1))
        return self.head(future_tokens + fused)
