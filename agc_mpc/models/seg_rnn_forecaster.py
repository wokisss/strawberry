# -*- coding: utf-8 -*-
"""SegRNN-style conditional baseline for AGC multi-step forecasting."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalSegRNNForecaster(nn.Module):
    """Encode segmented history and decode with known future weather/control."""

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
        seg_len: int = 12,
    ):
        super().__init__()
        self.seg_len = seg_len
        self.past_dim = past_dim
        self.num_segments = math.ceil(seq_len / seg_len)
        self.padded_len = self.num_segments * seg_len

        self.segment_proj = nn.Sequential(
            nn.Linear(seg_len * past_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.future_embed = nn.Sequential(
            nn.Linear(weather_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def _segment_history(self, x_past: torch.Tensor) -> torch.Tensor:
        if self.padded_len > x_past.size(1):
            pad_steps = self.padded_len - x_past.size(1)
            x_past = F.pad(x_past, (0, 0, pad_steps, 0), mode="replicate")
        x_past = x_past.reshape(x_past.size(0), self.num_segments, self.seg_len * self.past_dim)
        return self.segment_proj(x_past)

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        seg_tokens = self._segment_history(x_past)
        _, hidden = self.encoder(seg_tokens)
        context = hidden[-1]
        future_known = torch.cat([w_future, u_future], dim=-1)
        future_feat = self.future_embed(future_known)
        repeated_context = context.unsqueeze(1).expand(-1, future_feat.size(1), -1)
        decoder_in = torch.cat([future_feat, repeated_context], dim=-1)
        decoder_out, _ = self.decoder(decoder_in, hidden)
        return self.head(decoder_out)
