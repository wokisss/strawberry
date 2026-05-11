# -*- coding: utf-8 -*-
"""Conditional LSTM baseline for AGC multi-step forecasting."""

import torch
import torch.nn as nn


class ConditionalLSTMForecaster(nn.Module):
    """Encode past states and decode future targets conditioned on known future inputs."""

    def __init__(
        self,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=past_dim,
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
        self.decoder = nn.LSTM(
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

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(x_past)
        context = hidden[0][-1]
        future_known = torch.cat([w_future, u_future], dim=-1)
        future_feat = self.future_embed(future_known)
        repeated_context = context.unsqueeze(1).expand(-1, future_feat.size(1), -1)
        decoder_in = torch.cat([future_feat, repeated_context], dim=-1)
        decoder_out, _ = self.decoder(decoder_in, hidden)
        return self.head(decoder_out)
