# -*- coding: utf-8 -*-
"""Frequency-style conditional baseline for AGC multi-step forecasting."""

import torch
import torch.nn as nn


class ConditionalFrequencyMLPForecaster(nn.Module):
    """Use low-frequency history modes as a compact global context."""

    def __init__(
        self,
        seq_len: int,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
        num_modes: int = 16,
    ):
        super().__init__()
        self.num_modes = min(num_modes, seq_len // 2 + 1)
        freq_dim = self.num_modes * past_dim * 2
        self.freq_encoder = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.last_state_head = nn.Linear(past_dim, hidden_dim)
        self.future_head = nn.Sequential(
            nn.Linear(weather_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        centered = x_past - x_past.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=1)[:, : self.num_modes, :]
        freq_features = torch.cat([spectrum.real, spectrum.imag], dim=-1).flatten(start_dim=1)
        context = self.freq_encoder(freq_features) + self.last_state_head(x_past[:, -1, :])
        future_feat = self.future_head(torch.cat([w_future, u_future], dim=-1))
        return self.out_head(future_feat + context.unsqueeze(1))
