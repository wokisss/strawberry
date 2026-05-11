# -*- coding: utf-8 -*-
"""Conditional NLinear-style baseline for AGC multi-step forecasting."""

import torch
import torch.nn as nn


class ConditionalNLinearForecaster(nn.Module):
    """Normalize by the last observed state, then linearly project the history."""

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.temporal_linear = nn.Linear(seq_len, horizon)
        self.past_head = nn.Linear(past_dim, hidden_dim)
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
        last = x_past[:, -1:, :]
        centered = x_past - last
        projected = self.temporal_linear(centered.transpose(1, 2)).transpose(1, 2)
        past_proj = self.past_head(projected + last)
        future_proj = self.future_head(torch.cat([w_future, u_future], dim=-1))
        return self.out_head(past_proj + future_proj)
