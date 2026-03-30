# -*- coding: utf-8 -*-
"""Residual forecaster using DLinear main path and iTransformer residual path."""

import torch
import torch.nn as nn

from models.dlinear_forecaster import ConditionalDLinearForecaster
from models.itransformer_forecaster import ConditionalITransformerForecaster


class ConditionalITransformerResidualForecaster(nn.Module):
    """Use DLinear as the main path and iTransformer as a residual corrector."""

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        ff_dim: int = 192,
        kernel_size: int = 25,
    ):
        super().__init__()
        self.base_model = ConditionalDLinearForecaster(
            seq_len=seq_len,
            horizon=horizon,
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        )
        self.residual_model = ConditionalITransformerForecaster(
            seq_len=seq_len,
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            ff_dim=ff_dim,
        )
        self.gate = nn.Sequential(
            nn.Linear(past_dim + weather_dim + control_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.full((target_dim,), 0.1))

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        base_pred = self.base_model(x_past, w_future, u_future)
        residual_pred = self.residual_model(x_past, w_future, u_future)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, base_pred.size(1), -1)
        gate_input = torch.cat([past_summary, w_future, u_future], dim=-1)
        gate = self.gate(gate_input)
        scaled_residual = gate * self.residual_scale.view(1, 1, -1) * residual_pred
        return base_pred + scaled_residual
