# -*- coding: utf-8 -*-
"""Conditional DLinear-style baseline for AGC multi-step forecasting."""

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    """Moving average block used by DLinear-style decomposition."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.transpose(1, 2)
        x = self.avg(x)
        return x.transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """Split input sequence into residual and trend components."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class ConditionalDLinearForecaster(nn.Module):
    """DLinear-style temporal projection with future-known conditioning."""

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
        kernel_size: int = 25,
    ):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.seasonal_linear = nn.Linear(seq_len, horizon)
        self.trend_linear = nn.Linear(seq_len, horizon)
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
        seasonal, trend = self.decomp(x_past)
        seasonal = seasonal.transpose(1, 2)  # [B, C, L]
        trend = trend.transpose(1, 2)        # [B, C, L]
        seasonal_out = self.seasonal_linear(seasonal).transpose(1, 2)  # [B, H, C]
        trend_out = self.trend_linear(trend).transpose(1, 2)            # [B, H, C]
        past_proj = self.past_head(seasonal_out + trend_out)
        future_proj = self.future_head(torch.cat([w_future, u_future], dim=-1))
        return self.out_head(past_proj + future_proj)

