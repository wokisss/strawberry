# -*- coding: utf-8 -*-
"""Standalone CO2 specialist forecasters inspired by greenhouse CO2 papers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MovingAverage1D(nn.Module):
    """Replicated-edge moving average for 1D series."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        padded = F.pad(series.unsqueeze(1), (left, right), mode="replicate")
        return self.pool(padded).squeeze(1)


class ConditionalCO2LSTMForecaster(nn.Module):
    """Plain CO2-only LSTM conditioned on future weather and control."""

    def __init__(
        self,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        co2_past_idx: int = 2,
    ):
        super().__init__()
        if target_dim != 1:
            raise ValueError("ConditionalCO2LSTMForecaster expects target_dim=1.")
        self.co2_past_idx = co2_past_idx
        self.encoder = nn.LSTM(
            input_size=past_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.future_embed = nn.Sequential(
            nn.Linear(weather_dim + control_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(x_past)
        decoder_in = self.future_embed(torch.cat([w_future, u_future], dim=-1))
        decoded, _ = self.decoder(decoder_in, (hidden, cell))
        base = x_past[:, -1:, self.co2_past_idx : self.co2_past_idx + 1].expand(-1, w_future.size(1), -1)
        return base + self.head(decoded)


class ConditionalCO2VMDLSTMFusionForecaster(nn.Module):
    """VMD/WT-inspired decomposition plus LSTM-attention fusion for CO2."""

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
        co2_past_idx: int = 2,
    ):
        super().__init__()
        if target_dim != 1:
            raise ValueError("ConditionalCO2VMDLSTMFusionForecaster expects target_dim=1.")
        self.co2_past_idx = co2_past_idx
        self.fast_ma = _MovingAverage1D(9)
        self.slow_ma = _MovingAverage1D(33)
        branch_dim = past_dim + 1
        self.trend_encoder = nn.LSTM(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.detail_encoder = nn.LSTM(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.future_proj = nn.Sequential(
            nn.Linear(weather_dim + control_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trend_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.detail_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        co2_hist = x_past[:, :, self.co2_past_idx]
        smooth = self.fast_ma(co2_hist)
        trend = self.slow_ma(co2_hist)
        detail = co2_hist - smooth

        trend_input = torch.cat([x_past, trend.unsqueeze(-1)], dim=-1)
        detail_input = torch.cat([x_past, detail.unsqueeze(-1)], dim=-1)
        trend_memory, _ = self.trend_encoder(trend_input)
        detail_memory, _ = self.detail_encoder(detail_input)

        future_tokens = self.future_proj(torch.cat([w_future, u_future], dim=-1))
        trend_ctx, _ = self.trend_attn(future_tokens, trend_memory, trend_memory, need_weights=False)
        detail_ctx, _ = self.detail_attn(future_tokens, detail_memory, detail_memory, need_weights=False)
        gate = self.fusion_gate(torch.cat([future_tokens, trend_ctx, detail_ctx], dim=-1))
        fused = gate * detail_ctx + (1.0 - gate) * trend_ctx
        base = x_past[:, -1:, self.co2_past_idx : self.co2_past_idx + 1].expand(-1, w_future.size(1), -1)
        return base + self.head(torch.cat([future_tokens, fused], dim=-1))


class ConditionalCO2WaveletGRUAttnForecaster(nn.Module):
    """Wavelet-inspired multi-scale GRU with adaptive attention for CO2."""

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
        co2_past_idx: int = 2,
    ):
        super().__init__()
        if target_dim != 1:
            raise ValueError("ConditionalCO2WaveletGRUAttnForecaster expects target_dim=1.")
        self.co2_past_idx = co2_past_idx
        self.fast_ma = _MovingAverage1D(7)
        self.mid_ma = _MovingAverage1D(19)
        self.slow_ma = _MovingAverage1D(41)
        branch_dim = past_dim + 1
        self.low_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mid_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.high_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.future_proj = nn.Sequential(
            nn.Linear(weather_dim + control_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.low_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.mid_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.high_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.band_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        co2_hist = x_past[:, :, self.co2_past_idx]
        low_band = self.slow_ma(co2_hist)
        mid_band = self.mid_ma(co2_hist) - low_band
        high_band = co2_hist - self.fast_ma(co2_hist)

        low_memory, _ = self.low_encoder(torch.cat([x_past, low_band.unsqueeze(-1)], dim=-1))
        mid_memory, _ = self.mid_encoder(torch.cat([x_past, mid_band.unsqueeze(-1)], dim=-1))
        high_memory, _ = self.high_encoder(torch.cat([x_past, high_band.unsqueeze(-1)], dim=-1))

        horizon = w_future.size(1)
        horizon_ratio = torch.linspace(
            0.0,
            1.0,
            steps=horizon,
            device=w_future.device,
            dtype=w_future.dtype,
        ).view(1, horizon, 1).expand(w_future.size(0), -1, -1)
        future_tokens = self.future_proj(torch.cat([w_future, u_future, horizon_ratio], dim=-1))
        low_ctx, _ = self.low_attn(future_tokens, low_memory, low_memory, need_weights=False)
        mid_ctx, _ = self.mid_attn(future_tokens, mid_memory, mid_memory, need_weights=False)
        high_ctx, _ = self.high_attn(future_tokens, high_memory, high_memory, need_weights=False)

        gate_logits = self.band_gate(torch.cat([future_tokens, low_ctx, mid_ctx, high_ctx], dim=-1))
        band_weights = torch.softmax(gate_logits, dim=-1)
        fused = (
            band_weights[..., 0:1] * low_ctx
            + band_weights[..., 1:2] * mid_ctx
            + band_weights[..., 2:3] * high_ctx
        )
        base = x_past[:, -1:, self.co2_past_idx : self.co2_past_idx + 1].expand(-1, w_future.size(1), -1)
        return base + self.head(torch.cat([future_tokens, fused], dim=-1))
