# -*- coding: utf-8 -*-
"""Residual forecaster using DLinear main path and iTransformer residual path."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dlinear_forecaster import ConditionalDLinearForecaster
from models.itransformer_forecaster import ConditionalITransformerForecaster


class _MovingAverage1D(nn.Module):
    """Simple moving-average block used for CO2 trend extraction."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        padded = F.pad(series.unsqueeze(1), (left, right), mode="replicate")
        return self.pool(padded).squeeze(1)


class _CO2SpecialistBranch(nn.Module):
    """CO2-only residual branch with decomposition, recurrence, and dynamic fusion."""

    def __init__(
        self,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        hidden_dim: int,
        dropout: float,
        nhead: int,
        kernel_size: int,
        co2_past_idx: int = 2,
    ):
        super().__init__()
        self.co2_past_idx = co2_past_idx
        self.decomposition = _MovingAverage1D(kernel_size)
        self.trend_encoder = nn.GRU(
            input_size=past_dim + 1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.seasonal_encoder = nn.GRU(
            input_size=past_dim + 1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.future_proj = nn.Sequential(
            nn.Linear(weather_dim + control_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trend_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.seasonal_attention = nn.MultiheadAttention(
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
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        co2_hist = x_past[:, :, self.co2_past_idx]
        trend = self.decomposition(co2_hist)
        seasonal = co2_hist - trend

        trend_input = torch.cat([x_past, trend.unsqueeze(-1)], dim=-1)
        seasonal_input = torch.cat([x_past, seasonal.unsqueeze(-1)], dim=-1)
        trend_memory, _ = self.trend_encoder(trend_input)
        seasonal_memory, _ = self.seasonal_encoder(seasonal_input)

        future_tokens = self.future_proj(torch.cat([w_future, u_future], dim=-1))
        trend_context, _ = self.trend_attention(
            future_tokens,
            trend_memory,
            trend_memory,
            need_weights=False,
        )
        seasonal_context, _ = self.seasonal_attention(
            future_tokens,
            seasonal_memory,
            seasonal_memory,
            need_weights=False,
        )
        mix = self.fusion_gate(torch.cat([future_tokens, trend_context, seasonal_context], dim=-1))
        fused_context = mix * seasonal_context + (1.0 - mix) * trend_context
        residual = self.head(torch.cat([future_tokens, fused_context], dim=-1)).squeeze(-1)
        return self.output_scale * residual


class _CO2LateHorizonAdapter(nn.Module):
    """Lightweight CO2 adapter that emphasizes later-horizon corrections."""

    def __init__(
        self,
        weather_dim: int,
        control_dim: int,
        hidden_dim: int,
        dropout: float,
        kernel_size: int,
        co2_past_idx: int = 2,
    ):
        super().__init__()
        self.co2_past_idx = co2_past_idx
        self.decomposition = _MovingAverage1D(kernel_size)
        self.summary_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(weather_dim + control_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_scale = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        base_co2_pred: torch.Tensor,
    ) -> torch.Tensor:
        co2_hist = x_past[:, :, self.co2_past_idx]
        trend = self.decomposition(co2_hist)
        seasonal = co2_hist - trend
        tail = min(12, co2_hist.size(1))
        recent_raw = co2_hist[:, -tail:]
        recent_trend = trend[:, -tail:]
        recent_seasonal = seasonal[:, -tail:]
        slope = (recent_trend[:, -1] - recent_trend[:, 0]) / max(tail - 1, 1)
        summary_stats = torch.stack(
            [
                co2_hist[:, -1],
                trend[:, -1],
                seasonal[:, -1],
                recent_raw.mean(dim=1),
                recent_seasonal.std(dim=1, unbiased=False),
                slope,
            ],
            dim=-1,
        )
        summary_embed = self.summary_proj(summary_stats)

        horizon = w_future.size(1)
        horizon_ratio = torch.linspace(
            0.0,
            1.0,
            steps=horizon,
            device=w_future.device,
            dtype=w_future.dtype,
        ).view(1, horizon, 1).expand(w_future.size(0), -1, -1)
        token_input = torch.cat(
            [
                w_future,
                u_future,
                base_co2_pred.unsqueeze(-1),
                horizon_ratio,
            ],
            dim=-1,
        )
        token_embed = self.token_proj(token_input)
        summary_expand = summary_embed.unsqueeze(1).expand(-1, horizon, -1)
        fused = self.fusion(torch.cat([token_embed, summary_expand], dim=-1))
        gate = self.gate(torch.cat([token_embed, summary_expand], dim=-1)).squeeze(-1)
        residual = self.head(fused).squeeze(-1)
        return self.output_scale * horizon_ratio.squeeze(-1) * gate * residual


class _CO2WaveletAdapter(nn.Module):
    """Wavelet-inspired multi-scale CO2 adapter for multi-target residual correction."""

    def __init__(
        self,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        hidden_dim: int,
        dropout: float,
        nhead: int,
        co2_past_idx: int = 2,
    ):
        super().__init__()
        self.co2_past_idx = co2_past_idx
        self.fast_ma = _MovingAverage1D(7)
        self.mid_ma = _MovingAverage1D(19)
        self.slow_ma = _MovingAverage1D(41)
        branch_dim = past_dim + 1
        self.low_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.mid_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.high_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.future_proj = nn.Sequential(
            nn.Linear(weather_dim + control_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.low_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.mid_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.high_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.band_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_scale = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        base_co2_pred: torch.Tensor,
    ) -> torch.Tensor:
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
        token_input = torch.cat(
            [
                w_future,
                u_future,
                base_co2_pred.unsqueeze(-1),
                horizon_ratio,
            ],
            dim=-1,
        )
        future_tokens = self.future_proj(token_input)
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
        residual_gate = self.gate(torch.cat([future_tokens, fused], dim=-1)).squeeze(-1)
        residual = self.head(torch.cat([future_tokens, fused], dim=-1)).squeeze(-1)
        return self.output_scale * residual_gate * residual


class _CO2WaveletForecastExpert(nn.Module):
    """Standalone-style wavelet CO2 expert for absolute forecast fusion."""

    def __init__(
        self,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        hidden_dim: int,
        dropout: float,
        nhead: int,
        co2_past_idx: int = 2,
    ):
        super().__init__()
        self.co2_past_idx = co2_past_idx
        self.fast_ma = _MovingAverage1D(7)
        self.mid_ma = _MovingAverage1D(19)
        self.slow_ma = _MovingAverage1D(41)
        branch_dim = past_dim + 1
        self.low_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.mid_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.high_encoder = nn.GRU(
            input_size=branch_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.future_proj = nn.Sequential(
            nn.Linear(weather_dim + control_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.low_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.mid_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.high_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
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
        base = x_past[:, -1:, self.co2_past_idx : self.co2_past_idx + 1].expand(-1, horizon, -1)
        return base.squeeze(-1) + self.head(torch.cat([future_tokens, fused], dim=-1)).squeeze(-1)


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


class ConditionalITransformerCO2ResidualForecaster(nn.Module):
    """Augment iTransformer residual with a CO2-only specialist correction branch."""

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
        co2_target_idx: int = 2,
    ):
        super().__init__()
        self.co2_target_idx = min(co2_target_idx, target_dim - 1)
        self.main_model = ConditionalITransformerResidualForecaster(
            seq_len=seq_len,
            horizon=horizon,
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            ff_dim=ff_dim,
            kernel_size=kernel_size,
        )
        self.co2_branch = _CO2SpecialistBranch(
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nhead=nhead,
            kernel_size=kernel_size,
            co2_past_idx=min(2, past_dim - 1),
        )
        self.co2_gate = nn.Sequential(
            nn.Linear(past_dim + weather_dim + control_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        co2_residual = self.co2_branch(x_past, w_future, u_future)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, prediction.size(1), -1)
        gate_input = torch.cat([past_summary, w_future, u_future], dim=-1)
        co2_gate = self.co2_gate(gate_input).squeeze(-1)
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = prediction[..., self.co2_target_idx] + co2_gate * co2_residual
        return prediction


class ConditionalITransformerCO2LateResidualForecaster(nn.Module):
    """Use a lightweight late-horizon CO2 adapter on top of the iTransformer residual model."""

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
        co2_target_idx: int = 2,
    ):
        super().__init__()
        self.co2_target_idx = min(co2_target_idx, target_dim - 1)
        self.main_model = ConditionalITransformerResidualForecaster(
            seq_len=seq_len,
            horizon=horizon,
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            ff_dim=ff_dim,
            kernel_size=kernel_size,
        )
        self.co2_adapter = _CO2LateHorizonAdapter(
            weather_dim=weather_dim,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            co2_past_idx=min(2, past_dim - 1),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        co2_base = prediction[..., self.co2_target_idx]
        co2_adjustment = self.co2_adapter(x_past, w_future, u_future, co2_base)
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = prediction[..., self.co2_target_idx] + co2_adjustment
        return prediction


class ConditionalITransformerCO2WaveletResidualForecaster(nn.Module):
    """Use a wavelet-inspired multi-scale CO2 adapter on top of the iTransformer residual model."""

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
        co2_target_idx: int = 2,
    ):
        super().__init__()
        self.co2_target_idx = min(co2_target_idx, target_dim - 1)
        self.main_model = ConditionalITransformerResidualForecaster(
            seq_len=seq_len,
            horizon=horizon,
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            ff_dim=ff_dim,
            kernel_size=kernel_size,
        )
        self.co2_adapter = _CO2WaveletAdapter(
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nhead=nhead,
            co2_past_idx=min(2, past_dim - 1),
        )
        self.co2_gate = nn.Sequential(
            nn.Linear(past_dim + weather_dim + control_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        co2_base = prediction[..., self.co2_target_idx]
        co2_adjustment = self.co2_adapter(x_past, w_future, u_future, co2_base)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, prediction.size(1), -1)
        gate_input = torch.cat([past_summary, w_future, u_future], dim=-1)
        co2_gate = self.co2_gate(gate_input).squeeze(-1)
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = prediction[..., self.co2_target_idx] + co2_gate * co2_adjustment
        return prediction


class ConditionalITransformerCO2WaveletBlendForecaster(nn.Module):
    """Blend the multi-target forecaster CO2 output with a standalone-style wavelet CO2 expert."""

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
        co2_target_idx: int = 2,
    ):
        super().__init__()
        self.co2_target_idx = min(co2_target_idx, target_dim - 1)
        self.main_model = ConditionalITransformerResidualForecaster(
            seq_len=seq_len,
            horizon=horizon,
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            ff_dim=ff_dim,
            kernel_size=kernel_size,
        )
        self.co2_expert = _CO2WaveletForecastExpert(
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nhead=nhead,
            co2_past_idx=min(2, past_dim - 1),
        )
        self.blend_gate = nn.Sequential(
            nn.Linear(past_dim + weather_dim + control_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        main_co2 = prediction[..., self.co2_target_idx]
        expert_co2 = self.co2_expert(x_past, w_future, u_future)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, prediction.size(1), -1)
        blend_input = torch.cat([past_summary, w_future, u_future, main_co2.unsqueeze(-1)], dim=-1)
        blend = self.blend_gate(blend_input).squeeze(-1)
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = (1.0 - blend) * main_co2 + blend * expert_co2
        return prediction
