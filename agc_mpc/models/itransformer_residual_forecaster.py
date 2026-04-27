# -*- coding: utf-8 -*-
"""Residual forecaster using DLinear main path and iTransformer residual path."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.co2_specialist_forecasters import ConditionalCO2WaveletGRUAttnForecaster
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


class ConditionalITransformerCO2FrozenExpertForecaster(nn.Module):
    """Blend a multi-target backbone with a frozen standalone CO2 wavelet expert."""

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
        self.co2_expert = ConditionalCO2WaveletGRUAttnForecaster(
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            co2_past_idx=min(2, past_dim - 1),
        )
        self._freeze_expert()
        gate_in_dim = past_dim + weather_dim + control_dim + target_dim + 2
        self.blend_gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Start from the backbone and let the gate earn trust in the frozen expert.
        nn.init.constant_(self.blend_gate[-2].bias, -2.0)

    def _freeze_expert(self) -> None:
        for parameter in self.co2_expert.parameters():
            parameter.requires_grad_(False)
        self.co2_expert.eval()

    def load_frozen_expert_checkpoint(self, checkpoint_path: str, map_location=None) -> None:
        state = torch.load(checkpoint_path, map_location=map_location)
        self.co2_expert.load_state_dict(state)
        self._freeze_expert()

    def train(self, mode: bool = True):
        super().train(mode)
        self.co2_expert.eval()
        return self

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        main_co2 = prediction[..., self.co2_target_idx]
        with torch.no_grad():
            expert_co2 = self.co2_expert(x_past, w_future, u_future).squeeze(-1)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, prediction.size(1), -1)
        gate_input = torch.cat(
            [
                past_summary,
                w_future,
                u_future,
                prediction,
                (expert_co2 - main_co2).unsqueeze(-1),
                expert_co2.unsqueeze(-1),
            ],
            dim=-1,
        )
        blend = self.blend_gate(gate_input).squeeze(-1)
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = main_co2 + blend * (expert_co2 - main_co2)
        return prediction


class ConditionalITransformerCO2LateFrozenExpertForecaster(nn.Module):
    """Use a frozen CO2 expert, but only trust it progressively toward later horizons."""

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
        self.co2_expert = ConditionalCO2WaveletGRUAttnForecaster(
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            co2_past_idx=min(2, past_dim - 1),
        )
        self._freeze_expert()
        gate_in_dim = past_dim + weather_dim + control_dim + target_dim + 2
        self.blend_gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.blend_gate[-2].bias, -2.2)
        self.late_power = 2.0

    def _freeze_expert(self) -> None:
        for parameter in self.co2_expert.parameters():
            parameter.requires_grad_(False)
        self.co2_expert.eval()

    def load_frozen_expert_checkpoint(self, checkpoint_path: str, map_location=None) -> None:
        state = torch.load(checkpoint_path, map_location=map_location)
        self.co2_expert.load_state_dict(state)
        self._freeze_expert()

    def train(self, mode: bool = True):
        super().train(mode)
        self.co2_expert.eval()
        return self

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        main_co2 = prediction[..., self.co2_target_idx]
        with torch.no_grad():
            expert_co2 = self.co2_expert(x_past, w_future, u_future).squeeze(-1)
        horizon_ratio = torch.linspace(
            0.0,
            1.0,
            steps=prediction.size(1),
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, -1)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, prediction.size(1), -1)
        gate_input = torch.cat(
            [
                past_summary,
                w_future,
                u_future,
                prediction,
                (expert_co2 - main_co2).unsqueeze(-1),
                expert_co2.unsqueeze(-1),
            ],
            dim=-1,
        )
        base_gate = self.blend_gate(gate_input).squeeze(-1)
        late_gate = base_gate * horizon_ratio.pow(self.late_power)
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = main_co2 + late_gate * (expert_co2 - main_co2)
        return prediction


class ConditionalITransformerCO2TeacherDistillForecaster(nn.Module):
    """Student forecaster trained with a frozen standalone CO2 teacher."""

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
        self.student = ConditionalITransformerResidualForecaster(
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
        self.teacher = ConditionalCO2WaveletGRUAttnForecaster(
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            co2_past_idx=min(2, past_dim - 1),
        )
        self._freeze_teacher()
        self.late_power = 2.0

    def _freeze_teacher(self) -> None:
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    def load_frozen_expert_checkpoint(self, checkpoint_path: str, map_location=None) -> None:
        state = torch.load(checkpoint_path, map_location=map_location)
        self.teacher.load_state_dict(state)
        self._freeze_teacher()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def compute_auxiliary_loss(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        y_true: torch.Tensor,
        prediction: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_co2 = self.teacher(x_past, w_future, u_future).squeeze(-1)
        student_co2 = prediction[..., self.co2_target_idx]
        horizon_ratio = torch.linspace(
            0.0,
            1.0,
            steps=prediction.size(1),
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, -1)
        weights = 0.35 + 0.65 * horizon_ratio.pow(self.late_power)
        return ((student_co2 - teacher_co2) ** 2 * weights).mean()

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        return self.student(x_past, w_future, u_future)


class ConditionalITransformerCO2RecoupledExpertForecaster(nn.Module):
    """Late frozen CO2 expert with explicit variable recoupling and confidence-guided distillation."""

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
        self.target_dim = target_dim
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
        self.co2_expert = ConditionalCO2WaveletGRUAttnForecaster(
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            co2_past_idx=min(2, past_dim - 1),
        )
        self._freeze_expert()
        self.context_proj = nn.Sequential(
            nn.Linear(weather_dim + control_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.co2_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.co2_recoupling = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.recoupling_scale = nn.Parameter(torch.tensor(0.05))
        self.late_power = 2.0
        self.teacher_scale = 60.0

    def _freeze_expert(self) -> None:
        for parameter in self.co2_expert.parameters():
            parameter.requires_grad_(False)
        self.co2_expert.eval()

    def load_frozen_expert_checkpoint(self, checkpoint_path: str, map_location=None) -> None:
        state = torch.load(checkpoint_path, map_location=map_location)
        self.co2_expert.load_state_dict(state)
        self._freeze_expert()

    def train(self, mode: bool = True):
        super().train(mode)
        self.co2_expert.eval()
        return self

    def _horizon_ratio(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.linspace(
            0.0,
            1.0,
            steps=prediction.size(1),
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, -1)

    def _recouple(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        main_pred: torch.Tensor,
        expert_co2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        horizon_ratio = self._horizon_ratio(main_pred)
        main_co2 = main_pred[..., self.co2_target_idx]
        delta = expert_co2 - main_co2
        context = self.context_proj(
            torch.cat(
                [
                    w_future,
                    u_future,
                    delta.unsqueeze(-1),
                    horizon_ratio.unsqueeze(-1).expand(main_pred.size(0), -1, -1),
                ],
                dim=-1,
            )
        )
        target_past = x_past[:, :, : self.target_dim]
        past_last = target_past[:, -1, :]
        past_mean = target_past.mean(dim=1)
        node_features = torch.stack(
            [
                main_pred,
                past_last.unsqueeze(1).expand(-1, main_pred.size(1), -1),
                past_mean.unsqueeze(1).expand(-1, main_pred.size(1), -1),
            ],
            dim=-1,
        )
        node_embed = self.node_proj(node_features) + context.unsqueeze(2)
        q = self.q_proj(node_embed)
        k = self.k_proj(node_embed)
        v = self.v_proj(node_embed)
        attn_scores = torch.einsum("bthd,btkd->bthk", q, k) / (q.size(-1) ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)
        mixed = torch.einsum("bthk,btkd->bthd", attn, v)
        co2_node = mixed[:, :, self.co2_target_idx, :]
        gate_input = torch.cat(
            [
                co2_node,
                context,
                main_co2.unsqueeze(-1),
                expert_co2.unsqueeze(-1),
            ],
            dim=-1,
        )
        late_gate = self.co2_gate(gate_input).squeeze(-1) * horizon_ratio.pow(self.late_power)
        recoupling = self.recoupling_scale * self.co2_recoupling(torch.cat([co2_node, context], dim=-1)).squeeze(-1)
        return late_gate, recoupling

    def compute_auxiliary_loss(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        y_true: torch.Tensor,
        prediction: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_co2 = self.co2_expert(x_past, w_future, u_future).squeeze(-1)
        student_co2 = prediction[..., self.co2_target_idx]
        true_co2 = y_true[..., self.co2_target_idx]
        horizon_ratio = self._horizon_ratio(prediction)
        teacher_err = torch.abs(teacher_co2 - true_co2)
        student_err = torch.abs(student_co2 - true_co2)
        teacher_conf = torch.exp(-teacher_err / self.teacher_scale)
        teacher_advantage = torch.relu(student_err - teacher_err)
        teacher_advantage = teacher_advantage / (teacher_advantage.mean().detach() + 1e-6)
        weights = (0.25 + 0.75 * horizon_ratio.pow(self.late_power)) * teacher_conf * teacher_advantage.clamp(max=3.0)
        return ((student_co2 - teacher_co2) ** 2 * weights.detach()).mean()

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        main_co2 = prediction[..., self.co2_target_idx]
        with torch.no_grad():
            expert_co2 = self.co2_expert(x_past, w_future, u_future).squeeze(-1)
        late_gate, recoupling = self._recouple(x_past, w_future, u_future, prediction, expert_co2)
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = main_co2 + late_gate * (expert_co2 - main_co2) + recoupling
        return prediction


class ConditionalITransformerCO2ProtectedExpertForecaster(nn.Module):
    """Use the strong late-residual model as the main predictor and apply a cautious frozen-expert correction."""

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
        self.main_model = ConditionalITransformerCO2LateResidualForecaster(
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
            co2_target_idx=self.co2_target_idx,
        )
        self.co2_expert = ConditionalCO2WaveletGRUAttnForecaster(
            past_dim=past_dim,
            weather_dim=weather_dim,
            control_dim=control_dim,
            target_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            co2_past_idx=min(2, past_dim - 1),
        )
        self._freeze_expert()
        gate_in_dim = past_dim + weather_dim + control_dim + target_dim + 2
        self.blend_gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.blend_gate[-2].bias, -2.8)
        self.delta_temperature = nn.Parameter(torch.tensor(45.0))
        self.late_power = 2.0

    def _freeze_expert(self) -> None:
        for parameter in self.co2_expert.parameters():
            parameter.requires_grad_(False)
        self.co2_expert.eval()

    def load_frozen_expert_checkpoint(self, checkpoint_path: str, map_location=None) -> None:
        state = torch.load(checkpoint_path, map_location=map_location)
        self.co2_expert.load_state_dict(state)
        self._freeze_expert()

    def train(self, mode: bool = True):
        super().train(mode)
        self.co2_expert.eval()
        return self

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        main_co2 = prediction[..., self.co2_target_idx]
        with torch.no_grad():
            expert_co2 = self.co2_expert(x_past, w_future, u_future).squeeze(-1)

        horizon_ratio = torch.linspace(
            0.0,
            1.0,
            steps=prediction.size(1),
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, -1)
        delta = expert_co2 - main_co2
        delta_scale = torch.clamp(self.delta_temperature.abs(), min=10.0)
        agreement = torch.exp(-torch.abs(delta) / delta_scale)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, prediction.size(1), -1)
        gate_input = torch.cat(
            [
                past_summary,
                w_future,
                u_future,
                prediction,
                delta.unsqueeze(-1),
                expert_co2.unsqueeze(-1),
            ],
            dim=-1,
        )
        learned_gate = self.blend_gate(gate_input).squeeze(-1)
        late_gate = learned_gate * agreement * horizon_ratio.pow(self.late_power)
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = main_co2 + late_gate * delta
        return prediction


class ConditionalITransformerCO2ProtectedTerminalForecaster(ConditionalITransformerCO2ProtectedExpertForecaster):
    """Protected expert variant with explicit late/terminal CO2 training pressure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_power = 3.0

    def compute_auxiliary_loss(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        y_true: torch.Tensor,
        prediction: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        pred_co2 = prediction[..., self.co2_target_idx]
        true_co2 = y_true[..., self.co2_target_idx]
        horizon_ratio = torch.linspace(
            0.0,
            1.0,
            steps=prediction.size(1),
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, -1)
        weights = 0.2 + 0.8 * horizon_ratio.pow(self.terminal_power)
        weighted_mse = ((pred_co2 - true_co2) ** 2 * weights).mean()
        final_mse = ((pred_co2[:, -1] - true_co2[:, -1]) ** 2).mean()
        return weighted_mse + final_mse


class ConditionalITransformerCO2HorizonMixtureForecaster(ConditionalITransformerCO2ProtectedExpertForecaster):
    """Use protected expert correction early, then pull terminal CO2 back toward late-residual."""

    def __init__(
        self,
        *args,
        terminal_start: float = 0.72,
        terminal_power: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.terminal_start = terminal_start
        self.terminal_power = terminal_power
        self.auxiliary_power = 3.0

    def _horizon_ratio(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.linspace(
            0.0,
            1.0,
            steps=prediction.size(1),
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, -1)

    def _terminal_pullback(self, prediction: torch.Tensor) -> torch.Tensor:
        horizon_ratio = self._horizon_ratio(prediction)
        denominator = max(1.0 - self.terminal_start, 1e-6)
        terminal_ratio = torch.clamp((horizon_ratio - self.terminal_start) / denominator, min=0.0, max=1.0)
        return terminal_ratio.pow(self.terminal_power)

    def compute_auxiliary_loss(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        y_true: torch.Tensor,
        prediction: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        pred_co2 = prediction[..., self.co2_target_idx]
        true_co2 = y_true[..., self.co2_target_idx]
        horizon_ratio = self._horizon_ratio(prediction)
        weights = 0.15 + 0.85 * horizon_ratio.pow(self.auxiliary_power)
        weighted_mse = ((pred_co2 - true_co2) ** 2 * weights).mean()
        final_mse = ((pred_co2[:, -1] - true_co2[:, -1]) ** 2).mean()
        return 0.5 * weighted_mse + final_mse

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        main_co2 = prediction[..., self.co2_target_idx]
        with torch.no_grad():
            expert_co2 = self.co2_expert(x_past, w_future, u_future).squeeze(-1)

        horizon_ratio = self._horizon_ratio(prediction)
        terminal_pullback = self._terminal_pullback(prediction)
        delta = expert_co2 - main_co2
        delta_scale = torch.clamp(self.delta_temperature.abs(), min=10.0)
        agreement = torch.exp(-torch.abs(delta) / delta_scale)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, prediction.size(1), -1)
        gate_input = torch.cat(
            [
                past_summary,
                w_future,
                u_future,
                prediction,
                delta.unsqueeze(-1),
                expert_co2.unsqueeze(-1),
            ],
            dim=-1,
        )
        learned_gate = self.blend_gate(gate_input).squeeze(-1)
        protected_gate = learned_gate * agreement * horizon_ratio.pow(self.late_power)
        correction = protected_gate * (1.0 - terminal_pullback) * delta
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = main_co2 + correction
        return prediction


class ConditionalITransformerCO2FrozenBackboneHorizonMixtureForecaster(
    ConditionalITransformerCO2HorizonMixtureForecaster
):
    """Train only the horizon gate while keeping the late-residual backbone fixed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._freeze_main_model()

    def _freeze_main_model(self) -> None:
        for parameter in self.main_model.parameters():
            parameter.requires_grad_(False)
        self.main_model.eval()

    def load_main_checkpoint(self, checkpoint_path: str, map_location=None) -> None:
        state = torch.load(checkpoint_path, map_location=map_location)
        self.main_model.load_state_dict(state)
        self._freeze_main_model()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.main_model.eval()
        self.co2_expert.eval()
        return self

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        prediction = self.main_model(x_past, w_future, u_future)
        expert_co2 = self.co2_expert(x_past, w_future, u_future).squeeze(-1)
        main_co2 = prediction[..., self.co2_target_idx]

        horizon_ratio = self._horizon_ratio(prediction)
        terminal_pullback = self._terminal_pullback(prediction)
        delta = expert_co2 - main_co2
        delta_scale = torch.clamp(self.delta_temperature.abs(), min=10.0)
        agreement = torch.exp(-torch.abs(delta) / delta_scale)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, prediction.size(1), -1)
        gate_input = torch.cat(
            [
                past_summary,
                w_future,
                u_future,
                prediction,
                delta.unsqueeze(-1),
                expert_co2.unsqueeze(-1),
            ],
            dim=-1,
        )
        learned_gate = self.blend_gate(gate_input).squeeze(-1)
        protected_gate = learned_gate * agreement * horizon_ratio.pow(self.late_power)
        correction = protected_gate * (1.0 - terminal_pullback) * delta
        prediction = prediction.clone()
        prediction[..., self.co2_target_idx] = main_co2 + correction
        return prediction


class ConditionalITransformerCO2ControlAwareFusionForecaster(nn.Module):
    """Fuse late-frozen control behavior with horizon-mixture terminal gains."""

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
        co2_control_idx: int = 2,
        control_horizon: int = 6,
        blend_start: float = 0.5,
        blend_power: float = 2.0,
        gate_ceiling: float = 1.0,
        delta_smoothing_kernel: int = 5,
    ):
        super().__init__()
        self.co2_target_idx = min(co2_target_idx, target_dim - 1)
        self.co2_control_idx = min(co2_control_idx, control_dim - 1)
        self.control_horizon = control_horizon
        self.blend_power = blend_power
        self.blend_start = max(blend_start, control_horizon / max(horizon - 1, 1))
        self.gate_ceiling = gate_ceiling
        self.delta_smoothing_kernel = max(1, delta_smoothing_kernel)
        self.base_model = ConditionalITransformerCO2LateFrozenExpertForecaster(
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
            co2_target_idx=self.co2_target_idx,
        )
        self.terminal_model = ConditionalITransformerCO2HorizonMixtureForecaster(
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
            co2_target_idx=self.co2_target_idx,
        )
        gate_in_dim = past_dim + weather_dim + control_dim + target_dim + 4
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.fusion_gate[-2].bias, -3.2)
        self.delta_temperature = nn.Parameter(torch.tensor(35.0))
        self.first_step_weight = 1.0
        self.control_horizon_weight = 1.0
        self.gradient_weight = 0.5
        self._freeze_anchors()

    def _freeze_anchors(self) -> None:
        for module in (self.base_model, self.terminal_model):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

    def load_base_checkpoint(self, checkpoint_path: str, map_location=None) -> None:
        state = torch.load(checkpoint_path, map_location=map_location)
        self.base_model.load_state_dict(state)
        self._freeze_anchors()

    def load_terminal_checkpoint(self, checkpoint_path: str, map_location=None) -> None:
        state = torch.load(checkpoint_path, map_location=map_location)
        self.terminal_model.load_state_dict(state)
        self._freeze_anchors()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.base_model.eval()
        self.terminal_model.eval()
        return self

    def _horizon_ratio(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.linspace(
            0.0,
            1.0,
            steps=prediction.size(1),
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, -1)

    def _late_profile(self, prediction: torch.Tensor) -> torch.Tensor:
        horizon_ratio = self._horizon_ratio(prediction)
        denominator = max(1.0 - self.blend_start, 1e-6)
        shifted = torch.clamp((horizon_ratio - self.blend_start) / denominator, min=0.0, max=1.0)
        return shifted.pow(self.blend_power)

    def _select_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self.delta_smoothing_kernel <= 1 or delta.size(1) <= 2:
            return delta
        left = (self.delta_smoothing_kernel - 1) // 2
        right = self.delta_smoothing_kernel - 1 - left
        padded = F.pad(delta.unsqueeze(1), (left, right), mode="replicate")
        smoothed = F.avg_pool1d(padded, kernel_size=self.delta_smoothing_kernel, stride=1).squeeze(1)
        selected = delta.clone()
        if selected.size(1) > self.control_horizon:
            selected[:, self.control_horizon :] = smoothed[:, self.control_horizon :]
        return selected

    def _co2_fusion(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        base_prediction: torch.Tensor,
        terminal_prediction: torch.Tensor,
    ) -> torch.Tensor:
        base_co2 = base_prediction[..., self.co2_target_idx]
        terminal_co2 = terminal_prediction[..., self.co2_target_idx]
        delta = self._select_delta(terminal_co2 - base_co2)
        late_profile = self._late_profile(base_prediction)
        late_profile_expand = late_profile.unsqueeze(-1).expand(base_prediction.size(0), -1, -1)
        delta_scale = torch.clamp(self.delta_temperature.abs(), min=10.0)
        agreement = torch.exp(-torch.abs(delta) / delta_scale)
        past_summary = x_past.mean(dim=1, keepdim=True).expand(-1, base_prediction.size(1), -1)
        gate_input = torch.cat(
            [
                past_summary,
                w_future,
                u_future,
                base_prediction,
                base_co2.unsqueeze(-1),
                terminal_co2.unsqueeze(-1),
                delta.unsqueeze(-1),
                late_profile_expand,
            ],
            dim=-1,
        )
        learned_gate = self.fusion_gate(gate_input).squeeze(-1)
        return self.gate_ceiling * learned_gate * agreement * late_profile

    def _gradient_match_loss(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
    ) -> torch.Tensor:
        if not u_future.requires_grad:
            u_future = u_future.detach().clone().requires_grad_(True)
        base_prediction = self.base_model(x_past, w_future, u_future)
        terminal_prediction = self.terminal_model(x_past, w_future, u_future)
        fusion_gate = self._co2_fusion(x_past, w_future, u_future, base_prediction, terminal_prediction)
        student_co2 = base_prediction[..., self.co2_target_idx] + fusion_gate * (
            terminal_prediction[..., self.co2_target_idx] - base_prediction[..., self.co2_target_idx]
        )
        anchor_co2 = base_prediction[..., self.co2_target_idx]
        control_horizon = min(self.control_horizon, student_co2.size(1))
        student_signal = student_co2[:, :control_horizon].mean()
        anchor_signal = anchor_co2[:, :control_horizon].mean()
        student_grad = torch.autograd.grad(
            student_signal,
            u_future,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        anchor_grad = torch.autograd.grad(
            anchor_signal,
            u_future,
            create_graph=False,
            retain_graph=True,
            allow_unused=False,
        )[0].detach()
        return (
            (
                student_grad[:, :control_horizon, self.co2_control_idx]
                - anchor_grad[:, :control_horizon, self.co2_control_idx]
            )
            ** 2
        ).mean()

    def compute_auxiliary_loss(
        self,
        x_past: torch.Tensor,
        w_future: torch.Tensor,
        u_future: torch.Tensor,
        y_true: torch.Tensor,
        prediction: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        base_prediction = self.base_model(x_past, w_future, u_future)
        pred_co2 = prediction[..., self.co2_target_idx]
        base_co2 = base_prediction[..., self.co2_target_idx].detach()
        control_horizon = min(self.control_horizon, prediction.size(1))
        first_step_loss = ((pred_co2[:, 0] - base_co2[:, 0]) ** 2).mean()
        control_horizon_loss = ((pred_co2[:, :control_horizon] - base_co2[:, :control_horizon]) ** 2).mean()
        gradient_loss = self._gradient_match_loss(x_past, w_future, u_future)
        return (
            self.first_step_weight * first_step_loss
            + self.control_horizon_weight * control_horizon_loss
            + self.gradient_weight * gradient_loss
        )

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        base_prediction = self.base_model(x_past, w_future, u_future)
        terminal_prediction = self.terminal_model(x_past, w_future, u_future)
        fusion_gate = self._co2_fusion(x_past, w_future, u_future, base_prediction, terminal_prediction)
        prediction = base_prediction.clone()
        base_co2 = base_prediction[..., self.co2_target_idx]
        terminal_co2 = terminal_prediction[..., self.co2_target_idx]
        prediction[..., self.co2_target_idx] = base_co2 + fusion_gate * (terminal_co2 - base_co2)
        return prediction
