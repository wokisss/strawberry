# -*- coding: utf-8 -*-
"""DiffMPC-style Transformer-hybrid adapted to AGC conditional forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding for batch-first tensors."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DiffMPCStyleTransformerHybridForecaster(nn.Module):
    """Port of the old DiffMPC predictor to the AGC `x_past / w_future / u_future` API."""

    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        target_dim: int,
        seq_len: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        patch_len: int = 5,
        target_indices: list[int] | None = None,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.horizon = horizon
        self.patch_len = patch_len

        if target_indices is None:
            target_indices = list(range(target_dim))
        self.register_buffer(
            "target_indices_tensor",
            torch.tensor(target_indices, dtype=torch.long),
            persistent=False,
        )

        self.variate_embedding = nn.Sequential(
            nn.Linear(seq_len, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.temporal_embedding = nn.Conv1d(
            in_channels=past_dim,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=1,
            padding=0,
        )
        self.future_embedding = nn.Conv1d(
            in_channels=future_dim,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=1,
            padding=0,
        )
        self.gru_past = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.gru_future = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(seq_len, horizon) + patch_len + 4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.temp_expert_heat = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.temp_expert_vent = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.temp_expert_nat = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        self.hum_expert_heat = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.hum_expert_vent = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.hum_expert_nat = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        self.co2_expert_light = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.co2_expert_vent = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.co2_expert_nat = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        gating_input_dim = d_model + future_dim
        self.gating_temp = nn.Sequential(nn.Linear(gating_input_dim, 32), nn.ReLU(), nn.Linear(32, 3))
        self.gating_hum = nn.Sequential(nn.Linear(gating_input_dim, 32), nn.ReLU(), nn.Linear(32, 3))
        self.gating_co2 = nn.Sequential(nn.Linear(gating_input_dim, 32), nn.ReLU(), nn.Linear(32, 3))

    def _embed_history(self, x_past: torch.Tensor) -> torch.Tensor:
        x_past_invert = x_past.transpose(1, 2)
        memory_var = self.transformer_encoder(self.variate_embedding(x_past_invert))

        x_past_conv = F.pad(x_past.transpose(1, 2), (self.patch_len - 1, 0))
        temporal_tokens = self.temporal_embedding(x_past_conv).transpose(1, 2)
        temporal_tokens = self.pos_encoder(temporal_tokens)
        rnn_past_out, _ = self.gru_past(temporal_tokens)
        return torch.cat([memory_var, rnn_past_out], dim=1)

    def _embed_future(
        self,
        future_known: torch.Tensor,
        memory_fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        future_conv = F.pad(future_known.transpose(1, 2), (self.patch_len - 1, 0))
        future_tokens = self.future_embedding(future_conv).transpose(1, 2)
        future_tokens = self.pos_encoder(future_tokens)
        rnn_future_out, _ = self.gru_future(future_tokens)
        return rnn_future_out, self.transformer_decoder(tgt=rnn_future_out, memory=memory_fused)

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        future_known = torch.cat([w_future, u_future], dim=-1)
        memory_fused = self._embed_history(x_past)
        rnn_future_out, dec_out = self._embed_future(future_known, memory_fused)

        gating_context = torch.cat([rnn_future_out, future_known], dim=-1)
        w_temp = torch.softmax(self.gating_temp(gating_context), dim=-1)
        w_hum = torch.softmax(self.gating_hum(gating_context), dim=-1)
        w_co2 = torch.softmax(self.gating_co2(gating_context), dim=-1)

        pred_temp = (
            w_temp[:, :, 0:1] * self.temp_expert_heat(dec_out)
            + w_temp[:, :, 1:2] * self.temp_expert_vent(dec_out)
            + w_temp[:, :, 2:3] * self.temp_expert_nat(dec_out)
        )
        pred_hum = (
            w_hum[:, :, 0:1] * self.hum_expert_heat(dec_out)
            + w_hum[:, :, 1:2] * self.hum_expert_vent(dec_out)
            + w_hum[:, :, 2:3] * self.hum_expert_nat(dec_out)
        )
        pred_co2 = (
            w_co2[:, :, 0:1] * self.co2_expert_light(dec_out)
            + w_co2[:, :, 1:2] * self.co2_expert_vent(dec_out)
            + w_co2[:, :, 2:3] * self.co2_expert_nat(dec_out)
        )

        delta_pred_step = torch.cat([pred_temp, pred_hum, pred_co2], dim=-1)
        delta_pred_cum = torch.cumsum(delta_pred_step, dim=1)
        initial_state = x_past[:, -1:, self.target_indices_tensor]
        return initial_state + delta_pred_cum
