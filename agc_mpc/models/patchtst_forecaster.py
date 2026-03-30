# -*- coding: utf-8 -*-
"""Patch-based conditional forecaster for AGC multi-step forecasting."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_forecaster import PositionalEncoding


class ConditionalPatchTSTForecaster(nn.Module):
    """Encode patch tokens from history and decode future targets."""

    def __init__(
        self,
        seq_len: int,
        past_dim: int,
        weather_dim: int,
        control_dim: int,
        target_dim: int,
        hidden_dim: int = 96,
        patch_len: int = 12,
        patch_stride: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        ff_dim: int = 192,
        max_future_len: int = 128,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_proj = nn.Linear(past_dim * patch_len, hidden_dim)
        self.patch_pos = PositionalEncoding(hidden_dim, max_len=self._estimate_num_patches(seq_len))
        self.future_proj = nn.Linear(weather_dim + control_dim, hidden_dim)
        self.future_pos = PositionalEncoding(hidden_dim, max_len=max_future_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def _estimate_num_patches(self, seq_len: int) -> int:
        if seq_len <= self.patch_len:
            return 1
        return 1 + math.ceil((seq_len - self.patch_len) / self.patch_stride)

    def _patchify(self, x_past: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_past.shape
        if seq_len < self.patch_len:
            pad_len = self.patch_len - seq_len
        else:
            remainder = (seq_len - self.patch_len) % self.patch_stride
            pad_len = 0 if remainder == 0 else self.patch_stride - remainder
        if pad_len > 0:
            x_past = F.pad(x_past.transpose(1, 2), (0, pad_len), mode="replicate").transpose(1, 2)
        patches = x_past.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        return patches.contiguous().view(batch_size, patches.size(1), -1)

    def forward(self, x_past: torch.Tensor, w_future: torch.Tensor, u_future: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_proj(self._patchify(x_past))
        memory = self.encoder(self.patch_pos(patch_tokens))
        future_tokens = self.future_pos(self.future_proj(torch.cat([w_future, u_future], dim=-1)))
        decoded = self.decoder(future_tokens, memory)
        global_context = memory.mean(dim=1, keepdim=True).expand(-1, decoded.size(1), -1)
        fused = self.context_gate(torch.cat([decoded, global_context], dim=-1))
        return self.head(decoded + fused)
