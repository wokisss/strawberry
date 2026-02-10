# -*- coding: utf-8 -*-
"""
models/segmented_hybrid.py
---------------------------
分段混合模型 (Segmented Hybrid Model) - 三头专家系统

结合历史时序特征提取和未来控制信号处理的混合预测模型。
采用"专家混合 (Mixture of Experts)" 架构，针对不同控制模式设计专用预测头。
"""

import torch
import torch.nn as nn


class SegmentedHybridModel(nn.Module):
    """
    三头专家预测模型

    架构:
        Encoder: CNN → BiGRU → Attention (历史特征) + GRU (未来特征)
        Decoder: 3个专家头 (加热/通风/自然) + 门控融合

    Args:
        input_dim: 历史特征维度 (x_past 的最后一维)
        future_dim: 未来特征维度 (x_future 的最后一维)
        forecast_horizon: 预测窗口长度
        hidden_dim: 隐藏层维度
    """

    def __init__(self, input_dim, future_dim, forecast_horizon, hidden_dim=32):
        super(SegmentedHybridModel, self).__init__()

        # --- 共享特征提取器 (Encoder) ---
        # CNN 提取局部时序模式
        self.past_conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=64, kernel_size=3, padding=1
        )
        # BiGRU 捕捉长距离时序依赖
        self.past_bigru = nn.GRU(
            input_size=64, hidden_size=hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True
        )
        # Attention 计算时间步重要性
        self.past_attention = nn.Linear(hidden_dim * 2, 1)
        # GRU 处理未来输入序列
        self.future_gru = nn.GRU(
            input_size=future_dim, hidden_size=hidden_dim, num_layers=1,
            batch_first=True
        )

        # --- 分段输出头 (Expert Decoders) ---
        feature_size = hidden_dim * 2 + hidden_dim

        # 专家 A: 加热模式 (拟合温升曲线)
        self.fc_heat = nn.Sequential(
            nn.Linear(feature_size, 32), nn.ReLU(), nn.Linear(32, forecast_horizon)
        )
        # 专家 B: 通风模式 (拟合降温曲线)
        self.fc_vent = nn.Sequential(
            nn.Linear(feature_size, 32), nn.ReLU(), nn.Linear(32, forecast_horizon)
        )
        # 专家 C: 自然模式 (拟合自然冷却/加热曲线)
        self.fc_natural = nn.Sequential(
            nn.Linear(feature_size, 32), nn.ReLU(), nn.Linear(32, forecast_horizon)
        )

    def forward(self, x_past, x_future):
        """
        Args:
            x_past:   (batch, seq_len, input_dim)  — 历史观测序列
            x_future: (batch, horizon, future_dim) — 未来控制量+天气

        Returns:
            final_pred: (batch, horizon) — 预测目标序列
        """
        # 1. 历史特征提取
        x_p = x_past.permute(0, 2, 1)                    # (B, C, T)
        x_p = torch.relu(self.past_conv1(x_p))
        x_p = x_p.permute(0, 2, 1)                       # (B, T, 64)

        gru_out_p, _ = self.past_bigru(x_p)               # (B, T, H*2)
        weights_p = torch.softmax(self.past_attention(gru_out_p), dim=1)
        attended_p = torch.sum(weights_p * gru_out_p, dim=1)  # (B, H*2)

        # 2. 未来特征提取
        _, h_n = self.future_gru(x_future)
        future_features = h_n[-1]                         # (B, H)

        # 3. 特征融合
        combined = torch.cat([attended_p, future_features], dim=1)

        # 4. 专家预测
        pred_heat = self.fc_heat(combined)
        pred_vent = self.fc_vent(combined)
        pred_natural = self.fc_natural(combined)

        # 5. 门控融合 (根据控制信号强度加权)
        heater_signal = x_future[:, :, 0].mean(dim=1, keepdim=True)
        vent_signal = x_future[:, :, 1].mean(dim=1, keepdim=True)

        w_heat = heater_signal
        w_vent = vent_signal
        w_natural = torch.clamp(1.0 - w_heat - w_vent, min=0.0)

        final_pred = (w_heat * pred_heat) + (w_vent * pred_vent) + (w_natural * pred_natural)
        return final_pred
