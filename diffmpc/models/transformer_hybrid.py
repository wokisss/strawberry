# -*- coding: utf-8 -*-
"""
models/transformer_hybrid.py
-----------------------------
Transformer-MoE 混合专家预测模型

核心改进:
  1. MoE 门控改为 Softmax 归一化 (权重之和恒等于1)
  2. 专家头改为逐步输出 (避免展平导致参数爆炸)
"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """经典 Transformer 位置编码 (正余弦函数)"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerHybridModel(nn.Module):
    """
    基于 Transformer 的变量解耦 MoE 预测模型

    架构:
        - Past Encoder:  Transformer Encoder (历史物理状态的全局依赖)
        - Future Decoder: Transformer Decoder (Cross-Attention 融合未来干预序列)
        - MoE Heads:     逐步输出 + Softmax 门控归一化

    [修正1] MoE 门控: clamp → Softmax, 确保权重之和始终为1
    [修正2] 专家头:   展平+大MLP → 逐步输出(d_model→1), 参数量不随 horizon 爆炸
    """
    def __init__(
        self, 
        input_dim, 
        future_dim, 
        target_dim, 
        forecast_horizon, 
        target_indices=None,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    ):
        super(TransformerHybridModel, self).__init__()
        self.target_dim = target_dim
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        
        if target_indices is not None:
            self.register_buffer('target_indices_tensor', torch.tensor(target_indices, dtype=torch.long))
        else:
            self.target_indices_tensor = None

        # --- 1. Embedding 层 (引入 Patch Tokenization) ---
        # 传统 Point-wise Tokenization:
        # self.past_val_embedding = nn.Linear(input_dim, d_model)
        # self.future_val_embedding = nn.Linear(future_dim, d_model)
        
        # PatchTST-style Tokenization: 使用 Conv1d 将相邻时间点聚合成一个 Token
        self.patch_len = 5    # 每个 Patch 的时间长度 (感受野)
        stride = 1       # 步长为 1 保持序列长度不变，方便后续一维逐时间点对应
        
        # 利用因果卷积 (Causal Convolution) 避免未来数据泄露，padding=0 并在 forward 中手动 padding
        self.past_val_embedding = nn.Conv1d(in_channels=input_dim, out_channels=d_model, 
                                            kernel_size=self.patch_len, stride=stride, padding=0)
        
        self.future_val_embedding = nn.Conv1d(in_channels=future_dim, out_channels=d_model, 
                                              kernel_size=self.patch_len, stride=stride, padding=0)

        self.pos_encoder = PositionalEncoding(d_model)

        # --- 2. Transformer 核心层 ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- 3. 变量解耦逐步输出头 (Per-Step Output) ---
        # [修正] 每个专家头: d_model → 1 (逐时间步输出)
        # 参数量 = O(d_model) 而非 O(d_model * horizon), 不随 horizon 爆炸
        
        # [Temperature] 3 个工况专家
        self.temp_expert_heat = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.temp_expert_vent = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.temp_expert_nat  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        # [Humidity] 3 个工况专家
        self.hum_expert_heat = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.hum_expert_vent = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.hum_expert_nat  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

        # [CO2] 3 个工况专家
        self.co2_expert_light = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.co2_expert_vent  = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.co2_expert_nat   = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_past, x_future):
        """
        Args:
            x_past:   (batch, seq_len, input_dim)  — 历史观测序列
            x_future: (batch, horizon, future_dim) — 未来控制/扰动序列

        Returns:
            final_pred: (batch, horizon, target_dim) — 目标多变量预测
        """
        # ==================== A. Transformer 编码解码 ====================
        # x_past: (batch, seq_len, input_dim) -> Conv1d 需要 (batch, channels, seq_len)
        x_past_t = x_past.transpose(1, 2)
        # 因果卷积 Padding (仅在时间序列左侧 pad)
        x_past_t = torch.nn.functional.pad(x_past_t, (self.patch_len - 1, 0))
        enc_emb = self.past_val_embedding(x_past_t) # (batch, d_model, seq_len)
        enc_emb = enc_emb.transpose(1, 2)           # 转回 (batch, seq_len, d_model)
        enc_emb = self.pos_encoder(enc_emb)
        memory = self.transformer_encoder(enc_emb)

        # x_future: (batch, horizon, future_dim) -> Conv1d 需要 (batch, channels, horizon)
        x_future_t = x_future.transpose(1, 2)
        # 因果卷积 Padding (仅在时间序列左侧 pad)
        x_future_t = torch.nn.functional.pad(x_future_t, (self.patch_len - 1, 0))
        dec_emb = self.future_val_embedding(x_future_t) # (batch, d_model, horizon)
        dec_emb = dec_emb.transpose(1, 2)             # 转回 (batch, horizon, d_model)
        dec_emb = self.pos_encoder(dec_emb)
        # dec_out: (B, Horizon, d_model) — 每个时间步都有独立的特征向量
        dec_out = self.transformer_decoder(tgt=dec_emb, memory=memory)

        # ==================== B. Softmax 门控 MoE ====================
        # 提取各控制通道的逐步信号强度 (不再取均值, 保留时间分辨率)
        # x_future = ['Heater', 'Ventilation', 'Fog', 'Lighting', ...]
        heat_signal  = x_future[:, :, 0:1]   # (B, H, 1)
        vent_signal  = x_future[:, :, 1:2]   # (B, H, 1)
        light_signal = x_future[:, :, 3:4]   # (B, H, 1)

        # [修正] 直接归一化物理控制信号，权重之和恒等于1，避免 Softmax 带来的幅度坍缩
        # Temperature 门控: heat / vent / natural
        nat_temp = torch.clamp(1.0 - heat_signal - vent_signal, min=0.0)
        sum_temp = heat_signal + vent_signal + nat_temp + 1e-8
        w_t_heat = heat_signal / sum_temp
        w_t_vent = vent_signal / sum_temp
        w_t_nat  = nat_temp / sum_temp

        # Humidity 门控: heat / vent / natural
        nat_hum = torch.clamp(1.0 - heat_signal - vent_signal, min=0.0)
        sum_hum = heat_signal + vent_signal + nat_hum + 1e-8
        w_h_heat = heat_signal / sum_hum
        w_h_vent = vent_signal / sum_hum
        w_h_nat  = nat_hum / sum_hum

        # CO2 门控: light / vent / natural
        nat_co2 = torch.clamp(1.0 - light_signal - vent_signal, min=0.0)
        sum_co2 = light_signal + vent_signal + nat_co2 + 1e-8
        w_c_light = light_signal / sum_co2
        w_c_vent  = vent_signal / sum_co2
        w_c_nat   = nat_co2 / sum_co2

        # ==================== C. 逐步专家输出 ====================
        # dec_out: (B, H, d_model) → 每个专家头在每个时间步独立输出 1 个值
        # 1. Temperature
        pred_temp = (w_t_heat * self.temp_expert_heat(dec_out) +
                     w_t_vent * self.temp_expert_vent(dec_out) +
                     w_t_nat  * self.temp_expert_nat(dec_out))     # (B, H, 1)

        # 2. Humidity
        pred_hum = (w_h_heat * self.hum_expert_heat(dec_out) +
                    w_h_vent * self.hum_expert_vent(dec_out) +
                    w_h_nat  * self.hum_expert_nat(dec_out))       # (B, H, 1)

        # 3. CO2
        pred_co2 = (w_c_light * self.co2_expert_light(dec_out) +
                    w_c_vent  * self.co2_expert_vent(dec_out) +
                    w_c_nat   * self.co2_expert_nat(dec_out))      # (B, H, 1)

        # ==================== D. 流重组与残差锚定 (Residual Anchoring) ====================
        # Transformer 输出的是未来每个时间步相比上一步的变化量 (Delta per step)
        delta_pred_step = torch.cat([pred_temp, pred_hum, pred_co2], dim=2)  # (B, H, 3)

        # 累积积分 (Autoregressive Cumulative Sum) 以符合物理惯性原理
        delta_pred_cum = torch.cumsum(delta_pred_step, dim=1) # (B, H, 3)

        # 提取历史序列的最后一步 (t=0) 的真实状态作为绝对锚点
        if self.target_indices_tensor is not None:
            initial_state = x_past[:, -1:, self.target_indices_tensor] 
            final_pred = initial_state + delta_pred_cum
        elif hasattr(self, 'target_indices') and self.target_indices is not None:
            initial_state = x_past[:, -1:, self.target_indices] 
            final_pred = initial_state + delta_pred_cum
        else:
            final_pred = delta_pred_cum

        return final_pred
