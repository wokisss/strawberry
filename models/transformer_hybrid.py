# -*- coding: utf-8 -*-
"""
models/transformer_hybrid.py
-----------------------------
Transformer-MoE 混合专家预测模型

使用 Transformer Encoder-Decoder 架构代替原有的 CNN-BiGRU，
解决温室物理系统长距离时滞（大惯性）序列建模的问题。
保留了经典的三头物理专家门控融合机制 (MoE)。
"""

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    经典的 Transformer 位置编码
    (基于正余弦函数)
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerHybridModel(nn.Module):
    """
    基于 Transformer 的三头门控专家模型

    架构:
        - Past Encoder: Transformer Encoder (提取历史物理状态的全局/跨距离依赖)
        - Future Decoder: Transformer Decoder (Cross-Attention 融合未来干预序列)
        - MoE Heads: 3 个物理专家输出头 (加热/通风/自然) + 按动作强度融合
    """
    def __init__(
        self, 
        input_dim, 
        future_dim, 
        target_dim, 
        forecast_horizon, 
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

        # --- 1. Embedding 层 ---
        # 历史特征的值映射
        self.past_val_embedding = nn.Linear(input_dim, d_model)
        # 未来特征的值映射
        self.future_val_embedding = nn.Linear(future_dim, d_model)
        
        # 统一的位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # --- 2. Transformer 核心层 ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- 3. 专家输出头 (Expert Decoders) ---
        # Decoder 吐出的是 (batch, horizon, d_model) 的表征
        # 需要展平用于最终一次性预测全排序列
        feature_size = d_model * forecast_horizon
        out_features = forecast_horizon * target_dim

        # 专家 A: 加热模式 (重点拟合升温、降湿曲线)
        self.fc_heat = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features)
        )
        # 专家 B: 通风模式 (重点拟合快降温、快降湿、CO2交换曲线)
        self.fc_vent = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features)
        )
        # 专家 C: 自然模式 (无强干预时的缓慢大周期波动曲线)
        self.fc_natural = nn.Sequential(
            nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features)
        )

    def forward(self, x_past, x_future):
        """
        Args:
            x_past:   (batch, seq_len, input_dim)  — 历史观测序列
            x_future: (batch, horizon, future_dim) — 未来控制/扰动序列

        Returns:
            final_pred: (batch, horizon, target_dim) — 目标多变量预测序列
        """
        # ==================== A. Transformer 编码解码 ====================
        # 1. 历史序列 Embedding & 编码
        #    (B, Seq, In_Dim) -> (B, Seq, d_model)
        enc_emb = self.past_val_embedding(x_past)
        enc_emb = self.pos_encoder(enc_emb)
        
        # Memory 包含长距离过去的高阶非线性关联表征
        # memory: (B, Seq, d_model)
        memory = self.transformer_encoder(enc_emb)

        # 2. 未来序列 Embedding (作为 Query)
        #    (B, Horizon, Fut_Dim) -> (B, Horizon, d_model)
        dec_emb = self.future_val_embedding(x_future)
        dec_emb = self.pos_encoder(dec_emb)
        
        # 为了防止未来步泄露自身（因任务是非自回归一次性输出），
        # 在标准的回归/预测里，这里可以不加 Casual Mask，因为输入是已知条件（动作预案）。
        dec_out = self.transformer_decoder(tgt=dec_emb, memory=memory)

        # 展平以便送入 MLP Expert 头
        # (B, Horizon, d_model) -> (B, Horizon * d_model)
        combined_features = dec_out.reshape(dec_out.size(0), -1)

        # ==================== B. MoE 专家门控预测 ====================
        # 3. 独立并行的三个物理专家进行联合预演
        pred_heat_flat = self.fc_heat(combined_features)
        pred_vent_flat = self.fc_vent(combined_features)
        pred_natural_flat = self.fc_natural(combined_features)

        # 4. 根据实际未来动作的强度，软聚类(Soft-Gating)各专家的预测
        # 采用 x_future 的前两维（假设永远是 Heater 和 Ventilation）
        heater_signal = x_future[:, :, 0].mean(dim=1, keepdim=True)
        vent_signal = x_future[:, :, 1].mean(dim=1, keepdim=True)

        w_heat = heater_signal
        w_vent = vent_signal
        # 余下的权重分配给自然演化模式
        w_natural = torch.clamp(1.0 - w_heat - w_vent, min=0.0)

        # 加权融合
        final_pred_flat = (w_heat * pred_heat_flat) + (w_vent * pred_vent_flat) + (w_natural * pred_natural_flat)
        
        # (B, Horizon * Target) -> (B, Horizon, Target)
        final_pred = final_pred_flat.view(-1, self.forecast_horizon, self.target_dim)
        
        return final_pred
