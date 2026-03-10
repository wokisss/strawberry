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

        # --- 3. 变量解耦预测头 (Decoupled Variable Experts) ---
        # Decoder 吐出的是 (batch, horizon, d_model) 的隐式动力学特征
        # 展平以便通过全连接网络一次性预测未来整段曲线
        feature_size = d_model * forecast_horizon
        out_features = forecast_horizon * 1 # 每个头只负责 1 个变量的预测轨迹

        # [解耦分支 A: Temperature] 
        # 温度对高强度的 Heater 和 剧烈的 Ventilation 有中频热力学响应
        self.temp_expert_heat = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))
        self.temp_expert_vent = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))
        self.temp_expert_nat  = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))

        # [解耦分支 B: Humidity] 
        # 湿度对 Ventilation 有极高频的闪崩级响应，对 Heater 有迟滞蒸发响应
        self.hum_expert_heat = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))
        self.hum_expert_vent = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))
        self.hum_expert_nat  = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))

        # [解耦分支 C: CO2] 
        # 二氧化碳主要对 Ventilation 和 Lighting (光合作用) 有极低频的累积消耗响应
        self.co2_expert_light = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))
        self.co2_expert_vent  = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))
        self.co2_expert_nat   = nn.Sequential(nn.Linear(feature_size, 128), nn.ReLU(), nn.Linear(128, out_features))

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

        # ==================== B. 解耦级 MoE 专家门控预测 ====================
        # 提取各个控制通道在整个 Horizon 内的均值强度作为门控软指标 (Soft-Gating)
        # x_future = ['Heater', 'Ventilation', 'Fog', 'Lighting', ...]
        w_heat  = x_future[:, :, 0].mean(dim=1, keepdim=True)
        w_vent  = x_future[:, :, 1].mean(dim=1, keepdim=True)
        w_light = x_future[:, :, 3].mean(dim=1, keepdim=True)
        
        # 剥离互相覆盖的惩罚，保证自然演化态的基底
        w_nat_temp = torch.clamp(1.0 - w_heat - w_vent, min=0.0)
        w_nat_hum  = torch.clamp(1.0 - w_heat - w_vent, min=0.0)
        w_nat_co2  = torch.clamp(1.0 - w_light - w_vent, min=0.0)

        # 1. Temperature 解耦流 (中频热响应)
        pred_t_heat = self.temp_expert_heat(combined_features)
        pred_t_vent = self.temp_expert_vent(combined_features)
        pred_t_nat  = self.temp_expert_nat(combined_features)
        pred_temp   = (w_heat * pred_t_heat) + (w_vent * pred_t_vent) + (w_nat_temp * pred_t_nat)
        
        # 2. Humidity 解耦流 (高频水汽响应)
        pred_h_heat = self.hum_expert_heat(combined_features)
        pred_h_vent = self.hum_expert_vent(combined_features)
        pred_h_nat  = self.hum_expert_nat(combined_features)
        pred_hum    = (w_heat * pred_h_heat) + (w_vent * pred_h_vent) + (w_nat_hum * pred_h_nat)
        
        # 3. CO2 解耦流 (极低频光合/呼吸响应)
        pred_c_light = self.co2_expert_light(combined_features)
        pred_c_vent  = self.co2_expert_vent(combined_features)
        pred_c_nat   = self.co2_expert_nat(combined_features)
        pred_co2     = (w_light * pred_c_light) + (w_vent * pred_c_vent) + (w_nat_co2 * pred_c_nat)

        # ==================== C. 流重组 ====================
        # 将三条解耦流在特征维度重新拼接成 (B, horizon * 3) -> 变身为 (B, Horizon, 3)
        # 张量重排保证顺序对应: [Temp, Hum, CO2]
        pred_temp = pred_temp.view(-1, self.forecast_horizon, 1)
        pred_hum  = pred_hum.view(-1, self.forecast_horizon, 1)
        pred_co2  = pred_co2.view(-1, self.forecast_horizon, 1)

        final_pred = torch.cat([pred_temp, pred_hum, pred_co2], dim=2)
        
        return final_pred
