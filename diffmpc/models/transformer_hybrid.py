# -*- coding: utf-8 -*-
"""
models/transformer_hybrid.py
-----------------------------
iTransformer-MoE 混合专家预测模型

核心改进:
  1. [iTransformer] Encoder 端采用变量倒置 Tokenization，将每个传感器的完整时间序列
     作为独立 Token 输入 Self-Attention，极大增强跨变量物理耦合学习，降低计算量。
  2. MoE 门控改为 Softmax 归一化 (权重之和恒等于1)
  3. 专家头改为逐步输出 (避免展平导致参数爆炸)
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
    基于 iTransformer 思想融合 MoE 机制的多变量预测模型

    架构 (双路 Encoder 融合):
        - Path 1 (iTransformer): 变量级 Token，捕获跨变量物理耦合 (Batch, V, d_model)
        - Path 2 (Temporal):    时间步级 Token (Conv1d+GRU)，捕获分钟级时序定位 (Batch, T, d_model)
        - Future Decoder:       Cross-Attention 融合未来干预方案，同时查询两种 Memory
        - MoE Heads:            逐步输出 + Softmax 门控归一化
        
    [解决问题] 纯 iTransformer 丢失分钟级时间分辨率导致预测平滑、控制震荡。双路架构完美互补。
    """
    def __init__(
        self, 
        input_dim,          # 变量数量 V (例如 18 个传感器)
        seq_len,            # 历史窗口长度 T (例如 240)
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

        # --- 1A. Embedding 层 (iTransformer Variate Tokenization) ---
        # Encoder 端 Path 1: 将长度为 seq_len 的单变量时间序列映射为 d_model 的隐向量
        # 使用 3层 MLP 增加容量，避免线性平均导致高频特征丢失
        self.variate_embedding = nn.Sequential(
            nn.Linear(seq_len, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # --- 1B. Embedding 层 (Temporal Tokenization) ---
        # Encoder 端 Path 2: 传统的时序保留路径，用于提供分钟级的历史精确定位
        self.patch_len = 5
        stride = 1
        self.temporal_embedding = nn.Conv1d(
            in_channels=input_dim, out_channels=d_model, 
            kernel_size=self.patch_len, stride=stride, padding=0
        )
        self.gru_past = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

        # --- 1C. Future Embedding ---
        # Decoder 端 (未来控制): 保留时间级 Tokenization (将一拍的多变量预测映射为 d_model)
        self.future_val_embedding = nn.Conv1d(
            in_channels=future_dim, out_channels=d_model, 
            kernel_size=self.patch_len, stride=stride, padding=0
        )
        self.gru_future = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

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

        # --- 4. 带有物理物理惯性的门控网络 (Inertial MoE Gating) ---
        # 门控上下文 = dec_out/rnn_future_out (时序历史遗留, d_model) + 原始控制信号幅度 (future_dim)
        gating_input_dim = d_model + future_dim
        
        self.gating_temp = nn.Sequential(nn.Linear(gating_input_dim, 32), nn.ReLU(), nn.Linear(32, 3)) # Heat, Vent, Nat
        self.gating_hum = nn.Sequential(nn.Linear(gating_input_dim, 32), nn.ReLU(), nn.Linear(32, 3))  # Heat, Vent, Nat
        self.gating_co2 = nn.Sequential(nn.Linear(gating_input_dim, 32), nn.ReLU(), nn.Linear(32, 3))  # Light, Vent, Nat

    def forward(self, x_past, x_future):
        """
        Args:
            x_past:   (batch, seq_len, input_dim)  — 历史观测序列
            x_future: (batch, horizon, future_dim) — 未来控制/扰动序列

        Returns:
            final_pred: (batch, horizon, target_dim) — 目标多变量预测
        """
        # ==================== A. 双路 Encoder 编码 ====================
        
        # --- Path 1: iTransformer Encoder (变量关系建模) ---
        # 1. 变量维度反转: (B, T, V) -> (B, V, T)
        x_past_invert = x_past.transpose(1, 2)
        # 2. 映射到 d_model: (B, V, d_model)
        enc_emb_var = self.variate_embedding(x_past_invert)
        # 3. Transformer Encoder (Self-Attention on Variables)
        memory_var = self.transformer_encoder(enc_emb_var)  # (B, input_dim, d_model)

        # --- Path 2: Temporal Encoder (分钟级时序定位) ---
        # (B, T, V) -> (B, V, T) for Conv1d
        x_past_t = x_past.transpose(1, 2)
        x_past_t = torch.nn.functional.pad(x_past_t, (self.patch_len - 1, 0))
        enc_emb_time = self.temporal_embedding(x_past_t) # (B, d_model, T)
        enc_emb_time = enc_emb_time.transpose(1, 2)      # (B, T, d_model)
        # 加位置编码后送入 GRU
        enc_emb_time_pos = self.pos_encoder(enc_emb_time)
        rnn_past_out, _ = self.gru_past(enc_emb_time_pos) # (B, T, d_model)

        # --- 融合双路 Memory ---
        # 让 Decoder 可以同时查询跨变量关系 (V个Token) 和具体历史时间点 (T个Token)
        # 拼接后的 memory shape: (B, input_dim + seq_len, d_model)
        memory_fused = torch.cat([memory_var, rnn_past_out], dim=1)

        # ==================== B. Future Decoder 编码 ====================
        # x_future 依然保持时间步粒度，每个时间点是一个 Token
        # (batch, horizon, future_dim) -> Conv1d 需要 (batch, channels, horizon)
        x_future_t = x_future.transpose(1, 2)
        # 因果卷积 Padding (仅在时间序列左侧 pad)
        x_future_t = torch.nn.functional.pad(x_future_t, (self.patch_len - 1, 0))
        dec_emb = self.future_val_embedding(x_future_t) # (batch, d_model, horizon)
        dec_emb = dec_emb.transpose(1, 2)             # 转回 (batch, horizon, d_model)
        
        # [Fix] 先加位置编码，再送入 GRU (与 Past 路径保持一致)
        dec_emb_pos = self.pos_encoder(dec_emb)
        rnn_future_out, _ = self.gru_future(dec_emb_pos)

        # [注] Direct Multi-Step 架构下，x_future 是控制器预先规划好的完整动作序列，
        #      Decoder 需要感知整个控制方案来预测每个时间步的状态，因此不使用因果掩码 (tgt_mask)。
        #      如果未来改为自回归逐步生成，此处必须添加因果掩码以防止未来信息泄露。
        # dec_out: (B, Horizon, d_model) — 每个时间步都有独立的特征向量
        dec_out = self.transformer_decoder(tgt=rnn_future_out, memory=memory_fused)

        # ==================== C. 带有物理物理惯性的专家门控 (Inertial MoE) ====================
        # [修正] 不再简单地根据当前的开关量大小做软门控，因为这违背了长序列积分惯性
        # 将原始的 future 控制指令 (蕴含瞬时激活幅度) 和 rnn_future_out (蕴含经过时间平滑的动作累积量) 拼接
        
        # [Fix] gating_context: (B, H, d_model + future_dim)
        # 拼接顺序与 gating_input_dim = d_model + future_dim 声明一致
        gating_context = torch.cat([rnn_future_out, x_future], dim=-1)

        # Temperature 门控: heat / vent / natural
        w_temp = torch.softmax(self.gating_temp(gating_context), dim=-1)
        w_t_heat, w_t_vent, w_t_nat = w_temp[:, :, 0:1], w_temp[:, :, 1:2], w_temp[:, :, 2:3]

        # Humidity 门控: heat / vent / natural
        w_hum = torch.softmax(self.gating_hum(gating_context), dim=-1)
        w_h_heat, w_h_vent, w_h_nat = w_hum[:, :, 0:1], w_hum[:, :, 1:2], w_hum[:, :, 2:3]

        # CO2 门控: light / vent / natural
        w_co2 = torch.softmax(self.gating_co2(gating_context), dim=-1)
        w_c_light, w_c_vent, w_c_nat = w_co2[:, :, 0:1], w_co2[:, :, 1:2], w_co2[:, :, 2:3]

        # ==================== D. 逐步专家输出 ====================
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

        # ==================== E. 流重组与残差锚定 (Residual Anchoring) ====================
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
