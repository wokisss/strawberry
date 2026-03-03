# -*- coding: utf-8 -*-
"""
models/decision_model.py
-------------------------
决策专用代理模型 (Control-Oriented Surrogate Model)

包含物理引导梯度 (Physics-Guided Gradients / PGG) 机制。
通过显式注入物理先验知识，打破数据驱动模型的优化惰性。
"""

import torch
import torch.nn as nn


class DecisionControlModel(nn.Module):
    """
    物理引导梯度代理模型

    公式:
        T_decision = T_prediction + Cumsum( Gain_heat * u_heat + Gain_vent * u_vent )

    即使基础预测模型因数据惯性认为"加热器开不开都一样"，
    物理增益项也会让 MPC 求解器看到: 加热 → T升高 → Cost降低。

    Args:
        original_prediction_model: 基础预测模型 (SegmentedHybridModel)
        config: Config 对象 (必须提供，以读取多变量增益参数)
    """

    def __init__(self, original_prediction_model, config):
        super(DecisionControlModel, self).__init__()
        self.predictor = original_prediction_model

        self.heater_gain_temp = config.heater_gain_temp
        self.heater_gain_hum = config.heater_gain_hum
        self.vent_gain_temp = config.vent_gain_temp
        self.vent_gain_hum = config.vent_gain_hum
        self.vent_gain_co2 = config.vent_gain_co2
        self.fog_gain_temp = config.fog_gain_temp
        self.fog_gain_hum = config.fog_gain_hum
        self.lighting_gain_temp = config.lighting_gain_temp
        self.lighting_gain_co2 = config.lighting_gain_co2
            
        self._sigma = config.gain_sigma
        self._gain_min = config.gain_min
        self._gain_max = config.gain_max

    def forward(self, x_past, x_future, target_temp_norm=None):
        """
        Args:
            x_past:           (B, seq_len, feat_dim)
            x_future:         (B, horizon, future_dim)
            target_temp_norm: 目标温度归一化值 (标量，用于动态增益)

        Returns:
            control_prediction: (B, horizon, 3)
        """
        # 1. 基准预测 (Neutral Input — 假设不操作)
        x_future_neutral = x_future.clone()
        x_future_neutral[:, :, 0] = 0.0  # Heater=0
        x_future_neutral[:, :, 1] = 0.0  # Vent=0
        x_future_neutral[:, :, 2] = 0.0  # Fog=0
        x_future_neutral[:, :, 3] = 0.0  # Lighting=0
        base_prediction = self.predictor(x_past, x_future_neutral)

        # 2. 提取动作序列
        u_heat_seq = x_future[:, :, 0]   # (B, horizon)
        u_vent_seq = x_future[:, :, 1]   # (B, horizon)
        u_fog_seq = x_future[:, :, 2]    # (B, horizon)
        u_lighting_seq = x_future[:, :, 3]    # (B, horizon)

        # 3. 动态增益 (State-Dependent Gain)
        #    根据目标温度计算误差强度作为增益因子
        if target_temp_norm is not None:
            # 简化：使用基础预测的温度维度 [batch, horizon, 0] 计算误差
            temp_pred = base_prediction[:, :, 0] if base_prediction.dim() == 3 else base_prediction
            error_seq = torch.abs(target_temp_norm - temp_pred)
            gain_factor_seq = 1.0 - torch.exp(-error_seq ** 2 / (self._sigma ** 2))
            gain_factor_seq = torch.clamp(gain_factor_seq, min=self._gain_min, max=self._gain_max)
        else:
            gain_factor_seq = torch.ones_like(u_heat_seq)

        # 4. 组装多变量物理梯度 (Cumulative Integration)
        #    T[t] = base_T[t] + cumsum( delta_T )
        delta_temp = (u_heat_seq * self.heater_gain_temp * gain_factor_seq) + (u_vent_seq * self.vent_gain_temp * gain_factor_seq) + (u_fog_seq * self.fog_gain_temp * gain_factor_seq) + (u_lighting_seq * self.lighting_gain_temp * gain_factor_seq)
        delta_hum = (u_heat_seq * self.heater_gain_hum * gain_factor_seq) + (u_vent_seq * self.vent_gain_hum * gain_factor_seq) + (u_fog_seq * self.fog_gain_hum * gain_factor_seq)
        delta_co2 = (u_vent_seq * self.vent_gain_co2 * gain_factor_seq) + (u_lighting_seq * self.lighting_gain_co2 * gain_factor_seq)

        delta_temp_accum = torch.cumsum(delta_temp, dim=1).unsqueeze(2) # (B, horizon, 1)
        delta_hum_accum = torch.cumsum(delta_hum, dim=1).unsqueeze(2)   # (B, horizon, 1)
        delta_co2_accum = torch.cumsum(delta_co2, dim=1).unsqueeze(2)   # (B, horizon, 1)

        delta_physics_accum = torch.cat([delta_temp_accum, delta_hum_accum, delta_co2_accum], dim=2) # (B, horizon, 3)

        # 最终预测 = 数据驱动基准 + 物理修正
        control_prediction = base_prediction + delta_physics_accum

        return control_prediction
