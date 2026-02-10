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
        config: Config 对象 (可选)。若提供则从中读取增益参数。
        heater_gain: 加热物理增益 (归一化空间)
        vent_gain: 通风物理增益 (归一化空间)
    """

    def __init__(self, original_prediction_model, config=None,
                 heater_gain=0.05, vent_gain=-0.05):
        super(DecisionControlModel, self).__init__()
        self.predictor = original_prediction_model

        if config is not None:
            self.heater_gain = config.heater_gain
            self.vent_gain = config.vent_gain
            self._sigma = config.gain_sigma
            self._gain_min = config.gain_min
            self._gain_max = config.gain_max
        else:
            self.heater_gain = heater_gain
            self.vent_gain = vent_gain
            self._sigma = 0.1
            self._gain_min = 0.2
            self._gain_max = 1.0

    def forward(self, x_past, x_future, target_temp_norm=None):
        """
        Args:
            x_past:           (B, seq_len, feat_dim)
            x_future:         (B, horizon, future_dim)
            target_temp_norm: 目标温度归一化值 (标量，用于动态增益)

        Returns:
            control_prediction: (B, horizon)
        """
        # 1. 基准预测 (Neutral Input — 假设不操作)
        x_future_neutral = x_future.clone()
        x_future_neutral[:, :, 0] = 0.0  # Heater=0
        x_future_neutral[:, :, 1] = 0.0  # Vent=0
        base_prediction = self.predictor(x_past, x_future_neutral)

        # 2. 提取动作序列
        u_heat_seq = x_future[:, :, 0]   # (B, horizon)
        u_vent_seq = x_future[:, :, 1]   # (B, horizon)

        # 3. 动态增益 (State-Dependent Gain)
        #    高斯核: 误差越大增益越强，接近目标时减弱
        if target_temp_norm is not None:
            error_seq = torch.abs(target_temp_norm - base_prediction)
            gain_factor_seq = 1.0 - torch.exp(-error_seq ** 2 / (self._sigma ** 2))
            gain_factor_seq = torch.clamp(gain_factor_seq, min=self._gain_min, max=self._gain_max)
        else:
            gain_factor_seq = torch.ones_like(base_prediction)

        effective_heater_gain = self.heater_gain * gain_factor_seq
        effective_vent_gain = self.vent_gain * gain_factor_seq

        # 4. 物理梯度注入 (Cumulative Integration)
        #    T[t] = base[t] + cumsum(delta_physics)
        step_delta_physics = (u_heat_seq * effective_heater_gain) + (u_vent_seq * effective_vent_gain)
        delta_physics_accum = torch.cumsum(step_delta_physics, dim=1)

        # 最终预测 = 数据驱动基准 + 物理修正
        if base_prediction.dim() == 2 and delta_physics_accum.dim() == 2:
            control_prediction = base_prediction + delta_physics_accum
        else:
            control_prediction = base_prediction + delta_physics_accum.view_as(base_prediction)

        return control_prediction
