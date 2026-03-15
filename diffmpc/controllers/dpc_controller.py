# -*- coding: utf-8 -*-
"""
controllers/dpc_controller.py
-------------------------------
可微预测控制器 (Differentiable Predictive Controller / DPC)

基于梯度的模型预测控制: 将控制动作视为可学习参数，
通过 PyTorch 自动微分 + Adam 优化器在线求解最优动作。
"""

import numpy as np
import torch


class DPCController:
    """
    DPC 控制器 (原 MPC_Controller)

    核心流程:
        1. 初始化可学习动作张量 (温度误差驱动的热启动)
        2. 前向传播: 动作 → DecisionControlModel → 预测轨迹
        3. 计算 Loss = 跟踪误差 + 能耗 + 平滑度
        4. 反向传播更新动作
        5. 物理互斥 + 比例安全兜底
    """

    def __init__(self, model, scaler, target_indices, future_indices,
                 config=None, horizon=10, target_temp=25.0):
        self.model = model
        self.scaler = scaler
        self.target_indices = target_indices
        self.future_indices = future_indices

        # 从 config 或参数读取
        if config is not None:
            self.horizon = getattr(config, 'dpc_horizon', horizon) # 控制 horizon (20)
            self.pred_horizon = config.horizon # 预测模型输出长度 (120)
            self.target_temp = config.target_temp
            self.target_hum = config.target_hum
            self.target_co2 = config.target_co2
            self._lr = config.dpc_lr
            self._iterations = config.dpc_iterations
            self._w_track_temp = config.w_track_temp
            self._w_track_hum = config.w_track_hum
            self._w_track_co2 = config.w_track_co2
            self._w_energy = config.w_energy
            self._w_smooth = config.w_smooth
            # 安全兜底参数
            self._safety = {
                'tier1_error': config.safety_tier1_error,
                'tier1_min': config.safety_tier1_min,
                'tier2_error': config.safety_tier2_error,
                'tier2_min': config.safety_tier2_min,
                'tier3_error': config.safety_tier3_error,
                'tier3_min': config.safety_tier3_min,
                'tier4_min': config.safety_tier4_min,
            }
            self._integral_decay = config.integral_decay
            self._integral_clip = config.integral_clip
            self._vent_suppress_margin = getattr(config, 'vent_suppress_margin', 2.0)
        else:
            self.horizon = horizon
            self.pred_horizon = 120
            self.target_temp = target_temp
            self._lr = 0.2
            self._iterations = 100
            self._w_track = 20.0
            self._w_energy = 0.001
            self._w_smooth = 0.1
            self._safety = {
                'tier1_error': 3.0, 'tier1_min': 0.98,
                'tier2_error': 1.0, 'tier2_min': 0.90,
                'tier3_error': 0.3, 'tier3_min': 0.80,
                'tier4_min': 0.60,
            }
            self._integral_decay = 0.95
            self._integral_clip = 20.0
            self._vent_suppress_margin = 2.0

        # 设备
        self._device = next(model.parameters()).device

        # 积分误差 (Leaky Integral for PID-I)
        self.integral_error = 0.0

        # 目标归一化值
        dummy = np.zeros((1, len(scaler.scale_)))
        dummy[0, target_indices[0]] = self.target_temp
        dummy[0, target_indices[1]] = self.target_hum
        dummy[0, target_indices[2]] = self.target_co2
        
        self.target_temp_norm = scaler.transform(dummy)[0, target_indices[0]]
        self.target_hum_norm = scaler.transform(dummy)[0, target_indices[1]]
        self.target_co2_norm = scaler.transform(dummy)[0, target_indices[2]]

        self.target_norms = torch.tensor([
            self.target_temp_norm, self.target_hum_norm, self.target_co2_norm
        ], device=self._device).view(1, 1, 3)

        # 上一步动作记录 (4维)
        self.last_action = [0, 0, 0, 0]
        self.last_action_continuous = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self._device)

    def get_optimal_action(self, current_past_tensor, current_future_base, current_temp=None):
        """
        基于梯度的在线动作优化

        Args:
            current_past_tensor: (1, seq_len, feat_dim)
            current_future_base: (1, horizon, future_dim)
            current_temp: 当前实际温度 (°C)

        Returns:
            best_action: [heater, vent] 最优动作
            best_loss: 最优 loss 值
        """
        # 1. 智能初始化
        if current_temp is not None:
            temp_error = self.target_temp - current_temp
            if temp_error > 5.0:
                init_direct = torch.tensor([[[0.95, 0.0, 0.0, 0.0]]])
            elif temp_error > 2.0:
                init_direct = torch.tensor([[[0.85, 0.0, 0.0, 0.0]]])
            elif temp_error > 0.5:
                init_direct = torch.tensor([[[0.7, 0.0, 0.0, 0.0]]])
            elif temp_error > 0:
                init_direct = torch.tensor([[[0.5, 0.0, 0.0, 0.0]]])
            else:
                init_direct = torch.tensor([[[0.1, 0.3, 0.0, 0.0]]])
        else:
            init_direct = self.last_action_continuous.view(1, 1, 4).clone()

        action_param = init_direct.to(self._device).requires_grad_(True)

        optimizer = torch.optim.Adam([action_param], lr=self._lr)

        # 冻结模型参数
        for p in self.model.parameters():
            p.requires_grad = False

        original_mode = self.model.training
        self.model.train()  # cuDNN RNN backward 需要 train 模式 (模型无 Dropout，不影响结果)

        best_loss = float('inf')
        best_u_soft = None

        try:
            for _ in range(self._iterations):
                optimizer.zero_grad()

                # Clamp 到 [0, 1]
                u_soft = torch.clamp(action_param, min=0.0, max=1.0)

                # 时域扩展 (Constant Action Assumption)
                u_expanded = u_soft.repeat(1, self.horizon, 1)

                # ==========================================================
                # 解耦: 动作张量长度为 dpc_horizon(20)，但模型前传需要 120 步 future_base
                # 策略: 真实天气提供 120 步，控制动作前 20 步优化，后 100 步用边缘填充
                # ==========================================================
                f_weather = current_future_base[:, :, 4:]  # (1, 120, N_weather)
                
                # 动作填充: [0:20] = u_expanded, [20:120] 用最后一步的动作填充 (或全0)
                u_padding = u_expanded[:, -1:, :].repeat(1, self.pred_horizon - self.horizon, 1)
                u_full = torch.cat([u_expanded, u_padding], dim=1) # (1, 120, 4)
                
                # 拼接: 控制动作(120) + 固定天气(120)
                x_future_optim = torch.cat([u_full, f_weather], dim=2)

                # 前向预测 (B, 120, 3)
                pred_norm_full = self.model(
                    current_past_tensor, x_future_optim,
                    target_temp_norm=self.target_temp_norm
                )
                
                # 截断: 控制器只关心前 dpc_horizon 步的跟踪误差
                pred_norm = pred_norm_full[:, :self.horizon, :] # (B, 20, 3)

                # Loss (多目标追踪只计算前 dpc_horizon 步)
                track_error_sq = (pred_norm - self.target_norms) ** 2
                loss_temp = torch.mean(track_error_sq[:, :, 0])
                loss_hum = torch.mean(track_error_sq[:, :, 1])
                loss_co2 = torch.mean(track_error_sq[:, :, 2])
                
                track_loss = (self._w_track_temp * loss_temp +
                              self._w_track_hum * loss_hum +
                              self._w_track_co2 * loss_co2)
                              
                energy_loss = torch.mean(torch.abs(u_soft))
                prev_u = self.last_action_continuous.view(1, 1, 4).detach()
                smooth_loss = torch.mean((u_soft - prev_u) ** 2)

                total_loss = (track_loss
                              + self._w_energy * energy_loss
                              + self._w_smooth * smooth_loss)

                total_loss.backward()
                optimizer.step()

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_u_soft = u_soft.detach()

        finally:
            self.model.train(original_mode)

        # 更新连续状态
        if best_u_soft is not None:
            self.last_action_continuous = best_u_soft.squeeze(0).squeeze(0)
        else:
            self.last_action_continuous = (
                torch.clamp(action_param, 0.0, 1.0).detach().squeeze(0).squeeze(0)
            )

        final_heater = self.last_action_continuous[0].item()
        final_vent = self.last_action_continuous[1].item()
        final_fog = self.last_action_continuous[2].item()
        final_lighting = self.last_action_continuous[3].item()

        # 物理互斥
        if final_heater > 0.1 and final_vent > 0.1:
            if final_heater > final_vent:
                final_vent = 0.0
            else:
                final_heater = 0.0

        # 比例安全兜底 (Proportional Safety Net)
        if current_temp is not None:
            error = self.target_temp - current_temp

            s = self._safety
            if error >= s['tier1_error']:
                final_heater = max(final_heater, s['tier1_min'])
            elif error >= s['tier2_error']:
                final_heater = max(final_heater, s['tier2_min'])
            elif error >= s['tier3_error']:
                final_heater = max(final_heater, s['tier3_min'])
            elif error > 0:
                final_heater = max(final_heater, s['tier4_min'])

        best_action = [final_heater, final_vent, final_fog, final_lighting]
        self.last_action = best_action.copy()
        return best_action, best_loss

    def inverse_transform_target(self, val, col_idx):
        """反归一化目标列"""
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, col_idx] = val
        return self.scaler.inverse_transform(dummy)[0, col_idx]

    def update_integral(self, current_temp):
        """更新积分误差 (Leaky Integral + Anti-windup)"""
        error = self.target_temp - current_temp
        self.integral_error *= self._integral_decay
        self.integral_error += error
        self.integral_error = np.clip(
            self.integral_error, -self._integral_clip, self._integral_clip
        )
