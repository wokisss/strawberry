# -*- coding: utf-8 -*-
"""
controllers/sac_controller.py
-------------------------------
SAC 策略推理控制�?

将训练好�?SAC Agent 包装为与 DPC 相同�?get_optimal_action 接口�?
用于�?Simulator 对比仿真主循环中�?DPC 同台竞技�?

改进 v2:
    1. 正确�?current_past_tensor 中逆归一化提取室外温�?(消除硬编�?
    2. 构建�?Gym 训练一致的 11 �?observation (含趋�?+ 上一步动�?
"""

import numpy as np
import torch
import time
from .sac.sac_agent import SAC


class SACController:
    """
    SAC 策略推理控制�?
    
    核心流程:
        1. get_optimal_action(): 从时序张量中还原真实物理状态，构建 11 维观�?
        2. Actor forward (deterministic) �?输出最优动�?
        3. 比例安全兜底 + 物理互斥 (�?DPC 完全一�?
    """
    def __init__(self, agent_model, scaler, target_indices, future_indices,
                 feature_order=None, config=None, horizon=10, target_temp=25.0):
        self.agent = agent_model
        self.scaler = scaler
        self.target_indices = target_indices
        self.future_indices = future_indices
        self._device = self.agent.device

        # 特征列名顺序 (用于找到 Outdoor_Temp 的索�?
        self._feature_order = feature_order or []
        self._out_temp_idx = -1
        self._out_hum_idx = -1
        self._out_solar_idx = -1
        if 'Outdoor_Temp' in self._feature_order:
            self._out_temp_idx = self._feature_order.index('Outdoor_Temp')
        if 'Outdoor_Hum' in self._feature_order:
            self._out_hum_idx = self._feature_order.index('Outdoor_Hum')
        if 'Outdoor_Solar' in self._feature_order:
            self._out_solar_idx = self._feature_order.index('Outdoor_Solar')

        if config is not None:
            self.target_temp = config.target_temp
            self._integral_decay = config.integral_decay
            self._integral_clip = config.integral_clip
            self._safety = {
                'tier1_error': config.safety_tier1_error,
                'tier1_min': config.safety_tier1_min,
                'tier2_error': config.safety_tier2_error,
                'tier2_min': config.safety_tier2_min,
                'tier3_error': config.safety_tier3_error,
                'tier3_min': config.safety_tier3_min,
                'tier4_min': config.safety_tier4_min,
            }
        else:
            self.target_temp = target_temp
            self._integral_decay = 0.95
            self._integral_clip = 20.0
            self._safety = {
                'tier1_error': 3.0, 'tier1_min': 0.98,
                'tier2_error': 1.0, 'tier2_min': 0.90,
                'tier3_error': 0.3, 'tier3_min': 0.80,
                'tier4_min': 0.60,
            }

        # 积分误差
        self.integral_error = 0.0
        self.last_step_time = 0.0
        self.last_action = [0.0, 0.0, 0.0, 0.0]
        
        # 上一步的室内真实状�?(用于计算趋势)
        self._prev_state = np.zeros(3, dtype=np.float32)
        self._first_step = True

    def _inverse_transform(self, val, col_idx):
        """将归一化值反变换回原始物理量"""
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, col_idx] = val
        return self.scaler.inverse_transform(dummy)[0, col_idx]

    def get_optimal_action(self, current_past_tensor, current_future_base, current_temp=None):
        """
        基于 SAC Actor 网络的前向推�?
        
        构建�?gym_wrapper 训练一致的 15 �?observation:
        [Temp_in, Hum_in, CO2_in, dT, dHum, dCO2, last_act(4), Temp_out, Hum_out, Solar, Future_Temp_mean, Future_Solar_mean]
        """
        t_start = time.time()
        
        # --- 1. �?current_past_tensor 逆归一化提取室内真实状�?---
        last_state_norm = current_past_tensor[0, -1, :].cpu().numpy()
        t_idx, h_idx, c_idx = self.target_indices[0], self.target_indices[1], self.target_indices[2]
        
        r_temp = self._inverse_transform(last_state_norm[t_idx], t_idx)
        r_hum = self._inverse_transform(last_state_norm[h_idx], h_idx)
        r_co2 = self._inverse_transform(last_state_norm[c_idx], c_idx)
        
        current_state = np.array([r_temp, r_hum, r_co2], dtype=np.float32)
        
        # --- 2. 提取室外环境特征 (�?past 特征最后一帧中逆归一�? ---
        if self._out_temp_idx >= 0:
            out_temp = self._inverse_transform(last_state_norm[self._out_temp_idx], self._out_temp_idx)
        else:
            out_temp = 15.0
        
        # [Fix] 新增：提取室外湿度和太阳辐射，与 gym_wrapper 训练环境保持一�?
        if self._out_hum_idx >= 0:
            out_hum = self._inverse_transform(last_state_norm[self._out_hum_idx], self._out_hum_idx)
        else:
            out_hum = 50.0
        
        if self._out_solar_idx >= 0:
            out_solar = self._inverse_transform(last_state_norm[self._out_solar_idx], self._out_solar_idx)
        else:
            out_solar = 0.0
            
        # --- 3. 提取未来 120 步的天气预报特征 (Forecast) ---
        # future_indices 包含: 控制(4) + 天气/光照(0) + 时间(2) + 外部特征(4)
        # 由于我们排除�?Illumination, 所�?future_indices 构成�? 
        #   0:Heater, 1:Vent, 2:Fog, 3:Light, 
        #   4:Hour_Sin, 5:Hour_Cos, 
        #   6:Outdoor_Temp, 7:Outdoor_Solar, 8:Outdoor_Hum, 9:Outdoor_Wind
        # 直接使用预先存好的列名索引提取，最严谨�?
        
        # �?horizon 的均�?
        mean_future_norm = torch.mean(current_future_base[0], dim=0).cpu().numpy()
        
        idx_future_out_temp = self.future_indices.index(self._out_temp_idx) if self._out_temp_idx in self.future_indices else -1
        idx_future_out_solar = self.future_indices.index(self._out_solar_idx) if self._out_solar_idx in self.future_indices else -1

        if idx_future_out_temp >= 0:
            future_temp_mean = self._inverse_transform(mean_future_norm[idx_future_out_temp], self._out_temp_idx)
        else:
            future_temp_mean = out_temp
            
        if idx_future_out_solar >= 0:
            future_solar_mean = self._inverse_transform(mean_future_norm[idx_future_out_solar], self._out_solar_idx)
        else:
            future_solar_mean = out_solar
        
        # --- 4. 构建趋势特征 ---
        if self._first_step:
            delta = np.zeros(3, dtype=np.float32)
            self._first_step = False
        else:
            delta = current_state - self._prev_state
        
        self._prev_state = current_state.copy()
        
        # --- 5. 组装 15 维观�?---
        obs = np.concatenate([
            current_state,                                          # [Temp, Hum, CO2]    3 �?
            delta,                                                  # [dT, dHum, dCO2]    3 �?
            np.array(self.last_action, dtype=np.float32),           # [last_act * 4]      4 �?
            [out_temp, out_hum, out_solar],                         # [Temp_out, Hum_out, Solar] 3 �?
            [future_temp_mean, future_solar_mean]                   # [Future_Temp_mean, Future_Solar_mean] 2 �?
        ]).astype(np.float32)

        # --- 5. SAC Actor 前向 (deterministic) ---
        action = self.agent.select_action(obs, evaluate=True)

        # --- 6. 比例安全兜底 (�?DPC 完全一�? ---
        final_heater = float(action[0])
        final_vent = float(action[1])
        final_fog = float(action[2])
        final_lighting = float(action[3])

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

        # 物理互斥
        if final_heater > 0.1 and final_vent > 0.1:
            if final_heater > final_vent:
                final_vent = 0.0
            else:
                final_heater = 0.0

        best_action = [final_heater, final_vent, final_fog, final_lighting]
        self.last_action = best_action.copy()
        self.last_step_time = time.time() - t_start
        
        return best_action, 0.0

    def update_integral(self, current_temp):
        error = self.target_temp - current_temp
        self.integral_error *= self._integral_decay
        self.integral_error += error
        self.integral_error = np.clip(
            self.integral_error, -self._integral_clip, self._integral_clip
        )

    def reset(self):
        """Reset controller state before a fresh rollout."""
        self.integral_error = 0.0
        self.last_step_time = 0.0
        self.last_action = [0.0, 0.0, 0.0, 0.0]
        self._prev_state = np.zeros(3, dtype=np.float32)
        self._first_step = True
