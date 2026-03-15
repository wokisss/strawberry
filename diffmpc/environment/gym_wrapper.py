# -*- coding: utf-8 -*-
"""
environment/gym_wrapper.py
---------------------------
将 PhysicsGreenhouseEnv 包装为标准 Gymnasium 接口，用于 SAC 离线训练。

改进 v2:
    1. Reward 归一化到 [-1, 0] 区间，稳定 Critic 训练
    2. 观测空间扩展为 11 维: 室内状态(3) + 状态变化趋势(3) + 上一步动作(4) + 室外温度(1)
    3. 天气随机化增强，缩小 Domain Gap
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class GreenhouseGymEnv(gym.Env):
    """
    将 PhysicsGreenhouseEnv 包装为标准的 Gymnasium 接口，用于 SAC 离线训练。
    """
    # 观测维度: [Temp_in, Hum_in, CO2_in, dT, dHum, dCO2, last_act(4), Temp_out]
    OBS_DIM = 11

    def __init__(self, physics_env, config, scaler, target_indices):
        super(GreenhouseGymEnv, self).__init__()
        
        self.physics_env = physics_env
        self.config = config
        self.scaler = scaler
        self.target_indices = target_indices

        # 动作空间: [Heater, Vent, Fog, Lighting], 范围 [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # 观测空间: 11 维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32)

        # 记录每 Episode 步数
        self.current_step = 0
        self.max_steps = 300

        # 上一步的室内状态 (用于计算趋势)
        self._prev_state = np.zeros(3, dtype=np.float32)
        # 上一步的动作 (用于观测)
        self._prev_action = np.zeros(4, dtype=np.float32)

        # Reward 归一化用的参考尺度
        # 合理的最大单步惩罚: 温度差10°C, 湿度差30%, CO2差500ppm
        self._reward_scale = (
            abs(10.0 / 25.0) +    # 温度相对误差
            abs(30.0 / 70.0) +    # 湿度相对误差
            abs(500.0 / 800.0)    # CO2 相对误差
        )  # ≈ 1.46

        # 预加载天气
        self._generate_random_weather()

    def _generate_random_weather(self):
        """生成更有多样性的随机天气用于训练 (缩小 Domain Gap)"""
        self.weather_profile = np.zeros((self.max_steps, 4))
        
        # 更大范围的随机起点
        base_t = np.random.uniform(5, 30)       # 温度范围更宽
        base_h = np.random.uniform(30, 90)
        
        # 模拟真实日照变化: 正弦 + 随机扰动
        is_day = np.random.random() > 0.3  # 70% 概率有日照
        peak_solar = np.random.uniform(100, 700) if is_day else 0
        
        for t in range(self.max_steps):
            # 温度: 带趋势的随机游走 (模拟日夜温度波动)
            base_t += np.random.normal(0, 0.15)   # 稍大波动
            # 添加一个缓慢的周期性趋势
            base_t += 0.01 * np.sin(2 * np.pi * t / self.max_steps)
            base_t = np.clip(base_t, 0, 40)
            
            base_h += np.random.normal(0, 0.8)
            base_h = np.clip(base_h, 15, 98)
            
            # 日照: 半正弦曲线 + 噪声
            if is_day:
                solar = peak_solar * max(0, np.sin(np.pi * t / self.max_steps))
                solar += np.random.normal(0, 20)
                solar = np.clip(solar, 0, 900)
            else:
                solar = 0.0

            self.weather_profile[t] = [base_t, base_h, 400.0, solar]

    def _build_obs(self, state, weather):
        """构建 11 维观测向量"""
        # 趋势 = 当前 - 上一步
        delta = state - self._prev_state
        # 外部温度 (只传一个最重要的)
        out_temp = weather[0]
        
        obs = np.concatenate([
            state,                # [Temp_in, Hum_in, CO2_in]       3 维
            delta,                # [dT, dHum, dCO2]                3 维
            self._prev_action,    # [last_heat, last_vent, ...]     4 维
            [out_temp]            # [Temp_out]                      1 维
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._generate_random_weather()

        # 随机初始化室内状态
        init_state = np.array([
            np.random.uniform(10.0, 35.0),
            np.random.uniform(30.0, 90.0),
            np.random.uniform(400.0, 1500.0)
        ], dtype=np.float32)
        
        self.physics_env.reset(init_state)
        self._prev_state = init_state.copy()
        self._prev_action = np.zeros(4, dtype=np.float32)

        w = self.weather_profile[0]
        obs = self._build_obs(init_state, w)
        return obs, {}

    def step(self, action):
        w = self.weather_profile[self.current_step]
        out_temp, out_hum, out_co2, solar = w[0], w[1], w[2], w[3]

        # 步进物理环境
        next_state_real = self.physics_env.step(action, out_temp, out_hum, out_co2, solar)
        if isinstance(next_state_real, torch.Tensor):
            next_state_real = next_state_real.cpu().numpy()

        # 构建观测
        next_w = self.weather_profile[min(self.current_step + 1, self.max_steps - 1)]
        obs = self._build_obs(next_state_real, next_w)

        # ---- 计算 Reward (归一化到 [-1, 0] 附近) ----
        # 使用相对误差而非绝对误差，消除量级差异
        rel_error_temp = abs(next_state_real[0] - self.config.target_temp) / max(self.config.target_temp, 1.0)
        rel_error_hum = abs(next_state_real[1] - self.config.target_hum) / max(self.config.target_hum, 1.0)
        rel_error_co2 = abs(next_state_real[2] - self.config.target_co2) / max(self.config.target_co2, 1.0)
        
        # 归一化惩罚 (除以参考尺度，使 reward 大致 ∈ [-1, 0])
        track_penalty = (rel_error_temp + rel_error_hum + rel_error_co2) / self._reward_scale
        
        reward = -track_penalty  # 大致在 [-1, 0] 范围

        # 更新历史
        self._prev_state = next_state_real.copy()
        self._prev_action = np.array(action, dtype=np.float32)

        self.current_step += 1
        terminated = False
        truncated = bool(self.current_step >= self.max_steps)

        return obs, reward, terminated, truncated, {}
