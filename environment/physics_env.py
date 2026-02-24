# -*- coding: utf-8 -*-
"""
environment/physics_env.py
--------------------------
基于物理常识的简化温室环境 (Simplified Physics-based Greenhouse Environment)

用于在仿真实验中作为"真实世界"的反馈源 (Ground Truth)。
根据当前温度、控制动作（加热/通风）、外部天气（温度/光照）计算下一时刻的温度。

噪声模型 (v2):
    1. OU 过程噪声 — 替代简单白噪声，提供时间相关扰动
    2. 传感器噪声 — 仅影响观测返回值，不影响真实状态
    3. 执行器噪声 — 对控制动作施加乘性随机波动
    4. 风扰动     — 低概率随机脉冲，模拟开门/阵风事件
"""

import numpy as np


class PhysicsGreenhouseEnv:
    """
    简化温室热力学仿真环境

    基于一阶热力学微分方程的离散化:
        dT/dt = Q_loss + Q_heater + Q_vent + Q_solar

    其中:
        Q_loss   = -k_insulation * (T_in - T_out)          传导热损耗
        Q_heater = power_heater * u_heat                    加热增益
        Q_vent   = -eff_vent * u_vent * (T_in - T_out)     通风热交换
        Q_solar  = k_solar * Solar                          太阳辐射增益
    """

    def __init__(self, initial_temp, config=None):
        """
        Args:
            initial_temp: 初始室内温度 (°C)
            config: Config 对象，包含物理参数。若为 None 则使用默认值。
        """
        self.current_temp = initial_temp

        if config is not None:
            # 物理参数
            self.k_insulation = config.k_insulation
            self.power_heater = config.power_heater
            self.eff_vent = config.eff_vent
            self.k_solar = config.k_solar

            # OU 过程参数
            self._ou_theta = getattr(config, 'ou_theta', 0.15)
            self._ou_sigma = getattr(config, 'ou_sigma', 0.2)
            self._ou_mu = getattr(config, 'ou_mu', 0.0)

            # 传感器噪声
            self._sensor_noise_std = getattr(config, 'sensor_noise_std', 0.1)

            # 执行器噪声
            self._act_noise_low = getattr(config, 'actuator_noise_low', 0.9)
            self._act_noise_high = getattr(config, 'actuator_noise_high', 1.1)

            # 风扰动
            self._wind_gust_prob = getattr(config, 'wind_gust_prob', 0.05)
            self._wind_gust_mag = getattr(config, 'wind_gust_magnitude', 0.5)
        else:
            self.k_insulation = 0.05
            self.power_heater = 0.5
            self.eff_vent = 0.1
            self.k_solar = 0.01

            self._ou_theta = 0.15
            self._ou_sigma = 0.2
            self._ou_mu = 0.0
            self._sensor_noise_std = 0.1
            self._act_noise_low = 0.9
            self._act_noise_high = 1.1
            self._wind_gust_prob = 0.05
            self._wind_gust_mag = 0.5

        # OU 过程内部状态
        self._ou_state = 0.0

    def step(self, action, outside_temp, solar_radiation=0.0):
        """
        执行一步物理仿真 (1分钟)

        Args:
            action (list/array): 控制动作 [heater_power, ventilation_rate]，范围 [0, 1]
            outside_temp (float): 室外温度 (°C)
            solar_radiation (float): 太阳辐射强度

        Returns:
            float: 观测到的室内温度 (°C)，含传感器噪声
        """
        # ---- 1. 执行器噪声: 对实际控制动作施加乘性波动 ----
        act_noise_heat = np.random.uniform(self._act_noise_low, self._act_noise_high)
        act_noise_vent = np.random.uniform(self._act_noise_low, self._act_noise_high)
        noisy_heat = np.clip(action[0] * act_noise_heat, 0.0, 1.0)
        noisy_vent = np.clip(action[1] * act_noise_vent, 0.0, 1.0)

        # ---- 2. 物理过程计算 (使用带噪声的动作) ----
        # 传导热损耗
        delta_loss = -self.k_insulation * (self.current_temp - outside_temp)
        # 加热增益
        delta_heat = self.power_heater * noisy_heat
        # 通风热交换
        delta_vent = -self.eff_vent * noisy_vent * (self.current_temp - outside_temp)
        # 太阳辐射增益
        delta_solar = self.k_solar * solar_radiation

        # ---- 3. OU 过程噪声 (替代简单白噪声，具有时间相关性) ----
        # 离散 Ornstein-Uhlenbeck: X_{t+1} = X_t + θ(μ - X_t) + σ·N(0,1)
        self._ou_state += (self._ou_theta * (self._ou_mu - self._ou_state)
                           + self._ou_sigma * np.random.normal(0, 1))

        # ---- 4. 风扰动 (低概率随机脉冲) ----
        wind_gust = 0.0
        if np.random.random() < self._wind_gust_prob:
            wind_gust = np.random.uniform(-self._wind_gust_mag, self._wind_gust_mag)

        # ---- 5. 更新真实温度 ----
        self.current_temp += (
            delta_loss + delta_heat + delta_vent + delta_solar
            + self._ou_state + wind_gust
        )

        # ---- 6. 传感器噪声 (仅影响观测，不影响真实状态) ----
        observed_temp = self.current_temp + np.random.normal(0, self._sensor_noise_std)

        return observed_temp

    def reset(self, initial_temp):
        """重置环境温度和噪声状态"""
        self.current_temp = initial_temp
        self._ou_state = 0.0
