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
import torch


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

    def __init__(self, initial_state, config=None):
        """
        Args:
            initial_state: 初始室内状态 [Temp(°C), Hum(%), CO2(ppm)], numpy array or torch tensor
            config: Config 对象，包含物理参数。若为 None 则使用默认值。
        """
        # 判断输入类型，支持 PyTorch Tensor
        self.is_tensor = isinstance(initial_state, torch.Tensor)
        self.device = initial_state.device if self.is_tensor else None

        if self.is_tensor:
            self.current_temp = initial_state[0].clone()
            self.current_hum = initial_state[1].clone()
            self.current_co2 = initial_state[2].clone()
        else:
            self.current_temp = initial_state[0]
            self.current_hum = initial_state[1]
            self.current_co2 = initial_state[2]

        if config is not None:
            # 物理参数
            self.k_insulation = config.k_insulation
            self.power_heater = config.power_heater
            self.eff_vent = config.eff_vent
            self.k_solar = config.k_solar
            
            # 多变量新增物理参数 (真实执行器)
            self.power_fog_hum = getattr(config, 'power_fog_hum', 5.0)        # 起雾1分钟增加的湿度 %
            self.power_fog_temp = getattr(config, 'power_fog_temp', -0.5)     # 起雾蒸发导致的降温 °C
            self.power_lighting_co2 = getattr(config, 'power_lighting_co2', -50.0) # 补光灯促进光合作用消耗的 CO2 ppm
            self.power_lighting_temp = getattr(config, 'power_lighting_temp', 0.2) # 补光灯产热升温 °C
            
            self.crop_transpiration = getattr(config, 'crop_transpiration', 0.2)   # 作物蒸腾导致湿度每分钟上升的自然速率 %/min
            self.crop_photosynthesis = getattr(config, 'crop_photosynthesis', 2.0) # 自然光合作用消耗CO2的速率 ppm/min
            self.crop_respiration = getattr(config, 'crop_respiration', 0.5)       # 呼吸作用释放CO2的速率 ppm/min

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

            self.power_fog_hum = 5.0
            self.power_fog_temp = -0.5
            self.power_lighting_co2 = -50.0
            self.power_lighting_temp = 0.2
            
            self.crop_transpiration = 0.2
            self.crop_photosynthesis = 2.0
            self.crop_respiration = 0.5

            self._ou_theta = 0.15
            self._ou_sigma = 0.2
            self._ou_mu = 0.0
            self._sensor_noise_std = 0.1
            self._act_noise_low = 0.9
            self._act_noise_high = 1.1
            self._wind_gust_prob = 0.05
            self._wind_gust_mag = 0.5

        # OU 过程内部状态 (3维: Temp, Hum, CO2)
        if self.is_tensor:
            self._ou_state = torch.zeros(3, device=self.device)
        else:
            self._ou_state = np.zeros(3)

    def step(self, action, outside_temp, outside_hum=50.0, outside_co2=400.0, solar_radiation=0.0):
        """
        执行一步物理仿真 (1分钟)

        Args:
            action: 控制动作 [heater, vent, fog, lighting]，范围 [0, 1] (Tensor/Array)
            outside_temp: 室外温度 (°C)
            outside_hum: 室外湿度 (%)
            outside_co2: 室外CO2浓度 (ppm)
            solar_radiation: 太阳辐射强度

        Returns:
            np.ndarray / torch.Tensor: 观测到的室内状态 [Temp, Hum, CO2]，含传感器噪声
        """
        # ---- 1. 执行器噪声: 对实际控制动作施加乘性波动 ----
        if self.is_tensor:
            if not isinstance(outside_temp, torch.Tensor): outside_temp = torch.tensor(outside_temp, dtype=torch.float32, device=self.device)
            if not isinstance(outside_hum, torch.Tensor): outside_hum = torch.tensor(outside_hum, dtype=torch.float32, device=self.device)
            if not isinstance(outside_co2, torch.Tensor): outside_co2 = torch.tensor(outside_co2, dtype=torch.float32, device=self.device)
            if not isinstance(solar_radiation, torch.Tensor): solar_radiation = torch.tensor(solar_radiation, dtype=torch.float32, device=self.device)

            act_noise = torch.empty(4, device=self.device).uniform_(self._act_noise_low, self._act_noise_high)
            noisy_action = torch.clamp(action * act_noise, 0.0, 1.0)
            u_heat, u_vent, u_fog, u_lighting = noisy_action[0], noisy_action[1], noisy_action[2], noisy_action[3]
        else:
            act_noise = np.random.uniform(self._act_noise_low, self._act_noise_high, size=4)
            noisy_action = np.clip(np.array(action) * act_noise, 0.0, 1.0)
            u_heat, u_vent, u_fog, u_lighting = noisy_action

        # ---- 2. 物理过程计算 (使用带噪声的动作) ----
        # 1) 温度动态
        delta_loss_t = -self.k_insulation * (self.current_temp - outside_temp)
        delta_heat_t = self.power_heater * u_heat
        delta_vent_t = -self.eff_vent * u_vent * (self.current_temp - outside_temp)
        delta_solar_t = self.k_solar * solar_radiation
        delta_fog_t = self.power_fog_temp * u_fog
        delta_lighting_t = self.power_lighting_temp * u_lighting
        
        # 2) 湿度动态 (耦合温度和通风)
        # 加热导致相对湿度下降 (简化为线性耦合)
        delta_heat_h = -2.0 * delta_heat_t
        # 通风导致湿度向室外湿度靠拢
        delta_vent_h = -self.eff_vent * u_vent * (self.current_hum - outside_hum)
        # 蒸腾作用随光照增强
        actual_transpiration = self.crop_transpiration * (1.0 + 0.5 * min(1.0, solar_radiation / 500.0) if not self.is_tensor else 1.0 + 0.5 * torch.clamp(solar_radiation / 500.0, max=1.0))
        # 起雾机加湿
        delta_fog_h = self.power_fog_hum * u_fog
        
        # 3) CO2 动态
        # 通风导致CO2向室外靠拢
        delta_vent_c = -self.eff_vent * u_vent * (self.current_co2 - outside_co2)
        # 自然光合作用消耗 CO2
        if self.is_tensor:
            actual_photosynthesis = self.crop_photosynthesis * torch.clamp(solar_radiation / 300.0, max=1.0)
        else:
            actual_photosynthesis = self.crop_photosynthesis * min(1.0, solar_radiation / 300.0)
            
        # 补光灯大幅促进光合作用，强力消耗 CO2
        delta_lighting_c = self.power_lighting_co2 * u_lighting
        
        delta_bio_c = self.crop_respiration - actual_photosynthesis

        # ---- 3. OU 过程噪声 (3维) ----
        if self.is_tensor:
            self._ou_state += (self._ou_theta * (self._ou_mu - self._ou_state)
                               + self._ou_sigma * torch.randn(3, device=self.device))
        else:
            self._ou_state += (self._ou_theta * (self._ou_mu - self._ou_state)
                               + self._ou_sigma * np.random.normal(0, 1, size=3))

        # ---- 4. 风扰动 (低概率随机脉冲，同时影响所有变量) ----
        if self.is_tensor:
            wind_gust = torch.zeros(3, device=self.device)
            if torch.rand(1, device=self.device).item() < self._wind_gust_prob:
                wind_gust[0] = torch.empty(1, device=self.device).uniform_(-self._wind_gust_mag, self._wind_gust_mag)[0]
                wind_gust[1] = torch.empty(1, device=self.device).uniform_(-self._wind_gust_mag * 5.0, self._wind_gust_mag * 5.0)[0]
                wind_gust[2] = torch.empty(1, device=self.device).uniform_(-self._wind_gust_mag * 50.0, self._wind_gust_mag * 50.0)[0]
        else:
            wind_gust = np.zeros(3)
            if np.random.random() < self._wind_gust_prob:
                wind_gust[0] = np.random.uniform(-self._wind_gust_mag, self._wind_gust_mag)
                wind_gust[1] = np.random.uniform(-self._wind_gust_mag * 5.0, self._wind_gust_mag * 5.0)
                wind_gust[2] = np.random.uniform(-self._wind_gust_mag * 50.0, self._wind_gust_mag * 50.0)

        # ---- 5. 更新真实状态 ----
        self.current_temp += delta_loss_t + delta_heat_t + delta_vent_t + delta_solar_t + delta_fog_t + delta_lighting_t + self._ou_state[0] + wind_gust[0]
        self.current_hum += delta_heat_h + delta_vent_h + actual_transpiration + delta_fog_h + self._ou_state[1] + wind_gust[1]
        self.current_co2 += delta_vent_c + delta_bio_c + delta_lighting_c + self._ou_state[2] + wind_gust[2]
        
        # 物理硬边界裁剪
        if self.is_tensor:
            self.current_hum = torch.clamp(self.current_hum, 10.0, 100.0)
            self.current_co2 = torch.clamp(self.current_co2, min=200.0)
        else:
            self.current_hum = np.clip(self.current_hum, 10.0, 100.0)
            self.current_co2 = max(200.0, self.current_co2)

        # ---- 6. 传感器噪声 (仅影响观测，不影响真实状态) ----
        if self.is_tensor:
            obs_temp = self.current_temp + torch.randn(1, device=self.device)[0] * self._sensor_noise_std
            obs_hum = self.current_hum + torch.randn(1, device=self.device)[0] * (self._sensor_noise_std * 2.0)
            obs_co2 = self.current_co2 + torch.randn(1, device=self.device)[0] * (self._sensor_noise_std * 15.0)
            return torch.stack([obs_temp, obs_hum, obs_co2])
        else:
            obs_temp = self.current_temp + np.random.normal(0, self._sensor_noise_std)
            obs_hum = self.current_hum + np.random.normal(0, self._sensor_noise_std * 2.0)
            obs_co2 = self.current_co2 + np.random.normal(0, self._sensor_noise_std * 15.0)
            return np.array([obs_temp, obs_hum, obs_co2])

    def reset(self, initial_state):
        """重置环境温度和噪声状态"""
        self.is_tensor = isinstance(initial_state, torch.Tensor)
        self.device = initial_state.device if self.is_tensor else None

        if self.is_tensor:
            self.current_temp = initial_state[0].clone()
            self.current_hum = initial_state[1].clone()
            self.current_co2 = initial_state[2].clone()
            self._ou_state = torch.zeros(3, device=self.device)
        else:
            self.current_temp = initial_state[0]
            self.current_hum = initial_state[1]
            self.current_co2 = initial_state[2]
            self._ou_state = np.zeros(3)
