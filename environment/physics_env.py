# -*- coding: utf-8 -*-
"""
environment/physics_env.py
--------------------------
基于物理常识的简化温室环境 (Simplified Physics-based Greenhouse Environment)

用于在仿真实验中作为"真实世界"的反馈源 (Ground Truth)。
根据当前温度、控制动作（加热/通风）、外部天气（温度/光照）计算下一时刻的温度。
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
            self.k_insulation = config.k_insulation
            self.power_heater = config.power_heater
            self.eff_vent = config.eff_vent
            self.k_solar = config.k_solar
            self.noise_std = config.noise_std
        else:
            self.k_insulation = 0.05
            self.power_heater = 0.5
            self.eff_vent = 0.1
            self.k_solar = 0.01
            self.noise_std = 0.05

    def step(self, action, outside_temp, solar_radiation=0.0):
        """
        执行一步物理仿真 (1分钟)

        Args:
            action (list/array): 控制动作 [heater_power, ventilation_rate]，范围 [0, 1]
            outside_temp (float): 室外温度 (°C)
            solar_radiation (float): 太阳辐射强度

        Returns:
            float: 更新后的室内温度 (°C)
        """
        # 传导热损耗: 室内外温差越大，热损越快
        delta_loss = -self.k_insulation * (self.current_temp - outside_temp)

        # 加热增益: 加热器功率 × 开度
        delta_heat = self.power_heater * action[0]

        # 通风热交换: 通风将室内温度拉向室外温度
        delta_vent = -self.eff_vent * action[1] * (self.current_temp - outside_temp)

        # 太阳辐射增益: 温室效应
        delta_solar = self.k_solar * solar_radiation

        # 更新状态 + 随机过程噪声 (模拟未建模扰动)
        self.current_temp += (
            delta_loss + delta_heat + delta_vent + delta_solar
            + np.random.normal(0, self.noise_std)
        )
        return self.current_temp

    def reset(self, initial_temp):
        """重置环境温度"""
        self.current_temp = initial_temp
