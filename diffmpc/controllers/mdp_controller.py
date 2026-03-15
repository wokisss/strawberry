# -*- coding: utf-8 -*-
"""
controllers/mdp_controller.py
-------------------------------
MDP 基准控制器 (Rule-based Baseline)

使用简单阈值规则的马尔可夫决策控制器，用于与 DPC 进行性能对比。
"""


class LegacyMDPController:
    """
    基于规则的 MDP 控制器

    逻辑:
        T < target  → 加热
        T > 28°C    → 通风
        否则        → 待机
    """

    def __init__(self, target_temp=25.0, vent_threshold=28.0):
        self.target_temp = target_temp
        self.vent_threshold = vent_threshold

    def get_action(self, current_temp):
        """
        获取动作

        Args:
            current_temp: 当前温度 (°C)

        Returns:
            [heater, vent] — 离散 0/1
        """
        if current_temp < self.target_temp:
            return [1, 0]  # 太冷 → 加热
        elif current_temp > self.vent_threshold:
            return [0, 1]  # 太热 → 通风
        else:
            return [0, 0]  # 适宜 → 待机
