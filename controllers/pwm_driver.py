# -*- coding: utf-8 -*-
"""
controllers/pwm_driver.py
---------------------------
Sim-to-Real PWM 映射模块

将 DPC 输出的连续控制信号 [0, 1] 映射为继电器开关时序。
"""


class PWMDriver:
    """
    PWM 驱动器: 连续信号 → 继电器开关时序

    将连续占空比 (0.0~1.0) 转换为一个 PWM 周期内的 ON/OFF 序列。
    支持最小吸合/关断时间约束，防止继电器频繁切换造成损坏。

    Args:
        cycle_minutes: PWM 周期 (分钟)
        min_on_minutes: 最小吸合时间 (分钟)
        min_off_minutes: 最小关断时间 (分钟)
    """

    def __init__(self, cycle_minutes=10, min_on_minutes=1, min_off_minutes=1):
        self.cycle = cycle_minutes
        self.min_on = min_on_minutes
        self.min_off = min_off_minutes

    def duty_to_schedule(self, duty_cycle):
        """
        将占空比转为开关序列

        Args:
            duty_cycle: 占空比 [0.0, 1.0]

        Returns:
            list[bool]: 长度为 cycle 的开关序列 (True=ON, False=OFF)
        """
        duty_cycle = max(0.0, min(1.0, duty_cycle))
        on_ticks = round(duty_cycle * self.cycle)

        # 应用最小吸合/关断约束
        if 0 < on_ticks < self.min_on:
            on_ticks = self.min_on
        off_ticks = self.cycle - on_ticks
        if 0 < off_ticks < self.min_off:
            on_ticks = self.cycle - self.min_off
            on_ticks = max(0, on_ticks)

        on_ticks = max(0, min(self.cycle, on_ticks))
        return [True] * on_ticks + [False] * (self.cycle - on_ticks)


class PWMSimulator:
    """
    PWM 仿真器: 在仿真循环中模拟 PWM 离散化效果

    在每个仿真步中，根据当前 PWM 周期内的位置决定实际输出 0/1。

    Args:
        driver: PWMDriver 实例
    """

    def __init__(self, driver):
        self.driver = driver
        self._schedules = {}    # channel_name → schedule
        self._tick = 0

    def set_duty(self, channel, duty_cycle):
        """设置通道占空比 (下一个周期生效)"""
        self._schedules[channel] = self.driver.duty_to_schedule(duty_cycle)

    def step(self):
        """
        推进一个仿真步

        Returns:
            dict: {channel_name: bool} 每个通道的当前开关状态
        """
        states = {}
        for ch, schedule in self._schedules.items():
            idx = self._tick % len(schedule)
            states[ch] = schedule[idx]
        self._tick += 1
        return states

    def reset(self):
        """重置仿真时钟"""
        self._tick = 0
        self._schedules.clear()
