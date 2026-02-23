# -*- coding: utf-8 -*-
"""
simulation/simulator.py
-------------------------
DPC vs PSO 滚动仿真主循环

在物理引擎驱动下对比 DPC 和 PSO 控制器的性能。
支持 Sim-to-Real PWM 映射：将连续输出离散化为继电器 ON/OFF。
"""

import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimResult:
    """仿真结果容器"""
    history_dpc: List[float] = field(default_factory=list)
    history_pso: List[float] = field(default_factory=list)
    actions_dpc: List[list] = field(default_factory=list)       # DPC 连续输出 [0-1]
    actions_pso: List[list] = field(default_factory=list)       # PSO 连续输出 [0-1]
    pwm_actions_dpc: List[list] = field(default_factory=list)   # DPC PWM 离散化
    pwm_actions_pso: List[list] = field(default_factory=list)   # PSO PWM 离散化
    time_dpc: List[float] = field(default_factory=list)         # DPC 每步耗时 (ms)
    time_pso: List[float] = field(default_factory=list)         # PSO 每步耗时 (ms)
    target_temp: float = 25.0
    sim_steps: int = 0
    pwm_enabled: bool = False


class Simulator:
    """
    仿真主循环: DPC vs PSO

    Args:
        dpc: DPCController
        pso: PSOController
        env_class: 物理环境类 (PhysicsGreenhouseEnv)
        feature_order: 特征列名列表
        scaler: MinMaxScaler
        target_idx: 温度列在特征中的索引
        config: Config 对象
        device: torch 设备
        pwm_sim_dpc: DPC 的 PWMSimulator 实例 (可选)
        pwm_sim_pso: PSO 的 PWMSimulator 实例 (可选)
    """

    def __init__(self, dpc, pso, env_class, feature_order, scaler, target_idx,
                 config=None, device=None, pwm_sim_dpc=None, pwm_sim_pso=None):
        self.dpc = dpc
        self.pso = pso
        self.env_class = env_class
        self.feature_order = feature_order
        self.scaler = scaler
        self.target_idx = target_idx
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pwm_sim_dpc = pwm_sim_dpc
        self.pwm_sim_pso = pwm_sim_pso

        if config is not None:
            self._sim_steps = config.sim_steps
            self._start_idx = config.sim_start_idx
            self._target_temp = config.target_temp
            self._seq_len = config.seq_len
        else:
            self._sim_steps = 300
            self._start_idx = 0
            self._target_temp = 25.0
            self._seq_len = 60

    def _inverse_transform(self, val, col_idx):
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, col_idx] = val
        return self.scaler.inverse_transform(dummy)[0, col_idx]

    def _forward_transform(self, val, col_idx):
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, col_idx] = val
        return self.scaler.transform(dummy)[0, col_idx]

    def _apply_pwm(self, pwm_sim, action):
        """将连续动作通过 PWM 映射为离散 ON/OFF"""
        if pwm_sim is None:
            return action, None

        pwm_sim.set_duty('heater', action[0])
        pwm_sim.set_duty('vent', action[1])
        relay_states = pwm_sim.step()
        actual = [
            1.0 if relay_states.get('heater', False) else 0.0,
            1.0 if relay_states.get('vent', False) else 0.0
        ]
        return actual, actual

    def _step_env(self, env, actual_action, current_state, t, start_idx, X_test_p):
        """对一个控制器执行一步环境推演，返回新温度和新状态"""
        target_idx = self.target_idx
        seq_len = self._seq_len
        feature_order = self.feature_order

        tout_idx = feature_order.index('Outdoor_Temp') if 'Outdoor_Temp' in feature_order else -1
        solar_idx = feature_order.index('Outdoor_Solar') if 'Outdoor_Solar' in feature_order else -1

        if tout_idx != -1:
            curr_tout = self._inverse_transform(current_state[0, -1, tout_idx], tout_idx)
        else:
            curr_tout = 15.0

        if solar_idx != -1:
            curr_solar = self._inverse_transform(current_state[0, -1, solar_idx], solar_idx)
        else:
            curr_solar = 0.0

        next_temp_real = env.step(actual_action, curr_tout, curr_solar)
        next_temp_norm = self._forward_transform(next_temp_real, target_idx)

        new_step = current_state[0, 1:, :].copy()
        next_real_features = X_test_p[start_idx + t + 1][-1, :].copy()
        next_real_features[target_idx] = next_temp_norm
        next_real_features[0] = actual_action[0]
        next_real_features[1] = actual_action[1]
        new_state = np.concatenate(
            [new_step, next_real_features.reshape(1, -1)], axis=0
        ).reshape(1, seq_len, -1)

        return next_temp_real, new_state

    def run(self, X_test_p, X_test_f, config=None):
        """
        运行仿真

        Returns:
            SimResult
        """
        sim_steps = self._sim_steps
        start_idx = self._start_idx
        target_idx = self.target_idx
        pwm_on = self.pwm_sim_dpc is not None

        result = SimResult(
            target_temp=self._target_temp,
            sim_steps=sim_steps,
            pwm_enabled=pwm_on
        )

        # 初始状态 (两个控制器各自独立的状态副本)
        current_state_dpc = X_test_p[start_idx:start_idx + 1].copy()
        current_state_pso = X_test_p[start_idx:start_idx + 1].copy()
        future_base_seq = X_test_f[start_idx:start_idx + sim_steps]

        # 初始温度
        init_temp_norm = current_state_dpc[0, -1, target_idx]
        init_temp = self._inverse_transform(init_temp_norm, target_idx)
        env_dpc = self.env_class(init_temp, config)
        env_pso = self.env_class(init_temp, config)

        # 重置 PWM
        if self.pwm_sim_dpc is not None:
            self.pwm_sim_dpc.reset()
        if self.pwm_sim_pso is not None:
            self.pwm_sim_pso.reset()

        pwm_label = " + PWM离散化" if pwm_on else ""
        print(f"---> 正在进行 {sim_steps} 步滚动仿真 (DPC vs PSO{pwm_label})...")

        for t in range(sim_steps):
            current_future_base = torch.FloatTensor(future_base_seq[t]).unsqueeze(0).to(self.device)

            # ============ DPC 控制器 ============
            state_dpc_tensor = torch.FloatTensor(current_state_dpc).to(self.device)
            curr_dpc_temp = env_dpc.current_temp

            t0 = time.time()
            opt_action_dpc, _ = self.dpc.get_optimal_action(
                state_dpc_tensor, current_future_base, current_temp=curr_dpc_temp
            )
            result.time_dpc.append((time.time() - t0) * 1000)
            result.actions_dpc.append(opt_action_dpc)

            # DPC PWM
            actual_dpc, pwm_dpc = self._apply_pwm(self.pwm_sim_dpc, opt_action_dpc)
            if pwm_dpc is not None:
                result.pwm_actions_dpc.append(pwm_dpc)

            # ============ PSO 控制器 ============
            state_pso_tensor = torch.FloatTensor(current_state_pso).to(self.device)
            curr_pso_temp = env_pso.current_temp

            t0 = time.time()
            opt_action_pso, _ = self.pso.get_optimal_action(
                state_pso_tensor, current_future_base, current_temp=curr_pso_temp
            )
            result.time_pso.append((time.time() - t0) * 1000)
            result.actions_pso.append(opt_action_pso)

            # PSO PWM
            actual_pso, pwm_pso = self._apply_pwm(self.pwm_sim_pso, opt_action_pso)
            if pwm_pso is not None:
                result.pwm_actions_pso.append(pwm_pso)

            # ============ 环境推演 ============
            with torch.no_grad():
                # DPC 环境
                next_temp_dpc, current_state_dpc = self._step_env(
                    env_dpc, actual_dpc, current_state_dpc, t, start_idx, X_test_p
                )
                result.history_dpc.append(next_temp_dpc)
                self.dpc.update_integral(next_temp_dpc)

                # PSO 环境
                next_temp_pso, current_state_pso = self._step_env(
                    env_pso, actual_pso, current_state_pso, t, start_idx, X_test_p
                )
                result.history_pso.append(next_temp_pso)
                self.pso.update_integral(next_temp_pso)

            # 进度显示
            if (t + 1) % 50 == 0:
                print(f"    步 {t+1}/{sim_steps} | "
                      f"DPC={next_temp_dpc:.1f}°C (avg {np.mean(result.time_dpc[-50:]):.0f}ms/step) | "
                      f"PSO={next_temp_pso:.1f}°C (avg {np.mean(result.time_pso[-50:]):.0f}ms/step)")

        print("---> 仿真完成。")
        return result
