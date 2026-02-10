# -*- coding: utf-8 -*-
"""
simulation/simulator.py
-------------------------
MPC vs MDP 滚动仿真主循环

在物理引擎驱动下对比 DPC 和 MDP 控制器的性能。
支持 Sim-to-Real PWM 映射：将 DPC 连续输出离散化为继电器 ON/OFF。
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimResult:
    """仿真结果容器"""
    history_mpc: List[float] = field(default_factory=list)
    history_mdp: List[float] = field(default_factory=list)
    actions_mpc: List[list] = field(default_factory=list)       # DPC 连续输出 [0-1]
    actions_mdp: List[list] = field(default_factory=list)       # MDP 离散输出 [0/1]
    pwm_actions_mpc: List[list] = field(default_factory=list)   # PWM 离散化后 [0/1]
    target_temp: float = 25.0
    sim_steps: int = 0
    pwm_enabled: bool = False


class Simulator:
    """
    仿真主循环

    Args:
        dpc: DPCController
        mdp: LegacyMDPController
        env_class: 物理环境类 (PhysicsGreenhouseEnv)
        feature_order: 特征列名列表
        scaler: MinMaxScaler
        target_idx: 温度列在特征中的索引
        config: Config 对象
        device: torch 设备
        pwm_sim: PWMSimulator 实例 (可选，若提供则启用 PWM 离散化)
    """

    def __init__(self, dpc, mdp, env_class, feature_order, scaler, target_idx,
                 config=None, device=None, pwm_sim=None):
        self.dpc = dpc
        self.mdp = mdp
        self.env_class = env_class
        self.feature_order = feature_order
        self.scaler = scaler
        self.target_idx = target_idx
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pwm_sim = pwm_sim

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

    def run(self, X_test_p, X_test_f, config=None):
        """
        运行仿真

        Args:
            X_test_p: 测试集历史序列
            X_test_f: 测试集未来序列
            config: 可选 Config (用于获取物理参数)

        Returns:
            SimResult
        """
        sim_steps = self._sim_steps
        start_idx = self._start_idx
        target_idx = self.target_idx
        feature_order = self.feature_order
        seq_len = self._seq_len
        pwm_enabled = self.pwm_sim is not None

        result = SimResult(
            target_temp=self._target_temp,
            sim_steps=sim_steps,
            pwm_enabled=pwm_enabled
        )

        # 初始状态
        current_state_mpc = X_test_p[start_idx:start_idx + 1].copy()
        current_state_mdp = X_test_p[start_idx:start_idx + 1].copy()
        future_base_seq = X_test_f[start_idx:start_idx + sim_steps]

        # 初始温度
        init_temp_norm = current_state_mpc[0, -1, target_idx]
        init_temp = self._inverse_transform(init_temp_norm, target_idx)
        env_mpc = self.env_class(init_temp, config)
        env_mdp = self.env_class(init_temp, config)

        # 查找室外温度和太阳辐射列索引
        tout_idx = feature_order.index('Outdoor_Temp') if 'Outdoor_Temp' in feature_order else -1
        solar_idx = feature_order.index('Outdoor_Solar') if 'Outdoor_Solar' in feature_order else -1

        # 重置 PWM 仿真器
        if pwm_enabled:
            self.pwm_sim.reset()

        pwm_label = " + PWM离散化" if pwm_enabled else ""
        print(f"---> 正在进行 {sim_steps} 步滚动优化仿真 (物理引擎驱动{pwm_label})...")

        for t in range(sim_steps):
            state_mpc_tensor = torch.FloatTensor(current_state_mpc).to(self.device)
            current_future_base = torch.FloatTensor(future_base_seq[t]).unsqueeze(0).to(self.device)

            # --- DPC: 输出连续占空比 ---
            curr_mpc_temp = env_mpc.current_temp
            opt_action, _ = self.dpc.get_optimal_action(
                state_mpc_tensor, current_future_base, current_temp=curr_mpc_temp
            )
            result.actions_mpc.append(opt_action)

            # --- PWM 离散化: 连续 → ON/OFF ---
            if pwm_enabled:
                self.pwm_sim.set_duty('heater', opt_action[0])
                self.pwm_sim.set_duty('vent', opt_action[1])
                relay_states = self.pwm_sim.step()
                actual_action = [
                    1.0 if relay_states.get('heater', False) else 0.0,
                    1.0 if relay_states.get('vent', False) else 0.0
                ]
                result.pwm_actions_mpc.append(actual_action)
            else:
                actual_action = opt_action  # 无 PWM 时直接用连续值

            # --- MDP ---
            curr_temp_norm_mdp = current_state_mdp[0, -1, target_idx]
            curr_temp_mdp = self._inverse_transform(curr_temp_norm_mdp, target_idx)
            mdp_action = self.mdp.get_action(curr_temp_mdp)
            result.actions_mdp.append(mdp_action)

            # --- 环境演变 (使用 actual_action 驱动物理引擎) ---
            with torch.no_grad():
                if tout_idx != -1:
                    curr_tout = self._inverse_transform(current_state_mpc[0, -1, tout_idx], tout_idx)
                else:
                    curr_tout = 15.0

                if solar_idx != -1:
                    curr_solar = self._inverse_transform(current_state_mpc[0, -1, solar_idx], solar_idx)
                else:
                    curr_solar = 0.0

                # MPC 环境推演 (用 PWM 离散化后的 actual_action)
                next_temp_real = env_mpc.step(actual_action, curr_tout, curr_solar)
                next_temp_norm = self._forward_transform(next_temp_real, target_idx)

                new_step_mpc = current_state_mpc[0, 1:, :].copy()
                next_real_features = X_test_p[start_idx + t + 1][-1, :].copy()
                next_real_features[target_idx] = next_temp_norm
                # 历史窗口记录的是实际执行的离散动作
                next_real_features[0] = actual_action[0]
                next_real_features[1] = actual_action[1]
                current_state_mpc = np.concatenate(
                    [new_step_mpc, next_real_features.reshape(1, -1)], axis=0
                ).reshape(1, seq_len, -1)
                result.history_mpc.append(next_temp_real)
                self.dpc.update_integral(next_temp_real)

                # MDP 环境推演
                next_temp_mdp_real = env_mdp.step(mdp_action, curr_tout, curr_solar)
                next_temp_mdp_norm = self._forward_transform(next_temp_mdp_real, target_idx)

                new_step_mdp = current_state_mdp[0, 1:, :].copy()
                next_feat_mdp = X_test_p[start_idx + t + 1][-1, :].copy()
                next_feat_mdp[target_idx] = next_temp_mdp_norm
                next_feat_mdp[0] = mdp_action[0]
                next_feat_mdp[1] = mdp_action[1]
                current_state_mdp = np.concatenate(
                    [new_step_mdp, next_feat_mdp.reshape(1, -1)], axis=0
                ).reshape(1, seq_len, -1)
                result.history_mdp.append(next_temp_mdp_real)

        print("---> 仿真完成。")
        return result
