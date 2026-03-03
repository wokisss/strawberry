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
    history_dpc: List[np.ndarray] = field(default_factory=list) # [Temp, Hum, CO2]
    history_pso: List[np.ndarray] = field(default_factory=list) # [Temp, Hum, CO2]
    actions_dpc: List[list] = field(default_factory=list)       # DPC 连续输出 [0-1]
    actions_pso: List[list] = field(default_factory=list)       # PSO 连续输出 [0-1]
    pwm_actions_dpc: List[list] = field(default_factory=list)   # DPC PWM 离散化
    pwm_actions_pso: List[list] = field(default_factory=list)   # PSO PWM 离散化
    time_dpc: List[float] = field(default_factory=list)         # DPC 每步耗时 (ms)
    time_pso: List[float] = field(default_factory=list)         # PSO 每步耗时 (ms)
    
    # 目标轨迹 (3维)
    targets: List[np.ndarray] = field(default_factory=list)

    target_temp: float = 25.0
    target_hum: float = 70.0
    target_co2: float = 800.0
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

    def __init__(self, dpc, pso, env_class, feature_order, scaler, target_indices,
                 config=None, device=None, pwm_sim_dpc=None, pwm_sim_pso=None):
        self.dpc = dpc
        self.pso = pso
        self.env_class = env_class
        self.feature_order = feature_order
        self.scaler = scaler
        self.target_indices = target_indices
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pwm_sim_dpc = pwm_sim_dpc
        self.pwm_sim_pso = pwm_sim_pso

        if config is not None:
            self._sim_steps = config.sim_steps
            self._start_idx = config.sim_start_idx
            self._target_temp = config.target_temp
            self._target_hum = config.target_hum
            self._target_co2 = config.target_co2
            self._seq_len = config.seq_len
        else:
            self._sim_steps = 300
            self._start_idx = 0
            self._target_temp = 25.0
            self._target_hum = 70.0
            self._target_co2 = 800.0
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
        pwm_sim.set_duty('humidifier', action[2])
        pwm_sim.set_duty('co2_gen', action[3])
        relay_states = pwm_sim.step()
        actual = [
            1.0 if relay_states.get('heater', False) else 0.0,
            1.0 if relay_states.get('vent', False) else 0.0,
            1.0 if relay_states.get('humidifier', False) else 0.0,
            1.0 if relay_states.get('co2_gen', False) else 0.0,
        ]
        return actual, actual

    def _step_env(self, env, actual_action, current_state, t, start_idx, X_test_p):
        """对一个控制器执行一步环境推演，返回新状态矩阵和序列"""
        target_indices = self.target_indices
        seq_len = self._seq_len
        feature_order = self.feature_order

        # 环境扰动变量索引
        tout_idx = feature_order.index('Outdoor_Temp') if 'Outdoor_Temp' in feature_order else -1
        humout_idx = feature_order.index('Outdoor_Hum') if 'Outdoor_Hum' in feature_order else -1
        solar_idx = feature_order.index('Outdoor_Solar') if 'Outdoor_Solar' in feature_order else -1

        # 逆归一化室外干扰
        curr_tout = self._inverse_transform(current_state[0, -1, tout_idx], tout_idx) if tout_idx != -1 else 15.0
        curr_hout = self._inverse_transform(current_state[0, -1, humout_idx], humout_idx) if humout_idx != -1 else 50.0
        curr_solar = self._inverse_transform(current_state[0, -1, solar_idx], solar_idx) if solar_idx != -1 else 0.0

        # 执行物理环境 (全程 Tensor 运算, 免拷贝)
        if isinstance(current_state, torch.Tensor):
            actual_action_tensor = torch.tensor(actual_action, dtype=torch.float32, device=self.device)
            next_state_real = env.step(actual_action_tensor, curr_tout, curr_hout, 400.0, curr_solar)
        else:
            next_state_real = env.step(actual_action, curr_tout, curr_hout, 400.0, curr_solar)
        
        # 将三维预测映射回避归一化空间
        dummy_state = np.zeros(len(self.scaler.scale_))
        
        # 兼容 Tensor 与 Numpy 返回
        if isinstance(next_state_real, torch.Tensor):
            dummy_state[target_indices[0]] = next_state_real[0].item()
            dummy_state[target_indices[1]] = next_state_real[1].item()
            dummy_state[target_indices[2]] = next_state_real[2].item()
        else:
            dummy_state[target_indices[0]] = next_state_real[0]
            dummy_state[target_indices[1]] = next_state_real[1]
            dummy_state[target_indices[2]] = next_state_real[2]

        next_state_norm = self.scaler.transform(dummy_state.reshape(1, -1))[0, target_indices]

        # 组装新序列 (保持 Tensor 或 Array 一致性)
        if isinstance(current_state, torch.Tensor):
            new_step = current_state[0, 1:, :].clone()
            next_real_features = torch.tensor(X_test_p[start_idx + t + 1][-1, :], dtype=torch.float32, device=self.device)
            next_real_features[target_indices] = torch.tensor(next_state_norm, dtype=torch.float32, device=self.device)
            next_real_features[0:4] = actual_action_tensor[:4]
            new_state = torch.cat([new_step, next_real_features.unsqueeze(0)], dim=0).unsqueeze(0)
            
            # 返回以便记录，如果外部期望 Numpy，这里通过返回 Tensor 来保持性能，由外部记录时转换
            if isinstance(next_state_real, torch.Tensor):
                next_state_real_ret = next_state_real.cpu().numpy()
            else:
                next_state_real_ret = next_state_real
            return next_state_real_ret, new_state
        else:
            new_step = current_state[0, 1:, :].copy()
            next_real_features = X_test_p[start_idx + t + 1][-1, :].copy()
            next_real_features[target_indices] = next_state_norm
            next_real_features[0:4] = actual_action[:4]
            new_state = np.concatenate(
                [new_step, next_real_features.reshape(1, -1)], axis=0
            ).reshape(1, seq_len, -1)

            return next_state_real, new_state

    def run(self, X_test_p, X_test_f, config=None):
        """
        运行仿真

        Returns:
            SimResult
        """
        sim_steps = self._sim_steps
        start_idx = self._start_idx
        target_indices = self.target_indices
        pwm_on = self.pwm_sim_dpc is not None

        result = SimResult(
            target_temp=self._target_temp,
            target_hum=self._target_hum,
            target_co2=self._target_co2,
            sim_steps=sim_steps,
            pwm_enabled=pwm_on
        )

        # 初始状态 (转化为 Tensor 常驻显存，加速后续循环)
        current_state_dpc = torch.tensor(X_test_p[start_idx:start_idx + 1], dtype=torch.float32, device=self.device)
        current_state_pso = torch.tensor(X_test_p[start_idx:start_idx + 1], dtype=torch.float32, device=self.device)
        
        # 提前把 Future 序列批量放进显存
        future_base_seq_tensor = torch.tensor(
            X_test_f[start_idx:start_idx + sim_steps], dtype=torch.float32, device=self.device
        )

        # 记录每步动态目标值
        for t in range(sim_steps):
            t_norm = X_test_p[start_idx + t + 1][-1, target_indices]
            dummy = np.zeros(len(self.scaler.scale_))
            dummy[target_indices[0]] = t_norm[0]
            dummy[target_indices[1]] = t_norm[1]
            dummy[target_indices[2]] = t_norm[2]
            target_real = self.scaler.inverse_transform(dummy.reshape(1, -1))[0, target_indices]
            
            # 使用配置中的静态目标值替换数据集中的预测目标值
            target_real[0] = self._target_temp
            target_real[1] = self._target_hum
            target_real[2] = self._target_co2
            
            result.targets.append(target_real)

        # 初始状态
        init_state_norm = current_state_dpc[0, -1, target_indices].cpu().numpy()
        dummy = np.zeros(len(self.scaler.scale_))
        dummy[target_indices[0]] = init_state_norm[0]
        dummy[target_indices[1]] = init_state_norm[1]
        dummy[target_indices[2]] = init_state_norm[2]
        init_state_np = self.scaler.inverse_transform(dummy.reshape(1, -1))[0, target_indices]
        
        # 创建支持 GPU 的物理环境
        init_state_tensor = torch.tensor(init_state_np, dtype=torch.float32, device=self.device)
        env_dpc = self.env_class(init_state_tensor, config)
        env_pso = self.env_class(init_state_tensor, config)

        # 重置 PWM
        if self.pwm_sim_dpc is not None:
            self.pwm_sim_dpc.reset()
        if self.pwm_sim_pso is not None:
            self.pwm_sim_pso.reset()

        pwm_label = " + PWM离散化" if pwm_on else ""
        print(f"---> 正在进行 {sim_steps} 步滚动仿真 (DPC vs PSO{pwm_label})...")

        for t in range(sim_steps):
            current_future_base = future_base_seq_tensor[t].unsqueeze(0)

            # ============ DPC 控制器 ============
            curr_dpc_temp = env_dpc.current_temp.item() if isinstance(env_dpc.current_temp, torch.Tensor) else env_dpc.current_temp

            t0 = time.time()
            opt_action_dpc, _ = self.dpc.get_optimal_action(
                current_state_dpc, current_future_base, current_temp=curr_dpc_temp
            )
            result.time_dpc.append((time.time() - t0) * 1000)
            result.actions_dpc.append(opt_action_dpc)

            # DPC PWM
            actual_dpc, pwm_dpc = self._apply_pwm(self.pwm_sim_dpc, opt_action_dpc)
            if pwm_dpc is not None:
                result.pwm_actions_dpc.append(pwm_dpc)

            # ============ PSO 控制器 ============
            curr_pso_temp = env_pso.current_temp.item() if isinstance(env_pso.current_temp, torch.Tensor) else env_pso.current_temp

            t0 = time.time()
            opt_action_pso, _ = self.pso.get_optimal_action(
                current_state_pso, current_future_base, current_temp=curr_pso_temp
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
                next_state_dpc, current_state_dpc = self._step_env(
                    env_dpc, actual_dpc, current_state_dpc, t, start_idx, X_test_p
                )
                result.history_dpc.append(next_state_dpc)
                self.dpc.update_integral(next_state_dpc[0]) # 仅用温度作为主积分项

                # PSO 环境
                next_state_pso, current_state_pso = self._step_env(
                    env_pso, actual_pso, current_state_pso, t, start_idx, X_test_p
                )
                result.history_pso.append(next_state_pso)
                self.pso.update_integral(next_state_pso[0])

            # 进度显示
            if (t + 1) % 50 == 0:
                print(f"    步 {t+1}/{sim_steps} | "
                      f"DPC_T={next_state_dpc[0]:.1f}°C (avg {np.mean(result.time_dpc[-50:]):.0f}ms) | "
                      f"PSO_T={next_state_pso[0]:.1f}°C (avg {np.mean(result.time_pso[-50:]):.0f}ms)")

        print("---> 仿真完成。")
        return result
