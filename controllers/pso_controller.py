# -*- coding: utf-8 -*-
"""
controllers/pso_controller.py
-------------------------------
粒子群优化控制器 (Particle Swarm Optimization MPC Controller)

无梯度的模型预测控制: 使用粒子群算法在动作空间 [0,1]² 内搜索最优动作。
与 DPC 使用完全相同的预测模型和 Loss 函数，唯一区别是优化方法。
"""

import numpy as np
import torch
import time


class PSOController:
    """
    PSO 控制器

    核心流程:
        1. 随机初始化 N 个粒子，每个粒子 = [heater, vent]
        2. 每个粒子调用 DecisionControlModel 前向传播计算 Loss
        3. 更新个体最优 (pbest) 和全局最优 (gbest)
        4. 按速度公式更新粒子位置
        5. 物理互斥 + 比例安全兜底 (与 DPC 完全一致)

    Args:
        model: DecisionControlModel (含物理增益)
        scaler: MinMaxScaler
        target_idx: 温度列索引
        future_indices: 未来特征索引
        config: Config 对象
    """

    def __init__(self, model, scaler, target_idx, future_indices,
                 config=None, horizon=10, target_temp=25.0):
        self.model = model
        self.scaler = scaler
        self.target_idx = target_idx
        self.future_indices = future_indices

        if config is not None:
            self.horizon = config.horizon
            self.target_temp = config.target_temp
            self._w_track = config.w_track
            self._w_energy = config.w_energy
            self._w_smooth = config.w_smooth
            # PSO 参数
            self._n_particles = config.pso_n_particles
            self._n_generations = config.pso_n_generations
            self._w_inertia = config.pso_w_inertia
            self._c1 = config.pso_c1
            self._c2 = config.pso_c2
            # 安全兜底
            self._safety = {
                'tier1_error': config.safety_tier1_error,
                'tier1_min': config.safety_tier1_min,
                'tier2_error': config.safety_tier2_error,
                'tier2_min': config.safety_tier2_min,
                'tier3_error': config.safety_tier3_error,
                'tier3_min': config.safety_tier3_min,
                'tier4_min': config.safety_tier4_min,
            }
            self._integral_decay = config.integral_decay
            self._integral_clip = config.integral_clip
            self._vent_suppress_margin = getattr(config, 'vent_suppress_margin', 2.0)
        else:
            self.horizon = horizon
            self.target_temp = target_temp
            self._w_track = 20.0
            self._w_energy = 0.005
            self._w_smooth = 0.0
            self._n_particles = 30
            self._n_generations = 50
            self._w_inertia = 0.7
            self._c1 = 1.5
            self._c2 = 1.5
            self._safety = {
                'tier1_error': 3.0, 'tier1_min': 0.98,
                'tier2_error': 1.0, 'tier2_min': 0.90,
                'tier3_error': 0.3, 'tier3_min': 0.80,
                'tier4_min': 0.60,
            }
            self._integral_decay = 0.95
            self._integral_clip = 20.0
            self._vent_suppress_margin = 2.0

        # 设备
        self._device = next(model.parameters()).device

        # 积分误差
        self.integral_error = 0.0

        # 目标温度归一化值
        dummy = np.zeros((1, len(scaler.scale_)))
        dummy[0, target_idx] = self.target_temp
        self.target_temp_norm = scaler.transform(dummy)[0, target_idx]

        # 上一步动作记录
        self.last_action = [0, 0]
        self.last_action_continuous = np.array([0.0, 0.0])

        # 性能统计
        self.last_step_time = 0.0

    def _evaluate_particles(self, particles, current_past_tensor, current_future_base):
        """
        批量评估所有粒子的适应度 (Loss)

        Args:
            particles: (N, 2) ndarray，每行 = [heater, vent]
            current_past_tensor: (1, seq_len, feat_dim)
            current_future_base: (1, horizon, future_dim)

        Returns:
            losses: (N,) ndarray，每个粒子的 Loss 值
        """
        N = particles.shape[0]

        with torch.no_grad():
            # 批量前向传播: 复制输入 N 份
            past_batch = current_past_tensor.repeat(N, 1, 1)           # (N, seq, feat)
            future_batch = current_future_base.repeat(N, 1, 1)         # (N, horizon, fdim)

            # 替换控制动作
            actions_tensor = torch.FloatTensor(particles).to(self._device)  # (N, 2)
            u_heater = actions_tensor[:, 0:1].unsqueeze(1).repeat(1, self.horizon, 1)  # (N, H, 1)
            u_vent = actions_tensor[:, 1:2].unsqueeze(1).repeat(1, self.horizon, 1)    # (N, H, 1)

            f_weather = future_batch[:, :, 2:]   # (N, H, fdim-2)
            x_future_optim = torch.cat([u_heater, u_vent, f_weather], dim=2)

            # 前向预测
            pred_norm = self.model(past_batch, x_future_optim,
                                   target_temp_norm=self.target_temp_norm)

            # 计算 Loss
            track_loss = torch.mean((pred_norm - self.target_temp_norm) ** 2, dim=1)  # (N,)
            energy_loss = torch.mean(torch.abs(actions_tensor), dim=1)                  # (N,)

            # 平滑惩罚
            prev_u = torch.FloatTensor(self.last_action_continuous).to(self._device)
            smooth_loss = torch.mean((actions_tensor - prev_u.unsqueeze(0)) ** 2, dim=1)  # (N,)

            total_loss = (self._w_track * track_loss
                          + self._w_energy * energy_loss
                          + self._w_smooth * smooth_loss)

            return total_loss.cpu().numpy()

    def get_optimal_action(self, current_past_tensor, current_future_base, current_temp=None):
        """
        基于粒子群优化的在线动作搜索

        Args:
            current_past_tensor: (1, seq_len, feat_dim)
            current_future_base: (1, horizon, future_dim)
            current_temp: 当前实际温度 (°C)

        Returns:
            best_action: [heater, vent] 最优动作
            best_loss: 最优 loss 值
        """
        t_start = time.time()
        N = self._n_particles
        K = self._n_generations

        # 冻结模型
        original_mode = self.model.training
        self.model.eval()

        try:
            # ======== 初始化粒子群 ========
            # 位置: 随机撒在 [0, 1]²
            positions = np.random.uniform(0.0, 1.0, size=(N, 2))

            # 智能初始化: 将一部分粒子放在温度误差驱动的初始点附近
            if current_temp is not None:
                temp_error = self.target_temp - current_temp
                if temp_error > 5.0:
                    seed_action = [0.95, 0.0]
                elif temp_error > 2.0:
                    seed_action = [0.85, 0.0]
                elif temp_error > 0.5:
                    seed_action = [0.7, 0.0]
                elif temp_error > 0:
                    seed_action = [0.5, 0.0]
                else:
                    seed_action = [0.1, 0.3]

                # 前 25% 的粒子集中在种子点附近 (高斯扰动)
                n_seeded = max(1, N // 4)
                positions[:n_seeded] = np.clip(
                    np.array(seed_action) + np.random.normal(0, 0.1, size=(n_seeded, 2)),
                    0.0, 1.0
                )
                # 放一个粒子精确在上一步动作位置
                positions[0] = np.clip(self.last_action_continuous, 0.0, 1.0)

            # 速度: 初始小随机速度
            velocities = np.random.uniform(-0.1, 0.1, size=(N, 2))

            # 个体最优和全局最优
            pbest_positions = positions.copy()
            pbest_losses = np.full(N, float('inf'))
            gbest_position = positions[0].copy()
            gbest_loss = float('inf')

            # ======== 迭代优化 ========
            for gen in range(K):
                # 1. 评估所有粒子
                losses = self._evaluate_particles(
                    positions, current_past_tensor, current_future_base
                )

                # 2. 更新个体最优
                improved = losses < pbest_losses
                pbest_positions[improved] = positions[improved]
                pbest_losses[improved] = losses[improved]

                # 3. 更新全局最优
                gen_best_idx = np.argmin(losses)
                if losses[gen_best_idx] < gbest_loss:
                    gbest_loss = losses[gen_best_idx]
                    gbest_position = positions[gen_best_idx].copy()

                # 4. 更新速度和位置 (标准 PSO 公式)
                r1 = np.random.uniform(0, 1, size=(N, 2))
                r2 = np.random.uniform(0, 1, size=(N, 2))

                velocities = (self._w_inertia * velocities
                              + self._c1 * r1 * (pbest_positions - positions)
                              + self._c2 * r2 * (gbest_position - positions))

                # 速度限幅
                velocities = np.clip(velocities, -0.3, 0.3)

                # 更新位置
                positions = positions + velocities
                positions = np.clip(positions, 0.0, 1.0)

        finally:
            self.model.train(original_mode)

        # ======== 后处理 (与 DPC 完全一致) ========
        final_heater = gbest_position[0]
        final_vent = gbest_position[1]

        # 更新连续状态
        self.last_action_continuous = gbest_position.copy()

        # 物理互斥
        if final_heater > 0.1 and final_vent > 0.1:
            if final_heater > final_vent:
                final_vent = 0.0
            else:
                final_heater = 0.0

        # 比例安全兜底 (与 DPC 完全一致)
        if current_temp is not None:
            error = self.target_temp - current_temp
            # 通风抑制区 (与 DPC 一致)
            if error > -self._vent_suppress_margin:
                final_vent = 0.0

            s = self._safety
            if error >= s['tier1_error']:
                final_heater = max(final_heater, s['tier1_min'])
            elif error >= s['tier2_error']:
                final_heater = max(final_heater, s['tier2_min'])
            elif error >= s['tier3_error']:
                final_heater = max(final_heater, s['tier3_min'])
            elif error > 0:
                final_heater = max(final_heater, s['tier4_min'])

        best_action = [final_heater, final_vent]
        self.last_action = best_action.copy()
        self.last_step_time = time.time() - t_start
        return best_action, gbest_loss

    def update_integral(self, current_temp):
        """更新积分误差 (与 DPC 一致)"""
        error = self.target_temp - current_temp
        self.integral_error *= self._integral_decay
        self.integral_error += error
        self.integral_error = np.clip(
            self.integral_error, -self._integral_clip, self._integral_clip
        )
