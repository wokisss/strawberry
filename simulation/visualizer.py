# -*- coding: utf-8 -*-
"""
simulation/visualizer.py
--------------------------
仿真结果可视化

生成温度曲线 + 动作对比图，支持 PWM 离散化展示。
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    仿真结果可视化

    Args:
        config: Config 对象 (可选)
    """

    def __init__(self, config=None):
        if config is not None:
            self._results_dir = config.results_dir
            self._target_temp = config.target_temp
            self._pwm_cycle = config.pwm_cycle
        else:
            self._results_dir = 'results'
            self._target_temp = 25.0
            self._pwm_cycle = 10

    def plot_comparison(self, sim_result, save=True):
        """
        绘制 MPC vs MDP 对比图

        Args:
            sim_result: SimResult 对象
            save: 是否保存到文件

        Returns:
            str: 保存路径 (若 save=True)
        """
        sim_steps = sim_result.sim_steps
        target_temp = sim_result.target_temp
        time_axis = range(sim_steps)
        target_line = [target_temp] * sim_steps
        pwm_on = sim_result.pwm_enabled

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

        # --- 子图1: 温度曲线 ---
        axes[0].plot(time_axis, target_line, 'k--', label=f'目标温度 ({target_temp}°C)', alpha=0.6)
        axes[0].plot(time_axis, sim_result.history_mdp, color='gray', linestyle=':',
                     label='MDP 控制 (规则)', linewidth=1.5)
        axes[0].plot(time_axis, sim_result.history_mpc, color='red',
                     label='可微规划 (DPC+PWM)' if pwm_on else '可微规划 (DPC)', linewidth=2.0)
        axes[0].set_ylabel("室内温度 (°C)", fontsize=12)
        axes[0].set_title("控制效果对比: 温度保持", fontsize=14)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # MAE 统计
        mae_mpc = np.mean(np.abs(np.array(sim_result.history_mpc) - target_temp))
        mae_mdp = np.mean(np.abs(np.array(sim_result.history_mdp) - target_temp))

        # --- 子图2: 加热器动作 ---
        mpc_heater = [a[0] for a in sim_result.actions_mpc]
        mdp_heater = [a[0] for a in sim_result.actions_mdp]

        axes[1].step(time_axis, mdp_heater, color='gray', linestyle=':',
                     label='MDP Heater (0/1)', where='post', alpha=0.7)

        if pwm_on:
            # PWM 离散动作 (实际继电器状态)
            pwm_heater = [a[0] for a in sim_result.pwm_actions_mpc]
            self._draw_pwm_bars(axes[1], time_axis, pwm_heater, 'red', alpha=0.25)
            axes[1].plot(time_axis, mpc_heater, color='red',
                         label=f'DPC 连续占空比', linewidth=1.5)
            # PWM 周期分隔线
            for x in range(0, sim_steps, self._pwm_cycle):
                axes[1].axvline(x, color='red', alpha=0.08, linewidth=0.5)
            # 图例
            relay_patch = mpatches.Patch(color='red', alpha=0.25, label=f'PWM 继电器 ON (周期={self._pwm_cycle}min)')
            handles, labels = axes[1].get_legend_handles_labels()
            handles.append(relay_patch)
            axes[1].legend(handles=handles, loc='upper right', fontsize=9)
        else:
            axes[1].plot(time_axis, mpc_heater, color='red',
                         label='可微规划 PWM (0-100%)', linewidth=1.5)
            axes[1].fill_between(time_axis, mpc_heater, color='red', alpha=0.3)
            axes[1].legend(loc='upper right')

        axes[1].set_ylabel("加热功率/概率", fontsize=12)
        axes[1].set_title("执行机构动作: 加热器 (Heater)", fontsize=14)
        axes[1].set_yticks([0, 0.5, 1])
        axes[1].grid(True, alpha=0.3)

        # MAE 文本
        axes[1].text(0.02, 0.85,
                     f"MDP MAE: {mae_mdp:.2f}\nDPC MAE: {mae_mpc:.2f}",
                     bbox=dict(facecolor='white', alpha=0.8),
                     transform=axes[1].transAxes, verticalalignment='top', fontsize=10)

        # --- 子图3: 通风动作 ---
        mpc_vent = [a[1] for a in sim_result.actions_mpc]
        mdp_vent = [a[1] for a in sim_result.actions_mdp]

        axes[2].step(time_axis, mdp_vent, color='gray', linestyle=':',
                     label='MDP Vent (0/1)', where='post', alpha=0.7)

        if pwm_on:
            pwm_vent = [a[1] for a in sim_result.pwm_actions_mpc]
            self._draw_pwm_bars(axes[2], time_axis, pwm_vent, 'blue', alpha=0.25)
            axes[2].plot(time_axis, mpc_vent, color='blue',
                         label='DPC 连续占空比', linewidth=1.5)
            for x in range(0, sim_steps, self._pwm_cycle):
                axes[2].axvline(x, color='blue', alpha=0.08, linewidth=0.5)
            relay_patch = mpatches.Patch(color='blue', alpha=0.25, label=f'PWM 继电器 ON (周期={self._pwm_cycle}min)')
            handles, labels = axes[2].get_legend_handles_labels()
            handles.append(relay_patch)
            axes[2].legend(handles=handles, loc='upper right', fontsize=9)
        else:
            axes[2].plot(time_axis, mpc_vent, color='blue',
                         label='可微规划 PWM (0-100%)', linewidth=1.5)
            axes[2].fill_between(time_axis, mpc_vent, color='blue', alpha=0.3)
            axes[2].legend(loc='upper right')

        axes[2].set_ylabel("通风功率/概率", fontsize=12)
        axes[2].set_title("执行机构动作: 通风 (Ventilation)", fontsize=14)
        axes[2].set_yticks([0, 0.5, 1])
        axes[2].set_xlabel("模拟时间 (分钟)", fontsize=12)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = None
        if save:
            save_path = self.save_figure(fig)

        # 打印性能对比
        print("\n" + "=" * 60)
        print("--- 性能对比总结 ---")
        if pwm_on:
            print(f"[PWM 离散化已启用] 周期={self._pwm_cycle}分钟")
        print(f"MDP MAE: {mae_mdp:.4f}")
        print(f"DPC MAE: {mae_mpc:.4f}")
        if mae_mpc < mae_mdp:
            print(f"✓ DPC 优于 MDP，提升 {(mae_mdp - mae_mpc) / mae_mdp * 100:.1f}%")
        else:
            print(f"✗ DPC 未能超越 MDP，需进一步调优")
        print("=" * 60)

        return save_path

    def _draw_pwm_bars(self, ax, time_axis, pwm_values, color, alpha=0.3):
        """绘制 PWM ON/OFF 的柱状区域 (ON=1 时涂色)"""
        for t, val in zip(time_axis, pwm_values):
            if val > 0.5:  # ON
                ax.axvspan(t, t + 1, color=color, alpha=alpha, linewidth=0)

    def save_figure(self, fig):
        """保存图片到 results 目录"""
        os.makedirs(self._results_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self._results_dir, f"mpc_pwm_{timestamp}.png")
        fig.savefig(save_path, dpi=150)
        print(f"---> 结果图已保存至: {os.path.abspath(save_path)}")
        return save_path
