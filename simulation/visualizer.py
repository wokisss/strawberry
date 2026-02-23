# -*- coding: utf-8 -*-
"""
simulation/visualizer.py
--------------------------
仿真结果可视化

生成 DPC vs PSO 对比图：温度曲线 + 动作对比 + 计算耗时。
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
        绘制 DPC vs PSO 对比图 (4 子图)

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

        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

        # 颜色定义
        DPC_COLOR = '#E53935'   # 红色
        PSO_COLOR = '#1E88E5'   # 蓝色

        # --- 子图1: 温度曲线 ---
        axes[0].plot(time_axis, target_line, 'k--',
                     label=f'目标温度 ({target_temp}°C)', alpha=0.6, linewidth=1)
        axes[0].plot(time_axis, sim_result.history_pso, color=PSO_COLOR,
                     label='PSO 粒子群优化', linewidth=1.5, alpha=0.8)
        axes[0].plot(time_axis, sim_result.history_dpc, color=DPC_COLOR,
                     label='DPC 可微规划', linewidth=2.0)
        axes[0].set_ylabel("室内温度 (°C)", fontsize=12)
        axes[0].set_title("控制效果对比: DPC (梯度优化) vs PSO (粒子群优化)", fontsize=14)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # MAE
        mae_dpc = np.mean(np.abs(np.array(sim_result.history_dpc) - target_temp))
        mae_pso = np.mean(np.abs(np.array(sim_result.history_pso) - target_temp))

        axes[0].text(0.02, 0.85,
                     f"DPC MAE: {mae_dpc:.4f}\nPSO MAE: {mae_pso:.4f}",
                     bbox=dict(facecolor='white', alpha=0.8),
                     transform=axes[0].transAxes, verticalalignment='top', fontsize=10)

        # --- 子图2: 加热器动作 ---
        dpc_heater = [a[0] for a in sim_result.actions_dpc]
        pso_heater = [a[0] for a in sim_result.actions_pso]

        if pwm_on and sim_result.pwm_actions_dpc:
            pwm_dpc_h = [a[0] for a in sim_result.pwm_actions_dpc]
            pwm_pso_h = [a[0] for a in sim_result.pwm_actions_pso]
            self._draw_pwm_bars(axes[1], time_axis, pwm_dpc_h, DPC_COLOR, alpha=0.15)
            self._draw_pwm_bars(axes[1], time_axis, pwm_pso_h, PSO_COLOR, alpha=0.15)

        axes[1].plot(time_axis, dpc_heater, color=DPC_COLOR,
                     label='DPC 占空比', linewidth=1.5)
        axes[1].plot(time_axis, pso_heater, color=PSO_COLOR,
                     label='PSO 占空比', linewidth=1.5, alpha=0.8)
        axes[1].set_ylabel("加热功率", fontsize=12)
        axes[1].set_title("执行机构动作: 加热器 (Heater)", fontsize=14)
        axes[1].set_yticks([0, 0.5, 1])
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # --- 子图3: 通风动作 ---
        dpc_vent = [a[1] for a in sim_result.actions_dpc]
        pso_vent = [a[1] for a in sim_result.actions_pso]

        if pwm_on and sim_result.pwm_actions_dpc:
            pwm_dpc_v = [a[1] for a in sim_result.pwm_actions_dpc]
            pwm_pso_v = [a[1] for a in sim_result.pwm_actions_pso]
            self._draw_pwm_bars(axes[2], time_axis, pwm_dpc_v, DPC_COLOR, alpha=0.15)
            self._draw_pwm_bars(axes[2], time_axis, pwm_pso_v, PSO_COLOR, alpha=0.15)

        axes[2].plot(time_axis, dpc_vent, color=DPC_COLOR,
                     label='DPC 占空比', linewidth=1.5)
        axes[2].plot(time_axis, pso_vent, color=PSO_COLOR,
                     label='PSO 占空比', linewidth=1.5, alpha=0.8)
        axes[2].set_ylabel("通风功率", fontsize=12)
        axes[2].set_title("执行机构动作: 通风 (Ventilation)", fontsize=14)
        axes[2].set_yticks([0, 0.5, 1])
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # --- 子图4: 计算耗时对比 ---
        if sim_result.time_dpc and sim_result.time_pso:
            # 滑动平均 (窗口=10)
            window = 10
            dpc_t = np.array(sim_result.time_dpc)
            pso_t = np.array(sim_result.time_pso)

            if len(dpc_t) >= window:
                dpc_smooth = np.convolve(dpc_t, np.ones(window)/window, mode='valid')
                pso_smooth = np.convolve(pso_t, np.ones(window)/window, mode='valid')
                t_axis_smooth = range(window - 1, sim_steps)
                axes[3].plot(t_axis_smooth, dpc_smooth, color=DPC_COLOR,
                             label=f'DPC (avg {np.mean(dpc_t):.0f}ms)', linewidth=1.5)
                axes[3].plot(t_axis_smooth, pso_smooth, color=PSO_COLOR,
                             label=f'PSO (avg {np.mean(pso_t):.0f}ms)', linewidth=1.5, alpha=0.8)
            else:
                axes[3].plot(time_axis, dpc_t, color=DPC_COLOR,
                             label=f'DPC (avg {np.mean(dpc_t):.0f}ms)', linewidth=1.5)
                axes[3].plot(time_axis, pso_t, color=PSO_COLOR,
                             label=f'PSO (avg {np.mean(pso_t):.0f}ms)', linewidth=1.5, alpha=0.8)

        axes[3].set_ylabel("每步耗时 (ms)", fontsize=12)
        axes[3].set_title("计算效率对比: 每步优化耗时", fontsize=14)
        axes[3].set_xlabel("模拟时间 (分钟)", fontsize=12)
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = None
        if save:
            save_path = self.save_figure(fig)

        # 打印性能对比
        print("\n" + "=" * 60)
        print("--- 性能对比总结: DPC vs PSO ---")
        if pwm_on:
            print(f"[PWM 离散化已启用] 周期={self._pwm_cycle}分钟")
        print(f"DPC MAE: {mae_dpc:.4f}")
        print(f"PSO MAE: {mae_pso:.4f}")
        if mae_dpc < mae_pso:
            print(f"✓ DPC 优于 PSO，提升 {(mae_pso - mae_dpc) / mae_pso * 100:.1f}%")
        elif mae_pso < mae_dpc:
            print(f"✓ PSO 优于 DPC，提升 {(mae_dpc - mae_pso) / mae_dpc * 100:.1f}%")
        else:
            print(f"= 两者持平")

        if sim_result.time_dpc and sim_result.time_pso:
            avg_dpc = np.mean(sim_result.time_dpc)
            avg_pso = np.mean(sim_result.time_pso)
            print(f"\nDPC 平均耗时: {avg_dpc:.1f} ms/step")
            print(f"PSO 平均耗时: {avg_pso:.1f} ms/step")
            speedup = avg_pso / avg_dpc if avg_dpc > 0 else float('inf')
            if speedup > 1:
                print(f"DPC 快 {speedup:.1f}x")
            else:
                print(f"PSO 快 {1/speedup:.1f}x")

        print("=" * 60)

        return save_path

    def _draw_pwm_bars(self, ax, time_axis, pwm_values, color, alpha=0.3):
        """绘制 PWM ON/OFF 的柱状区域"""
        for t, val in zip(time_axis, pwm_values):
            if val > 0.5:
                ax.axvspan(t, t + 1, color=color, alpha=alpha, linewidth=0)

    def save_figure(self, fig):
        """保存图片"""
        os.makedirs(self._results_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self._results_dir, f"dpc_vs_pso_{timestamp}.png")
        fig.savefig(save_path, dpi=150)
        print(f"---> 结果图已保存至: {os.path.abspath(save_path)}")
        return save_path
