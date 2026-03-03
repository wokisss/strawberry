# -*- coding: utf-8 -*-
"""
simulation/visualizer.py
--------------------------
仿真结果可视化

生成 DPC vs SAC 对比图：温度曲线 + 动作对比 + 计算耗时。
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
        绘制 DPC vs SAC 对比图 (3 子图: Temperature, Humidity, CO2)

        Args:
            sim_result: SimResult 对象
            save: 是否保存到文件

        Returns:
            str: 保存路径 (若 save=True)
        """
        sim_steps = sim_result.sim_steps
        time_axis = range(sim_steps)
        pwm_on = sim_result.pwm_enabled

        hist_dpc = np.array(sim_result.history_dpc) # (steps, 3)
        hist_sac = np.array(sim_result.history_sac) # (steps, 3)
        targets = np.array(sim_result.targets)      # (steps, 3)
        
        # 动作历史
        acts_dpc = np.array(sim_result.actions_dpc) # (steps, 4)
        acts_sac = np.array(sim_result.actions_sac) # (steps, 4)

        # 提前计算 MAE 用于图例展示
        mae_temp_dpc = np.mean(np.abs(hist_dpc[:, 0] - targets[:, 0]))
        mae_temp_sac = np.mean(np.abs(hist_sac[:, 0] - targets[:, 0]))
        mae_hum_dpc = np.mean(np.abs(hist_dpc[:, 1] - targets[:, 1]))
        mae_hum_sac = np.mean(np.abs(hist_sac[:, 1] - targets[:, 1]))
        mae_co2_dpc = np.mean(np.abs(hist_dpc[:, 2] - targets[:sim_steps, 2]))
        mae_co2_sac = np.mean(np.abs(hist_sac[:, 2] - targets[:sim_steps, 2]))

        # 颜色定义
        DPC_COLOR = '#E53935'   # 红色
        SAC_COLOR = '#1E88E5'   # 蓝色

        fig, axes = plt.subplots(5, 1, figsize=(16, 22), sharex=True, gridspec_kw={'height_ratios': [3, 3, 3, 2, 2]})

        # --- 子图1: 温度曲线 ---
        axes[0].plot(time_axis, targets[:, 0], 'k--',
                     label=f'目标温度 ({sim_result.target_temp}°C)', alpha=0.6, linewidth=1)
        axes[0].plot(time_axis, hist_sac[:, 0], color=SAC_COLOR,
                     label=f'SAC (MAE: {mae_temp_sac:.2f}°C)', linewidth=1.5, alpha=0.8)
        axes[0].plot(time_axis, hist_dpc[:, 0], color=DPC_COLOR,
                     label=f'DPC (MAE: {mae_temp_dpc:.2f}°C)', linewidth=2.0)
        axes[0].set_ylabel("室内温度 (°C)", fontsize=12)
        axes[0].set_title("Temperature: DPC (Gradient) vs SAC (Soft Actor-Critic)", fontsize=14)
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # --- 子图2: 湿度曲线 ---
        axes[1].plot(time_axis, targets[:, 1], 'k--',
                     label=f'目标湿度 ({sim_result.target_hum}%)', alpha=0.6, linewidth=1)
        axes[1].plot(time_axis, hist_sac[:, 1], color=SAC_COLOR,
                     label=f'SAC (MAE: {mae_hum_sac:.2f}%)', linewidth=1.5, alpha=0.8)
        axes[1].plot(time_axis, hist_dpc[:, 1], color=DPC_COLOR,
                     label=f'DPC (MAE: {mae_hum_dpc:.2f}%)', linewidth=2.0)
        axes[1].set_ylabel("相对湿度 (%)", fontsize=12)
        axes[1].set_title("Humidity", fontsize=14)
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # --- 子图3: CO2曲线 ---
        axes[2].plot(time_axis, targets[:sim_steps, 2], 'k--',
                     label=f'目标 CO2 ({sim_result.target_co2}ppm)', alpha=0.6, linewidth=1)
        axes[2].plot(time_axis, hist_sac[:, 2], color=SAC_COLOR,
                     label=f'SAC (MAE: {mae_co2_sac:.2f}ppm)', linewidth=1.5, alpha=0.8)
        axes[2].plot(time_axis, hist_dpc[:, 2], color=DPC_COLOR,
                     label=f'DPC (MAE: {mae_co2_dpc:.2f}ppm)', linewidth=2.0)
        axes[2].set_ylabel("CO2 浓度 (ppm)", fontsize=12)
        axes[2].set_title("CO2 Level", fontsize=14)
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        # --- 子图4: DPC 动作曲线 ---
        acts_labels = ['Heater', 'Ventilation', 'Fog', 'Lighting']
        acts_colors = ['#FFA000', '#00ACC1', '#8E24AA', '#43A047']
        for i in range(4):
            axes[3].plot(time_axis, acts_dpc[:, i], color=acts_colors[i], label=acts_labels[i], alpha=0.8, linewidth=1.5)
        axes[3].set_ylabel("DPC 输出 (0-1)", fontsize=12)
        axes[3].set_title("DPC Continuous Actions", fontsize=14)
        axes[3].legend(loc='upper right', ncol=4, fontsize=9)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim([-0.05, 1.05])
        
        # --- 子图5: SAC 动作曲线 ---
        for i in range(4):
            axes[4].plot(time_axis, acts_sac[:, i], color=acts_colors[i], label=acts_labels[i], alpha=0.8, linewidth=1.5)
        axes[4].set_ylabel("SAC 输出 (0-1)", fontsize=12)
        axes[4].set_title("SAC Continuous Actions", fontsize=14)
        axes[4].set_xlabel("模拟时间 (分钟)", fontsize=12)
        axes[4].legend(loc='upper right', ncol=4, fontsize=9)
        axes[4].grid(True, alpha=0.3)
        axes[4].set_ylim([-0.05, 1.05])

        plt.tight_layout()

        save_path = None
        if save:
            save_path = self.save_figure(fig)

        # 打印性能对比
        print("\n" + "=" * 60)
        print("--- 多变量控制对比总结: DPC vs SAC ---")
        if pwm_on:
            print(f"[PWM 离散化已启用] 周期={self._pwm_cycle}分钟")

        print(f"[{'DPC':<5}] MAE Temp: {mae_temp_dpc:.2f}°C, Hum: {mae_hum_dpc:.2f}%, CO2: {mae_co2_dpc:.2f}ppm")
        print(f"[{'SAC':<5}] MAE Temp: {mae_temp_sac:.2f}°C, Hum: {mae_hum_sac:.2f}%, CO2: {mae_co2_sac:.2f}ppm")

        if sim_result.time_dpc and sim_result.time_sac:
            avg_dpc = np.mean(sim_result.time_dpc)
            avg_sac = np.mean(sim_result.time_sac)
            print(f"\nDPC 平均耗时: {avg_dpc:.1f} ms/step")
            print(f"SAC 平均耗时: {avg_sac:.1f} ms/step")
            speedup = avg_sac / avg_dpc if avg_dpc > 0 else float('inf')
            if speedup > 1:
                print(f"DPC 快 {speedup:.1f}x")
            else:
                print(f"SAC 快 {1/speedup:.1f}x")

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
        save_path = os.path.join(self._results_dir, f"dpc_vs_sac_{timestamp}.png")
        fig.savefig(save_path, dpi=150)
        print(f"---> 结果图已保存至: {os.path.abspath(save_path)}")
        return save_path
