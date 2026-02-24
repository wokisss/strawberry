# -*- coding: utf-8 -*-
"""
config.py
---------
全局配置中心 (Global Configuration Center)

所有可调参数集中管理，避免硬编码散落各处。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """温室控制系统全局配置"""

    # ======================== 随机种子 ========================
    seed: int = 42

    # ======================== 数据路径 ========================
    dataset_path: str = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'
    weather_path: str = 'POWER_Point_Hourly_20250517_20250618_048d33N_025d93E_LST.csv'
    results_dir: str = 'results'
    model_save_path: str = 'best_model_A2.pth'
    cf_cache_path: str = 'data/cf_cache_baseline_h15_weather.npz'

    # ======================== 数据特征 ========================
    target_col: str = 'Temperature, °C'
    outdoor_temp_col: str = 'Outdoor_Temp'
    outdoor_solar_col: str = 'Outdoor_Solar'
    outdoor_hum_col: str = 'Outdoor_Hum'
    outdoor_wind_col: str = 'Outdoor_Wind'

    # 开关量列 (需要转为二进制)
    binary_cols: List[str] = field(default_factory=lambda: [
        'Heater', 'Ventilation', 'Lighting', 'Pump 1', 'Valve 1'
    ])

    # 特征顺序 (Heater, Ventilation 在最前面，方便控制器操控)
    feature_order_base: List[str] = field(default_factory=lambda: [
        'Heater', 'Ventilation', 'Lighting',
        'Temperature, °C', 'Humidity, %', 'Illumination, lx', 'CO?, ppm',
        'Hour_Sin', 'Hour_Cos'
    ])

    outdoor_cols: List[str] = field(default_factory=lambda: [
        'Outdoor_Temp', 'Outdoor_Solar', 'Outdoor_Hum', 'Outdoor_Wind'
    ])

    # 控制变量列名
    control_cols: List[str] = field(default_factory=lambda: ['Heater', 'Ventilation'])

    # 时间编码列名
    time_cols: List[str] = field(default_factory=lambda: ['Hour_Sin', 'Hour_Cos'])

    # 室内光照代理列 (作为太阳辐射代理)
    indoor_solar_proxy: List[str] = field(default_factory=lambda: ['Illumination, lx'])

    # ======================== 序列参数 ========================
    seq_len: int = 60       # 历史窗口长度 (分钟)
    horizon: int = 10       # 预测窗口长度 (分钟)
    train_ratio: float = 0.8

    # ======================== 物理环境参数 ========================
    k_insulation: float = 0.05    # 隔热系数 (越小保温越好)
    power_heater: float = 0.5     # 加热器最大功率 (°C/min)
    eff_vent: float = 0.1         # 通风效率
    k_solar: float = 0.01         # 太阳辐射增益系数
    noise_std: float = 0.05       # 过程噪声标准差 (已被 OU 过程替代，保留兼容)

    # ======================== 环境噪声参数 ========================
    # OU 过程噪声 (时间相关的过程扰动，替代简单白噪声)
    ou_theta: float = 0.15          # OU 回复速率 (越大噪声越快回到均值)
    ou_sigma: float = 0.05          # OU 波动强度 (°C)
    ou_mu: float = 0.0              # OU 均值

    # 传感器噪声 (测量不确定性，仅影响观测值)
    sensor_noise_std: float = 0.05  # 传感器高斯噪声标准差 (°C)

    # 执行器噪声 (功率波动，乘性噪声)
    actuator_noise_low: float = 0.95  # 执行器乘性噪声下界
    actuator_noise_high: float = 1.05 # 执行器乘性噪声上界

    # 风扰动 (随机阵风脉冲)
    wind_gust_prob: float = 0.05    # 每步发生阵风概率 (5%)
    wind_gust_magnitude: float = 0.2 # 阵风造成温度变化幅度 (°C)

    # ======================== 模型参数 ========================
    hidden_dim: int = 32
    batch_size: int = 256
    num_epochs: int = 30
    learning_rate: float = 0.001
    lambda_trend: float = 0.3     # 趋势惩罚权重

    # ======================== DPC控制器参数 ========================
    heater_gain: float = 0.15     # 加热物理增益 (校准后: 让优化器输出更高的加热功率)
    vent_gain: float = -0.3       # 通风物理增益 (归一化空间)
    target_temp: float = 25.0     # 目标温度 (°C)
    dpc_lr: float = 0.2           # DPC优化器学习率
    dpc_iterations: int = 100     # DPC优化迭代次数
    w_track: float = 20.0         # 跟踪误差权重
    w_energy: float = 0.001       # 节能权重 (降低以给优化器更多自由度)
    w_smooth: float = 0.1         # 平滑惩罚权重 (抑制动作跳变)
    vent_suppress_margin: float = 2.0  # 通风抑制边界: temp < target+2°C 时禁止通风

    # ======================== MDP参数 ========================
    mdp_vent_threshold: float = 28.0  # MDP 开通风阈值

    # ======================== PSO控制器参数 ========================
    pso_n_particles: int = 30         # 粒子数量
    pso_n_generations: int = 50       # 迭代代数
    pso_w_inertia: float = 0.7        # 惯性权重
    pso_c1: float = 1.5               # 认知因子 (向个体最优靠拢)
    pso_c2: float = 1.5               # 社会因子 (向全局最优靠拢)

    # ======================== PWM驱动参数 ========================
    pwm_cycle: int = 10       # PWM周期 (分钟)
    pwm_min_on: int = 1       # 最小吸合时间 (分钟)
    pwm_min_off: int = 1      # 最小关断时间 (分钟)

    # ======================== 仿真参数 ========================
    sim_steps: int = 300      # 仿真步数 (分钟)
    sim_start_idx: int = 0    # 仿真起始索引

    # ======================== 动态增益参数 ========================
    gain_sigma: float = 0.1       # 高斯核宽度 (动态增益计算)
    gain_min: float = 0.2         # 最小增益因子
    gain_max: float = 1.0         # 最大增益因子

    # ======================== 比例兜底参数 ========================
    safety_tier1_error: float = 3.0    # 一级兜底: error >= 3°C
    safety_tier1_min: float = 0.98     # 一级兜底: 最低加热功率
    safety_tier2_error: float = 1.0    # 二级兜底: error >= 1°C
    safety_tier2_min: float = 0.90     # 二级兜底: 最低加热功率
    safety_tier3_error: float = 0.3    # 三级兜底: error >= 0.3°C
    safety_tier3_min: float = 0.80     # 三级兜底: 最低加热功率
    safety_tier4_min: float = 0.60     # 四级兜底: error > 0, 最低加热功率

    # ======================== 积分控制参数 ========================
    integral_decay: float = 0.95       # 积分衰减因子
    integral_clip: float = 20.0        # 积分防饱和上限
