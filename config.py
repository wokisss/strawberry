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
    # 多目标控制：温度、湿度、CO2
    target_cols: List[str] = field(default_factory=lambda: [
        'Temperature, °C', 'Humidity, %', 'CO?, ppm'
    ])
    
    outdoor_temp_col: str = 'Outdoor_Temp'
    outdoor_solar_col: str = 'Outdoor_Solar'
    outdoor_hum_col: str = 'Outdoor_Hum'
    outdoor_wind_col: str = 'Outdoor_Wind'

    # 开关量列 (需要转为二进制)
    binary_cols: List[str] = field(default_factory=lambda: [
        'Heater', 'Ventilation', 'Fog', 'Lighting', 'Pump 1', 'Valve 1'
    ])

    # 特征顺序 (控制变量在最前面)
    feature_order_base: List[str] = field(default_factory=lambda: [
        'Heater', 'Ventilation', 'Fog', 'Lighting',
        'Temperature, °C', 'Humidity, %', 'Illumination, lx', 'CO?, ppm',
        'Hour_Sin', 'Hour_Cos'
    ])

    outdoor_cols: List[str] = field(default_factory=lambda: [
        'Outdoor_Temp', 'Outdoor_Solar', 'Outdoor_Hum', 'Outdoor_Wind'
    ])

    # 控制变量列名 (4维动作空间)
    control_cols: List[str] = field(default_factory=lambda: [
        'Heater', 'Ventilation', 'Fog', 'Lighting'
    ])

    # 时间编码列名
    time_cols: List[str] = field(default_factory=lambda: ['Hour_Sin', 'Hour_Cos'])

    # 室内光照代理列 (如果有额外的光照度列，这里不强制加Lighting是因为Lighting算控制量)
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
    # 多变量物理增益
    heater_gain_temp: float = 0.15      # 加热对温度的影响
    heater_gain_hum: float = -0.1       # 加热对相对湿度的影响 (温度升高，RH下降)
    vent_gain_temp: float = -0.3        # 通风对温度的影响
    vent_gain_hum: float = -0.4         # 通风对湿度的影响
    vent_gain_co2: float = -0.5         # 通风对CO2的影响 (排出高浓度CO2)
    fog_gain_temp: float = -0.05        # 起雾/加湿对温度的影响(蒸发降温)
    fog_gain_hum: float = 0.5           # 起雾/加湿对湿度的影响
    lighting_gain_temp: float = 0.05    # 补光灯开启附带微弱升温
    lighting_gain_co2: float = -0.6     # 补光灯开启促进光合作用，极其强烈地消耗室内 CO2

    # 多目标设定值
    target_temp: float = 25.0     # 目标温度 (°C)
    target_hum: float = 70.0      # 目标湿度 (%)
    target_co2: float = 800.0     # 目标CO2浓度 (ppm)

    dpc_lr: float = 0.2           # DPC优化器学习率
    dpc_iterations: int = 100     # DPC优化迭代次数
    
    # 追踪误差权重 (Temperature, Humidity, CO2) - 加大湿度和CO2权重逼迫优化器牺牲温度开通风
    w_track_temp: float = 20.0
    w_track_hum: float = 12.0
    w_track_co2: float = 8.0
    
    w_energy: float = 0.001       # 节能权重
    w_smooth: float = 0.1         # 平滑惩罚权重
    vent_suppress_margin: float = -100.0  # (已废除) 通风抑制边界，设为极负值永不触发

    # ======================== MDP参数 ========================
    mdp_vent_threshold: float = 28.0  # MDP 开通风阈值

    # ======================== SAC控制器参数 ========================
    sac_hidden_dim: int = 256
    sac_batch_size: int = 256
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_lr: float = 3e-4
    sac_alpha: float = 0.2     # 熵系数
    sac_target_update_interval: int = 1
    sac_train_steps: int = 50000  # 离线环境交互训练总步数
    sac_replay_size: int = 100000

    # ======================== PWM驱动参数 ========================
    pwm_cycle: int = 10       # PWM周期 (分钟)
    pwm_min_on: int = 1       # 最小吸合时间 (分钟)
    pwm_min_off: int = 1      # 最小关断时间 (分钟)

    # ======================== 仿真参数 ========================
    sim_steps: int = 300      # 仿真步数 (分钟)
    sim_start_idx: int = 0    # 仿真起始索引

    # ======================== 动态增益参数 ========================
    gain_sigma: float = 0.1       # 高斯核宽度
    gain_min: float = 0.2         # 最小增益因子
    gain_max: float = 1.0         # 最大增益因子

    # ======================== 比例兜底参数 ========================
    # 仍主要以温度作为关键安全指标
    safety_tier1_error: float = 3.0
    safety_tier1_min: float = 0.98
    safety_tier2_error: float = 1.0
    safety_tier2_min: float = 0.90
    safety_tier3_error: float = 0.3
    safety_tier3_min: float = 0.80
    safety_tier4_min: float = 0.60

    # ======================== 积分控制参数 ========================
    integral_decay: float = 0.95
    integral_clip: float = 20.0
