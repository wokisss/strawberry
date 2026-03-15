# -*- coding: utf-8 -*-
"""
data_processing/processor.py
-------------------------------
数据加载、清洗、特征工程、序列化

将 main() 中 L600-730 的数据处理逻辑封装为可复用的 DataProcessor 类。
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# ======================== ODE 导数特征提取器 ========================
class _ODEF(nn.Module):
    """Neural ODE 动力函数: 学习数据的时间变化率"""
    def __init__(self, input_dim, hidden_dim=64):
        super(_ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, t, y):
        t_vec = torch.ones_like(y[..., :1]) * t
        return self.net(torch.cat([y, t_vec], dim=-1))


def _generate_ode_derivatives(df, target_cols, train_epochs=300):
    """
    使用 Neural ODE 为指定列计算物理变化率特征
    
    不改变原始数据的值，而是新增 *_Deriv 列表示每分钟的变化率。
    """
    print(f"---> [ODE] 正在计算物理导数特征: {target_cols}...")
    
    df_clean = df[target_cols].dropna()
    scaler_ode = MinMaxScaler()
    data_np = scaler_ode.fit_transform(df_clean.values)
    
    timestamps = (df_clean.index - df_clean.index[0]).total_seconds() / 3600.0
    t_tensor = torch.FloatTensor(timestamps.values if hasattr(timestamps, 'values') else timestamps).reshape(-1, 1)
    y_tensor = torch.FloatTensor(data_np)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_tensor, y_tensor = t_tensor.to(device), y_tensor.to(device)
    
    ode_func = _ODEF(input_dim=len(target_cols)).to(device)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=0.02)
    
    ode_func.train()
    for _ in range(train_epochs):
        optimizer.zero_grad()
        dy_dt_target = (y_tensor[1:] - y_tensor[:-1]) / (t_tensor[1:] - t_tensor[:-1])
        t_mid = (t_tensor[1:] + t_tensor[:-1]) / 2
        y_mid = (y_tensor[1:] + y_tensor[:-1]) / 2
        loss = torch.mean((ode_func(t_mid, y_mid) - dy_dt_target) ** 2)
        loss.backward()
        optimizer.step()
    
    ode_func.eval()
    full_index = pd.date_range(start=df_clean.index[0], end=df_clean.index[-1], freq='1min')
    df_linear_temp = df[target_cols].reindex(full_index).interpolate(method='linear').ffill().bfill()
    y_interp_tensor = torch.FloatTensor(scaler_ode.transform(df_linear_temp.values)).to(device)
    t_full = torch.FloatTensor(
        (full_index - df_clean.index[0]).total_seconds().values / 3600.0
    ).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        derivs = ode_func(t_full, y_interp_tensor).cpu().numpy()
    
    derivs_restored = derivs / (scaler_ode.scale_ + 1e-8)
    new_cols = [f"{c}_Deriv" for c in target_cols]
    df_derivs = pd.DataFrame(derivs_restored, index=full_index, columns=new_cols)
    
    print(f"    [ODE] 成功生成 {len(new_cols)} 个导数特征: {new_cols}")
    return df_derivs


class DataProcessor:
    """
    数据处理器

    职责:
        1. 加载 CSV 数据集
        2. 清洗 (列名、二值化、数值转换、重采样)
        3. 合并外部天气数据 (NASA POWER)
        4. 添加时间编码
        5. 特征排序 + 归一化
        6. 滑窗序列化 + 训练/测试划分

    Args:
        config: Config 对象
    """

    def __init__(self, config):
        self.cfg = config
        self.scaler = MinMaxScaler()
        self.feature_order = None
        self.future_indices = None
        self.target_indices = None

    def load_and_preprocess(self):
        """
        加载并预处理数据集

        Returns:
            pd.DataFrame: 清洗后的数据
        """
        cfg = self.cfg

        if not os.path.exists(cfg.dataset_path):
            raise FileNotFoundError(f"未找到数据集: {cfg.dataset_path}")

        df = pd.read_csv(
            cfg.dataset_path, encoding='latin1', sep=';', decimal=',',
            parse_dates=['Timestamp'], dayfirst=True, index_col='Timestamp', engine='python'
        )

        # 清洗列名
        df.columns = [c.replace('"', '').strip() for c in df.columns]

        # 处理开关量
        for col in cfg.binary_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: 1 if str(x).lower() in ['on', 'yes', '1'] else 0
                )

        # 数值转换 + 重采样
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.resample('1min').mean().interpolate().ffill().bfill()

        print(f"---> 数据预处理完成，维度: {df.shape}")
        return df

    def merge_weather(self, df):
        """
        合并 NASA POWER 外部天气数据

        Args:
            df: 主数据 DataFrame

        Returns:
            pd.DataFrame: 合并天气后的数据
        """
        cfg = self.cfg

        if not os.path.exists(cfg.weather_path):
            print(f"---> [警告] 未找到外部天气文件: {cfg.weather_path}")
            return df

        print(f"---> 正在加载外部天气数据: {cfg.weather_path}")
        df_weather = pd.read_csv(cfg.weather_path, skiprows=12)

        # 构建时间戳
        df_weather['Timestamp'] = pd.to_datetime(
            df_weather['YEAR'].astype(str) + '-' +
            df_weather['MO'].astype(str).str.zfill(2) + '-' +
            df_weather['DY'].astype(str).str.zfill(2) + ' ' +
            df_weather['HR'].astype(str).str.zfill(2) + ':00:00'
        )
        df_weather = df_weather.set_index('Timestamp')

        # 重命名列
        df_weather = df_weather.rename(columns={
            'T2M': cfg.outdoor_temp_col,
            'ALLSKY_SFC_SW_DWN': cfg.outdoor_solar_col,
            'RH2M': cfg.outdoor_hum_col,
            'WS2M': cfg.outdoor_wind_col,
        })

        # 物理启发式重采样与插值 (Physical-informed Interpolation)
        weather_cols = cfg.outdoor_cols
        df_weather = df_weather[weather_cols].resample('1min').asfreq()
        
        try:
            # 1. 温湿度使用 Cubic (三次样条) 插值，保证平滑过度
            df_weather[cfg.outdoor_temp_col] = df_weather[cfg.outdoor_temp_col].interpolate(method='cubic').ffill().bfill()
            df_weather[cfg.outdoor_hum_col] = df_weather[cfg.outdoor_hum_col].interpolate(method='cubic').ffill().bfill()
            
            # 2. 短波辐射使用 PCHIP (保形分段三次插值)，防止强拟合导致日出日落出现负值
            df_weather[cfg.outdoor_solar_col] = df_weather[cfg.outdoor_solar_col].interpolate(method='pchip').clip(lower=0.0).ffill().bfill()
            
            # 3. 风速使用 PCHIP 提取基础风流，然后注入高斯噪声模拟阵风突变
            base_wind = df_weather[cfg.outdoor_wind_col].interpolate(method='pchip').clip(lower=0.0).ffill().bfill()
            gust_noise = np.random.normal(0, 0.4, size=len(base_wind)) # STD=0.4m/s 的阵风
            df_weather[cfg.outdoor_wind_col] = np.clip(base_wind + gust_noise, a_min=0.0, a_max=None)
            print("    [Info] 成功启用高阶物理气象插值 (Cubic/PCHIP + 阵风注入)")
            
        except ImportError:
            # 若未安装 scipy 则回退到原始无脑线性插值
            print("    [Warning] 缺少 scipy 库，回退到普通线性插值。建议 pip install scipy")
            df_weather = df_weather.interpolate(method='linear').ffill().bfill()

        # 合并
        df_weather = df_weather.reindex(df.index, method='ffill')
        for col in df_weather.columns:
            df[col] = df_weather[col]
        df = df.ffill().bfill()

        print(f"    成功合并 {len(weather_cols)} 个天气特征: {weather_cols}")
        return df

    def add_time_encoding(self, df):
        """添加时间位置编码 (sin/cos)"""
        hour_float = df.index.hour + df.index.minute / 60.0
        df['Hour_Sin'] = np.sin(2 * np.pi * hour_float / 24.0)
        df['Hour_Cos'] = np.cos(2 * np.pi * hour_float / 24.0)
        print("---> 添加时间编码特征: Hour_Sin, Hour_Cos")
        return df

    def add_energy_features(self, df):
        """
        [新增] 添加能量累积特征
        
        计算各物理控制量过去60分钟的滚动累积值。
        让模型理解"加热器已连续开了多久"这类时序惯性信息。
        """
        cfg = self.cfg
        energy_window = 60  # 60分钟滚动窗口

        # 只对实际存在的控制列计算
        control_cols_available = [c for c in cfg.control_cols if c in df.columns]
        
        if control_cols_available:
            new_cols = []
            for col in control_cols_available:
                new_col = f'{col}_Energy_60m'
                df[new_col] = df[col].rolling(window=energy_window, min_periods=1).sum()
                new_cols.append(new_col)
            df.fillna(0, inplace=True)
            print(f"---> [能量特征] 已添加 {len(new_cols)} 个累积特征: {new_cols}")
        
        return df

    def add_ode_derivatives(self, df):
        """
        [新增] 使用 Neural ODE 计算户外气象列的物理变化率特征
        
        为 Outdoor_Temp、Outdoor_Solar 添加 *_Deriv 列，
        帮助模型理解"温度正在上升还是下降"这类动态趋势信息。
        """
        ode_target_cols = ['Outdoor_Temp', 'Outdoor_Solar']
        ode_target_cols = [c for c in ode_target_cols if c in df.columns]
        
        if not ode_target_cols:
            print("---> [ODE] 未找到气象列，跳过导数特征")
            return df
        
        try:
            df_derivs = _generate_ode_derivatives(df, ode_target_cols)
            df = pd.concat([df, df_derivs], axis=1)
            df = df.ffill().bfill()
        except Exception as e:
            print(f"---> [ODE] 导数特征生成失败，跳过: {e}")
        
        return df

    def prepare_features(self, df):
        """
        特征排序 + 归一化 + 计算 future_indices

        Args:
            df: 预处理后的 DataFrame

        Returns:
            np.ndarray: 归一化后的数据矩阵
        """
        cfg = self.cfg

        # 构建特征顺序 (自动加入能量累积特征)
        feature_order = list(cfg.feature_order_base) + list(cfg.outdoor_cols)
        # 自动检测并加入 Energy_60m 累积特征 和 ODE 导数特征
        energy_cols = [c for c in df.columns if c.endswith('_Energy_60m')]
        deriv_cols = [c for c in df.columns if c.endswith('_Deriv')]
        feature_order = feature_order + energy_cols + deriv_cols
        feature_order = [f for f in feature_order if f in df.columns]
        self.feature_order = feature_order

        df = df[feature_order]
        self.target_indices = [feature_order.index(c) for c in cfg.target_cols]

        # 计算 future_indices
        control_indices = [feature_order.index(c) for c in cfg.control_cols if c in feature_order]
        solar_indices = [feature_order.index(c) for c in cfg.indoor_solar_proxy if c in feature_order]
        time_indices = [feature_order.index(c) for c in cfg.time_cols if c in feature_order]
        weather_indices = [feature_order.index(c) for c in cfg.outdoor_cols if c in feature_order]
        self.future_indices = control_indices + solar_indices + time_indices + weather_indices

        print(f"---> future_indices 包含 {len(self.future_indices)} 列:")
        print(f"    控制量: {[feature_order[i] for i in control_indices]}")
        print(f"    天气/光照: {[feature_order[i] for i in solar_indices + weather_indices]}")
        print(f"    时间编码: {[feature_order[i] for i in time_indices]}")

        # 归一化 [修正: 仅在训练行上 fit，避免测试集极值污染归一化边界]
        # train_end_row 对应 80% 的原始行数，与 prepare_datasets 的划分口径一致
        train_end_row = int(len(df) * cfg.train_ratio)
        self.scaler.fit(df.iloc[:train_end_row])   # ✅ 只让训练集决定 min/max
        data_scaled = self.scaler.transform(df)    # ✅ 全量 transform（含测试集）
        print(f"    [Scaler] 仅在前 {train_end_row} 行（训练集）上 fit，共 {len(df)} 行全量 transform")
        return data_scaled

    @staticmethod
    def create_sequences(data, seq_length, forecast_horizon, future_indices, target_indices):
        """
        滑窗序列化

        Returns:
            (X_past, X_future, y) — numpy arrays
        """
        xs_past, xs_future, ys = [], [], []
        for i in range(len(data) - seq_length - forecast_horizon + 1):
            xs_past.append(data[i:(i + seq_length)])
            xs_future.append(data[i + seq_length: i + seq_length + forecast_horizon, future_indices])
            ys.append(data[i + seq_length: i + seq_length + forecast_horizon, target_indices])
        return np.array(xs_past), np.array(xs_future), np.array(ys)

    def prepare_datasets(self, data_scaled):
        """
        划分训练/测试集，严格隔断信息泄露（Data Leakage）

        Args:
            data_scaled: 归一化后的数据矩阵

        Returns:
            dict: {X_train_p, X_train_f, y_train, X_test_p, X_test_f, y_test}
        """
        cfg = self.cfg
        X_past, X_future, y = self.create_sequences(
            data_scaled, cfg.seq_len, cfg.horizon, self.future_indices, self.target_indices
        )

        train_size = int(len(X_past) * cfg.train_ratio)
        
        # [核心修复]: 强制加入信息隔离带 (Gap = seq_len + horizon)
        # 抛弃重叠区间的样本，确保测试集开头的历史 (past) 
        # 绝对不包含任何训练集末尾的未来目标 (target) 信息！
        gap = cfg.seq_len + cfg.horizon
        test_start = train_size + gap
        
        if test_start >= len(X_past):
            # 防御：防数据集过短
            print(f"    [警告] 数据集过短，隔离带 {gap} 超出边界。缩小为 train_ratio 控制。")
            test_start = train_size

        return {
            'X_train_p': X_past[:train_size],
            'X_train_f': X_future[:train_size],
            'y_train': y[:train_size],
            'X_test_p': X_past[test_start:],
            'X_test_f': X_future[test_start:],
            'y_test': y[test_start:],
        }
