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
from sklearn.preprocessing import MinMaxScaler


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
            parse_dates=['Timestamp'], dayfirst=True, index_col='Timestamp'
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

        # 重采样到分钟级
        weather_cols = cfg.outdoor_cols
        df_weather = df_weather[weather_cols]
        df_weather = df_weather.resample('1min').interpolate(method='linear').ffill().bfill()

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

    def prepare_features(self, df):
        """
        特征排序 + 归一化 + 计算 future_indices

        Args:
            df: 预处理后的 DataFrame

        Returns:
            np.ndarray: 归一化后的数据矩阵
        """
        cfg = self.cfg

        # 构建特征顺序
        feature_order = list(cfg.feature_order_base) + list(cfg.outdoor_cols)
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

        # 归一化
        data_scaled = self.scaler.fit_transform(df)
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
        划分训练/测试集

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
        return {
            'X_train_p': X_past[:train_size],
            'X_train_f': X_future[:train_size],
            'y_train': y[:train_size],
            'X_test_p': X_past[train_size:],
            'X_test_f': X_future[train_size:],
            'y_test': y[train_size:],
        }
