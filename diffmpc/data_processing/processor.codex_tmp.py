# -*- coding: utf-8 -*-
"""Data loading, feature engineering, scaling, and dataset creation."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class _ODEF(nn.Module):
    """Small network used to estimate weather derivatives."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, y):
        t_vec = torch.ones_like(y[..., :1]) * t
        return self.net(torch.cat([y, t_vec], dim=-1))


def _generate_ode_derivatives(df, target_cols, fit_end_idx=None, train_epochs=300):
    """Estimate d/dt features on the selected columns."""
    print(f"---> [ODE] Building derivative features for {target_cols}...")

    df_clean = df[target_cols].dropna()
    if len(df_clean) < 2:
        raise ValueError("Need at least two rows to estimate derivatives.")

    if fit_end_idx is not None and fit_end_idx < len(df_clean):
        train_df = df_clean.iloc[:fit_end_idx]
        print(f"    [ODE] Fit on first {fit_end_idx} rows only to avoid test leakage.")
    else:
        train_df = df_clean

    if len(train_df) < 2:
        raise ValueError("Need at least two training rows to fit ODE derivatives.")

    scaler_ode = MinMaxScaler()
    train_data_np = scaler_ode.fit_transform(train_df.values)

    timestamps_train = (train_df.index - train_df.index[0]).total_seconds() / 3600.0
    t_tensor = torch.FloatTensor(
        timestamps_train.values if hasattr(timestamps_train, "values") else timestamps_train
    ).reshape(-1, 1)
    y_tensor = torch.FloatTensor(train_data_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_tensor = t_tensor.to(device)
    y_tensor = y_tensor.to(device)

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
    full_index = pd.date_range(start=df_clean.index[0], end=df_clean.index[-1], freq="1min")
    df_linear = df[target_cols].reindex(full_index).interpolate(method="linear").ffill().bfill()

    y_interp_tensor = torch.FloatTensor(scaler_ode.transform(df_linear.values)).to(device)
    t_full = torch.FloatTensor(
        (full_index - df_clean.index[0]).total_seconds().values / 3600.0
    ).reshape(-1, 1).to(device)

    with torch.no_grad():
        derivs = ode_func(t_full, y_interp_tensor).cpu().numpy()

    derivs_restored = derivs / (scaler_ode.scale_ + 1e-8)
    new_cols = [f"{col}_Deriv" for col in target_cols]
    df_derivs = pd.DataFrame(derivs_restored, index=full_index, columns=new_cols)

    print(f"    [ODE] Created {len(new_cols)} derivative columns: {new_cols}")
    return df_derivs


class DataProcessor:
    def __init__(self, config):
        self.cfg = config
        self.scaler = MinMaxScaler()
        self.feature_order = None
        self.future_indices = None
        self.target_indices = None

    def load_and_preprocess(self):
        cfg = self.cfg
        if not os.path.exists(cfg.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")

        df = pd.read_csv(
            cfg.dataset_path,
            encoding="latin1",
            sep=";",
            decimal=",",
            parse_dates=["Timestamp"],
            dayfirst=True,
            index_col="Timestamp",
            engine="python",
        )

        df.columns = [col.replace('"', "").strip() for col in df.columns]

        for col in cfg.binary_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ["on", "yes", "1"] else 0)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.resample("1min").mean().interpolate().ffill().bfill()
        print(f"---> Preprocessed dataset shape: {df.shape}")
        return df

    def merge_weather(self, df):
        cfg = self.cfg
        if not os.path.exists(cfg.weather_path):
            print(f"---> [Warning] Weather file not found: {cfg.weather_path}")
            return df

        print(f"---> Loading external weather data: {cfg.weather_path}")
        df_weather = pd.read_csv(cfg.weather_path, skiprows=12)

        df_weather["Timestamp"] = pd.to_datetime(
            df_weather["YEAR"].astype(str)
            + "-"
            + df_weather["MO"].astype(str).str.zfill(2)
            + "-"
            + df_weather["DY"].astype(str).str.zfill(2)
            + " "
            + df_weather["HR"].astype(str).str.zfill(2)
            + ":00:00"
        )
        df_weather = df_weather.set_index("Timestamp")
        df_weather = df_weather.rename(
            columns={
                "T2M": cfg.outdoor_temp_col,
                "ALLSKY_SFC_SW_DWN": cfg.outdoor_solar_col,
                "RH2M": cfg.outdoor_hum_col,
                "WS2M": cfg.outdoor_wind_col,
            }
        )

        weather_cols = cfg.outdoor_cols
        df_weather = df_weather[weather_cols].resample("1min").asfreq()

        try:
            df_weather[cfg.outdoor_temp_col] = (
                df_weather[cfg.outdoor_temp_col].interpolate(method="cubic").ffill().bfill()
            )
            df_weather[cfg.outdoor_hum_col] = (
                df_weather[cfg.outdoor_hum_col].interpolate(method="cubic").ffill().bfill()
            )
            df_weather[cfg.outdoor_solar_col] = (
                df_weather[cfg.outdoor_solar_col]
                .interpolate(method="pchip")
                .clip(lower=0.0)
                .ffill()
                .bfill()
            )

            base_wind = (
                df_weather[cfg.outdoor_wind_col]
                .interpolate(method="pchip")
                .clip(lower=0.0)
                .ffill()
                .bfill()
            )
            gust_noise = np.random.normal(0, 0.4, size=len(base_wind))
            df_weather[cfg.outdoor_wind_col] = np.clip(base_wind + gust_noise, a_min=0.0, a_max=None)
            print("    [Info] Used cubic/PCHIP weather interpolation.")
        except ImportError:
            print("    [Warning] scipy not available; falling back to linear interpolation.")
            df_weather = df_weather.interpolate(method="linear").ffill().bfill()

        df_weather = df_weather.reindex(df.index, method="ffill")
        for col in df_weather.columns:
            df[col] = df_weather[col]

        df = df.ffill().bfill()
        print(f"    Merged weather columns: {weather_cols}")
        return df

    def add_time_encoding(self, df):
        hour_float = df.index.hour + df.index.minute / 60.0
        df["Hour_Sin"] = np.sin(2 * np.pi * hour_float / 24.0)
        df["Hour_Cos"] = np.cos(2 * np.pi * hour_float / 24.0)
        print("---> Added time encoding features: Hour_Sin, Hour_Cos")
        return df

    def add_energy_features(self, df):
        cfg = self.cfg
        energy_window = 60
        control_cols = [col for col in cfg.control_cols if col in df.columns]

        if control_cols:
            new_cols = []
            for col in control_cols:
                new_col = f"{col}_Energy_60m"
                df[new_col] = df[col].rolling(window=energy_window, min_periods=1).sum()
                new_cols.append(new_col)
            df.fillna(0, inplace=True)
            print(f"---> Added {len(new_cols)} rolling energy features: {new_cols}")

        return df

    def add_ode_derivatives(self, df):
        ode_target_cols = [
            col
            for col in [self.cfg.outdoor_temp_col, self.cfg.outdoor_solar_col]
            if col in df.columns
        ]

        if not ode_target_cols:
            print("---> [ODE] No eligible weather columns found; skip derivatives.")
            return df

        try:
            train_end_row = int(len(df) * self.cfg.train_ratio)
            df_derivs = _generate_ode_derivatives(df, ode_target_cols, fit_end_idx=train_end_row)
            df = pd.concat([df, df_derivs], axis=1).ffill().bfill()
        except Exception as exc:
            print(f"---> [ODE] Failed to create derivative features, skip. Reason: {exc}")

        return df

    def prepare_features(self, df):
        cfg = self.cfg

        feature_order = list(cfg.feature_order_base) + list(cfg.outdoor_cols)
        energy_cols = [col for col in df.columns if col.endswith("_Energy_60m")]
        deriv_cols = [col for col in df.columns if col.endswith("_Deriv")]
        feature_order.extend(energy_cols)
        feature_order.extend(deriv_cols)
        feature_order = [feature for feature in feature_order if feature in df.columns]

        missing_targets = [col for col in cfg.target_cols if col not in feature_order]
        if missing_targets:
            raise ValueError(f"Missing target columns from feature set: {missing_targets}")

        self.feature_order = feature_order
        self.target_indices = [feature_order.index(col) for col in cfg.target_cols]

        control_indices = [feature_order.index(col) for col in cfg.control_cols if col in feature_order]
        solar_indices = [feature_order.index(col) for col in cfg.indoor_solar_proxy if col in feature_order]
        time_indices = [feature_order.index(col) for col in cfg.time_cols if col in feature_order]
        weather_indices = [feature_order.index(col) for col in cfg.outdoor_cols if col in feature_order]
        self.future_indices = control_indices + solar_indices + time_indices + weather_indices

        if not self.future_indices:
            raise ValueError("future_indices is empty; no future conditioning features were selected.")

        df = df[feature_order]
        train_end_row = int(len(df) * cfg.train_ratio)
        if train_end_row <= 0:
            raise ValueError("Training split is empty; cannot fit scaler.")

        self.scaler.fit(df.iloc[:train_end_row])
        data_scaled = self.scaler.transform(df)

        print(f"---> future_indices includes {len(self.future_indices)} columns")
        print(f"    controls: {[feature_order[i] for i in control_indices]}")
        print(f"    weather/solar: {[feature_order[i] for i in solar_indices + weather_indices]}")
        print(f"    time encodings: {[feature_order[i] for i in time_indices]}")
        print(
            f"    [Scaler] fit on first {train_end_row} rows only; "
            f"transformed full dataset with {len(df)} rows"
        )
        return data_scaled

    @staticmethod
    def create_sequences(data, seq_length, forecast_horizon, future_indices, target_indices):
        xs_past, xs_future, ys = [], [], []
        last_start = len(data) - seq_length - forecast_horizon + 1
        for i in range(last_start):
            xs_past.append(data[i : i + seq_length])
            xs_future.append(data[i + seq_length : i + seq_length + forecast_horizon, future_indices])
            ys.append(data[i + seq_length : i + seq_length + forecast_horizon, target_indices])
        return np.array(xs_past), np.array(xs_future), np.array(ys)

    def prepare_datasets(self, data_scaled):
        cfg = self.cfg
        X_past, X_future, y = self.create_sequences(
            data_scaled, cfg.seq_len, cfg.horizon, self.future_indices, self.target_indices
        )

        if len(X_past) == 0:
            raise ValueError(
                "No sequences were created. Check dataset length, seq_len, and horizon."
            )

        train_size = int(len(X_past) * cfg.train_ratio)
        gap = cfg.seq_len + cfg.horizon
        test_start = train_size + gap

        if train_size <= 0:
            raise ValueError("Training split is empty after sequence generation.")
        if test_start >= len(X_past):
            raise ValueError(
                "Not enough sequence samples to keep a leak-free train/test gap. "
                f"total={len(X_past)}, train={train_size}, required_gap={gap}"
            )

        datasets = {
            "X_train_p": X_past[:train_size],
            "X_train_f": X_future[:train_size],
            "y_train": y[:train_size],
            "X_test_p": X_past[test_start:],
            "X_test_f": X_future[test_start:],
            "y_test": y[test_start:],
        }

        if len(datasets["X_test_p"]) == 0:
            raise ValueError("Test split is empty after applying the leak-free gap.")

        print(
            f"---> Created datasets: train={len(datasets['X_train_p'])}, "
            f"test={len(datasets['X_test_p'])}, gap={gap}"
        )
        return datasets
