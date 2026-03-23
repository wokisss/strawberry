# -*- coding: utf-8 -*-
"""AGC 2019 data loading, cleaning, merging, sequence construction, and scaling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import AGCConfig
from schema import build_feature_groups


class AGCDataProcessor:
    """Prepare AGC 2019 data for forecasting and predictive control."""

    def __init__(self, config: AGCConfig):
        self.cfg = config
        self.feature_groups = build_feature_groups(config)
        self.scalers: Dict[str, StandardScaler] = {}

    @staticmethod
    def _excel_time_to_datetime(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, unit="D", origin="1899-12-30")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        df = df.rename(columns=self.cfg.column_aliases)
        return df

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col == "timestamp":
                continue
            if df[col].dtype == object:
                cleaned = df[col].astype(str).str.strip()
                cleaned = cleaned.where(
                    ~cleaned.isin({"NaN", "nan", "None", ""}),
                    np.nan,
                )
                df[col] = pd.to_numeric(cleaned, errors="coerce")
        return df

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        df = self._normalize_columns(df)
        if "%time" not in df.columns:
            raise KeyError(f"Missing '%time' column in {path}")
        df["timestamp"] = self._excel_time_to_datetime(df["%time"]).dt.round(self.cfg.freq)
        df = self._coerce_numeric(df)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        return df

    def load_weather(self) -> pd.DataFrame:
        weather_path = Path(self.cfg.data_root) / "Weather" / "Weather.csv"
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather file not found: {weather_path}")

        weather = self._load_csv(weather_path)
        keep_cols = ["timestamp"] + self.cfg.future_weather_cols
        missing = [col for col in keep_cols if col not in weather.columns]
        if missing:
            raise KeyError(f"Missing weather columns: {missing}")
        return weather[keep_cols]

    def load_compartment_climate(self, compartment: str) -> pd.DataFrame:
        climate_path = Path(self.cfg.data_root) / compartment / "GreenhouseClimate.csv"
        if not climate_path.exists():
            raise FileNotFoundError(f"Climate file not found: {climate_path}")
        return self._load_csv(climate_path)

    def _apply_setpoint_fallbacks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for sp_col, vip_col in self.cfg.sp_vip_fallbacks.items():
            if sp_col in df.columns and vip_col in df.columns:
                df[sp_col] = df[sp_col].fillna(df[vip_col])
        return df

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        day_of_year = df["timestamp"].dt.dayofyear
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
        df["day_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
        return df

    def build_model_frame(self, compartment: str) -> pd.DataFrame:
        climate = self.load_compartment_climate(compartment)
        weather = self.load_weather()

        merged = pd.merge(climate, weather, on="timestamp", how="inner")
        merged = self._apply_setpoint_fallbacks(merged)
        merged = self._add_time_features(merged)
        merged["compartment"] = compartment

        required_cols = (
            ["timestamp", "compartment"]
            + self.feature_groups["x_past"]
            + self.feature_groups["w_future"]
            + self.feature_groups["u_future"]
            + self.feature_groups["y_future"]
        )
        required_cols = list(dict.fromkeys(required_cols))
        missing = [col for col in required_cols if col not in merged.columns]
        if missing:
            raise KeyError(f"Missing required columns for compartment {compartment}: {missing}")

        frame = merged[required_cols].copy()
        frame = frame.sort_values("timestamp")

        numeric_cols = [col for col in frame.columns if col not in {"timestamp", "compartment"}]
        frame[numeric_cols] = frame[numeric_cols].ffill().bfill()
        frame = frame.dropna(subset=numeric_cols)
        return frame

    def _build_sequences_from_frame(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        seq_len = self.cfg.seq_len
        horizon = self.cfg.horizon

        x_cols = self.feature_groups["x_past"]
        w_cols = self.feature_groups["w_future"]
        u_cols = self.feature_groups["u_future"]
        y_cols = self.feature_groups["y_future"]

        x_data = frame[x_cols].to_numpy(dtype=np.float32)
        w_data = frame[w_cols].to_numpy(dtype=np.float32)
        u_data = frame[u_cols].to_numpy(dtype=np.float32)
        y_data = frame[y_cols].to_numpy(dtype=np.float32)
        timestamps = frame["timestamp"].to_numpy()

        total = len(frame) - seq_len - horizon + 1
        if total <= 0:
            raise ValueError(
                f"Not enough rows ({len(frame)}) for seq_len={seq_len} and horizon={horizon}"
            )

        X_past, W_future, U_future, Y_future = [], [], [], []
        meta_time = []
        for start in range(total):
            split = start + seq_len
            end = split + horizon
            X_past.append(x_data[start:split])
            W_future.append(w_data[split:end])
            U_future.append(u_data[split:end])
            Y_future.append(y_data[split:end])
            meta_time.append(timestamps[split])

        return {
            "X_past": np.asarray(X_past, dtype=np.float32),
            "W_future": np.asarray(W_future, dtype=np.float32),
            "U_future": np.asarray(U_future, dtype=np.float32),
            "Y_future": np.asarray(Y_future, dtype=np.float32),
            "t0": np.asarray(meta_time),
        }

    def _split_sequences(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n = len(arrays["X_past"])
        gap = self.cfg.seq_len + self.cfg.horizon
        train_end = int(n * self.cfg.train_ratio)
        val_end = int(n * (self.cfg.train_ratio + self.cfg.val_ratio))

        if val_end + gap >= n:
            raise ValueError(
                f"Not enough sequence samples ({n}) for leak-free split with gap={gap}"
            )

        split_idx = {
            "train": slice(0, train_end),
            "val": slice(train_end + gap, val_end),
            "test": slice(val_end + gap, n),
        }

        out: Dict[str, np.ndarray] = {}
        for split_name, split_slice in split_idx.items():
            for key, value in arrays.items():
                out[f"{key}_{split_name}"] = value[split_slice]
        return out

    def _fit_scalers(self, split_arrays: Dict[str, np.ndarray]) -> None:
        layout = {
            "X_past": "x",
            "W_future": "w",
            "U_future": "u",
            "Y_future": "y",
        }

        for key_prefix, scaler_name in layout.items():
            train_key = f"{key_prefix}_train"
            scaler = StandardScaler()
            train_arr = split_arrays[train_key]
            scaler.fit(train_arr.reshape(-1, train_arr.shape[-1]))
            self.scalers[scaler_name] = scaler

    def _apply_scalers(self, split_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = dict(split_arrays)
        layout = {
            "X_past": "x",
            "W_future": "w",
            "U_future": "u",
            "Y_future": "y",
        }

        for key_prefix, scaler_name in layout.items():
            scaler = self.scalers[scaler_name]
            for split_name in ["train", "val", "test"]:
                key = f"{key_prefix}_{split_name}"
                arr = split_arrays[key]
                scaled = scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)
                result[key] = scaled.astype(np.float32)

        return result

    def build_compartment_raw_bundle(self, compartment: str) -> Dict[str, np.ndarray]:
        frame = self.build_model_frame(compartment)
        sequences = self._build_sequences_from_frame(frame)
        split_arrays = self._split_sequences(sequences)
        split_arrays["feature_groups"] = self.feature_groups
        split_arrays["compartment"] = compartment
        split_arrays["frame"] = frame
        return split_arrays

    def build_compartment_bundle(self, compartment: str) -> Dict[str, np.ndarray]:
        raw = self.build_compartment_raw_bundle(compartment)
        self._fit_scalers(raw)
        scaled = self._apply_scalers(raw)
        scaled["feature_groups"] = self.feature_groups
        scaled["compartment"] = compartment
        scaled["frame"] = raw["frame"]
        scaled["scalers"] = self.scalers
        return scaled

    def build_multi_compartment_bundle(self) -> Dict[str, np.ndarray]:
        raw_bundles = [self.build_compartment_raw_bundle(comp) for comp in self.cfg.selected_compartments]

        merged: Dict[str, np.ndarray] = {"feature_groups": self.feature_groups}
        for prefix in ["X_past", "W_future", "U_future", "Y_future", "t0"]:
            for split in ["train", "val", "test"]:
                key = f"{prefix}_{split}"
                merged[key] = np.concatenate([bundle[key] for bundle in raw_bundles], axis=0)

        self._fit_scalers(merged)
        merged = self._apply_scalers(merged)
        merged["compartments"] = list(self.cfg.selected_compartments)
        merged["scalers"] = self.scalers
        return merged

    def summarize_bundle(self, bundle: Dict[str, np.ndarray]) -> List[str]:
        lines = []
        for split in ["train", "val", "test"]:
            x_shape = bundle[f"X_past_{split}"].shape
            w_shape = bundle[f"W_future_{split}"].shape
            u_shape = bundle[f"U_future_{split}"].shape
            y_shape = bundle[f"Y_future_{split}"].shape
            lines.append(
                f"{split}: X_past={x_shape}, W_future={w_shape}, U_future={u_shape}, Y_future={y_shape}"
            )
        return lines
