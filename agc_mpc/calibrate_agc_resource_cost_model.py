# -*- coding: utf-8 -*-
"""Calibrate simple AGC resource-consumption estimators from recorded data."""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import AGCConfig
from results_utils import ensure_results_layout


COMPARTMENTS = [
    "AICU",
    "Automatoes",
    "Digilog",
    "IUACAAS",
    "Reference",
    "TheAutomators",
]

CONTROL_FEATURES = [
    "t_heat_sp",
    "t_vent_sp",
    "co2_sp",
    "assim_sp",
    "window_pos_lee_sp",
    "water_sup_intervals_sp_min",
    "scr_enrg_sp",
    "scr_blck_sp",
]

WEATHER_FEATURES = ["Tout", "Iglob", "PARout", "Rain", "Windsp"]

TARGETS = {
    "heat_cons_mj_m2": "Heat_cons",
    "electricity_kwh_m2": "electricity_total",
    "co2_cons_kg_m2": "CO2_cons",
    "irrigation_l_m2": "Irr",
}


def excel_time_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="D", origin="1899-12-30")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().replace(" ", "_").replace("\t", "") for col in df.columns]
    df = df.rename(columns={"%Time": "%time", "%Time_": "%time", "%time_": "%time"})
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def read_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig").replace("Weight\tDMC_fruit", "Weight,DMC_fruit")
    df = pd.read_csv(StringIO(text), low_memory=False)
    df = normalise_columns(df)
    if "%time" not in df.columns:
        raise KeyError(f"Missing %time in {path}")
    df["timestamp"] = excel_time_to_datetime(df["%time"])
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    day_of_year = df["timestamp"].dt.dayofyear.astype(float)
    df["day_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    df["day_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)
    return df


def _daily_aggregate(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    work = frame[["date"] + [col for col in columns if col in frame.columns]].copy()
    numeric = [col for col in work.columns if col != "date"]
    grouped = work.groupby("date")[numeric].agg(["mean", "max", "min"])
    grouped.columns = [f"{prefix}_{col}_{stat}" for col, stat in grouped.columns]
    return grouped.reset_index()


def build_training_frame(data_root: Path, compartments: list[str]) -> pd.DataFrame:
    weather = read_csv(data_root / "Weather" / "Weather.csv")
    weather["timestamp"] = weather["timestamp"].dt.round("5min")
    weather = add_time_features(weather)

    rows = []
    for compartment in compartments:
        base = data_root / compartment
        climate = read_csv(base / "GreenhouseClimate.csv")
        resources = read_csv(base / "Resources.csv")
        climate["timestamp"] = climate["timestamp"].dt.round("5min")
        climate = climate.merge(weather, on="timestamp", how="left", suffixes=("", "_weather"))
        climate = add_time_features(climate)
        climate["date"] = climate["timestamp"].dt.normalize()
        resources["date"] = resources["timestamp"].dt.normalize()

        control_daily = _daily_aggregate(climate, CONTROL_FEATURES, "control")
        weather_daily = _daily_aggregate(climate, WEATHER_FEATURES + ["day_sin", "day_cos"], "weather")
        daily = resources.merge(control_daily, on="date", how="left").merge(weather_daily, on="date", how="left")
        daily["compartment"] = compartment
        daily["electricity_total"] = daily["ElecHigh"].fillna(0.0) + daily["ElecLow"].fillna(0.0)

        heat_mean = daily.get("control_t_heat_sp_mean", pd.Series(0.0, index=daily.index))
        tout_mean = daily.get("weather_Tout_mean", pd.Series(0.0, index=daily.index))
        co2_mean = daily.get("control_co2_sp_mean", pd.Series(0.0, index=daily.index))
        window_mean = daily.get("control_window_pos_lee_sp_mean", pd.Series(0.0, index=daily.index))
        assim_mean = daily.get("control_assim_sp_mean", pd.Series(0.0, index=daily.index))
        water_mean = daily.get("control_water_sup_intervals_sp_min_mean", pd.Series(0.0, index=daily.index))

        daily["derived_heat_drive"] = np.maximum(heat_mean - tout_mean, 0.0)
        daily["derived_co2_pressure"] = np.maximum(co2_mean - 400.0, 0.0)
        daily["derived_ventilation_pressure"] = np.maximum(window_mean, 0.0)
        daily["derived_light_pressure"] = np.maximum(assim_mean, 0.0)
        daily["derived_irrigation_request"] = np.maximum(water_mean, 0.0)
        rows.append(daily)

    frame = pd.concat(rows, ignore_index=True)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("control_", "weather_", "derived_")
    return [col for col in frame.columns if col.startswith(prefixes)]


def fit_models(frame: pd.DataFrame, alpha: float) -> tuple[dict, pd.DataFrame]:
    features = feature_columns(frame)
    clean = frame[["compartment", "date"] + features + list(TARGETS.values())].copy()
    clean[features] = clean[features].ffill().bfill().fillna(0.0)
    models = {}
    validation_rows = []

    for label, target_col in TARGETS.items():
        target = pd.to_numeric(clean[target_col], errors="coerce")
        mask = target.notna()
        x = clean.loc[mask, features].to_numpy(dtype=np.float64)
        y = target.loc[mask].to_numpy(dtype=np.float64)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, positive=True))
        model.fit(x, y)
        pred = model.predict(x)
        models[label] = model
        validation_rows.append(
            {
                "target": label,
                "samples": int(mask.sum()),
                "train_mae": float(mean_absolute_error(y, pred)),
                "train_r2": float(r2_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
                "observed_mean": float(np.mean(y)),
                "predicted_mean": float(np.mean(pred)),
            }
        )
    return models, pd.DataFrame(validation_rows)


def coefficient_rows(models: dict, features: list[str]) -> list[dict]:
    rows = []
    for target, model in models.items():
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        coef_raw = ridge.coef_ / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        intercept_raw = ridge.intercept_ - float(np.sum(coef_raw * scaler.mean_))
        rows.append({"target": target, "feature": "__intercept__", "coefficient": float(intercept_raw)})
        for feature, coef in zip(features, coef_raw):
            rows.append({"target": target, "feature": feature, "coefficient": float(coef)})
    return rows


def model_spec(models: dict, features: list[str]) -> dict:
    spec = {
        "features": features,
        "targets": list(models.keys()),
        "models": {},
        "cost_rules": {
            "heat_eur_per_mj": 0.0083,
            "electricity_eur_per_kwh_mixed": 0.06,
            "co2_eur_per_kg_low_tier": 0.08,
            "co2_eur_per_kg_high_tier": 0.20,
        },
    }
    for target, model in models.items():
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        coef_raw = ridge.coef_ / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        intercept_raw = ridge.intercept_ - float(np.sum(coef_raw * scaler.mean_))
        spec["models"][target] = {
            "intercept": float(intercept_raw),
            "coefficients": {feature: float(coef) for feature, coef in zip(features, coef_raw)},
        }
    return spec


def write_coefficients(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target", "feature", "coefficient"])
        writer.writeheader()
        writer.writerows(rows)


def write_validation(path: Path, validation: pd.DataFrame, spec_path: Path) -> None:
    lines = [
        "# AGC Resource Cost Model Validation",
        "",
        "The estimator is a simple positive-coefficient ridge model fitted on daily AGC records. It is intended as an interpretable cost bridge from MPC action trajectories to approximate resource implications, not as a crop-profit simulator.",
        "",
        f"Serialized model spec: `{spec_path.name}`",
        "",
        "| target | samples | MAE | R2 | observed mean | predicted mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in validation.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["target"]),
                    f"{int(row['samples'])}",
                    f"{float(row['train_mae']):.4f}",
                    f"{float(row['train_r2']):.3f}",
                    f"{float(row['observed_mean']):.4f}",
                    f"{float(row['predicted_mean']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary:",
            "",
            "- The model estimates recorded daily resource consumption from setpoint/weather summaries.",
            "- It can compare selected short MPC rollouts after scaling the estimate by rollout length.",
            "- It must not be used to claim season-long commercial net-profit improvement, because yield dynamics are not modeled in closed-loop.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_validation(path: Path, frame: pd.DataFrame, models: dict, features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = frame[features].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.reshape(-1)
    for ax, (label, target_col) in zip(axes, TARGETS.items()):
        y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(y)
        pred = models[label].predict(x[mask])
        ax.scatter(y[mask], pred, s=14, alpha=0.45)
        lo = min(float(np.nanmin(y[mask])), float(np.nanmin(pred)))
        hi = max(float(np.nanmax(y[mask])), float(np.nanmax(pred)))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2)
        ax.set_title(label)
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--compartments", nargs="+", default=COMPARTMENTS)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = AGCConfig()
    if args.data_root is not None:
        cfg.data_root = str(args.data_root)
    ensure_results_layout(cfg)

    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    frame = build_training_frame(data_root, args.compartments)
    features = feature_columns(frame)
    models, validation = fit_models(frame, args.alpha)

    coef_path = project_root / cfg.control_summaries_dir / "agc_resource_cost_model_coefficients.csv"
    validation_path = project_root / cfg.control_summaries_dir / "agc_resource_cost_model_validation.md"
    spec_path = project_root / cfg.control_summaries_dir / "agc_resource_cost_model.json"
    fig_path = project_root / cfg.control_figures_dir / "agc_resource_cost_model_validation.png"

    write_coefficients(coef_path, coefficient_rows(models, features))
    spec_path.write_text(json.dumps(model_spec(models, features), indent=2, ensure_ascii=False), encoding="utf-8")
    write_validation(validation_path, validation, spec_path)
    plot_validation(fig_path, frame, models, features)

    print(f"Saved: {coef_path}")
    print(f"Saved: {validation_path}")
    print(f"Saved: {spec_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
