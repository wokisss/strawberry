# -*- coding: utf-8 -*-
"""Evaluate selected MPC rollouts with the calibrated real-resource estimator."""

from __future__ import annotations

import argparse
import csv
import json
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AGCConfig
from results_utils import ensure_results_layout


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_weather(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig")
    df = pd.read_csv(StringIO(text), low_memory=False)
    df.columns = [str(col).strip().replace(" ", "_").replace("\t", "") for col in df.columns]
    df = df.rename(columns={"%Time": "%time", "%Time_": "%time", "%time_": "%time"})
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["%time"], errors="coerce"), unit="D", origin="1899-12-30").dt.round("5min")
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("timestamp")


def _rollout_feature_vector(trace: dict, weather: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    timestamps = pd.to_datetime(trace["timestamps"]).round("5min")
    controls = pd.DataFrame(trace["executed_actions"], columns=trace["control_cols"])
    controls["timestamp"] = timestamps
    controls = controls.sort_values("timestamp")
    merged = pd.merge_asof(controls, weather, on="timestamp", direction="nearest")
    day_of_year = merged["timestamp"].dt.dayofyear.astype(float)
    merged["day_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    merged["day_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)

    values: dict[str, float] = {}
    for col in trace["control_cols"]:
        for stat in ["mean", "max", "min"]:
            values[f"control_{col}_{stat}"] = float(getattr(merged[col], stat)())
    for col in ["Tout", "Iglob", "PARout", "Rain", "Windsp", "day_sin", "day_cos"]:
        if col not in merged.columns:
            continue
        for stat in ["mean", "max", "min"]:
            values[f"weather_{col}_{stat}"] = float(getattr(merged[col], stat)())

    heat_mean = values.get("control_t_heat_sp_mean", 0.0)
    tout_mean = values.get("weather_Tout_mean", 0.0)
    co2_mean = values.get("control_co2_sp_mean", 0.0)
    window_mean = values.get("control_window_pos_lee_sp_mean", 0.0)
    assim_mean = values.get("control_assim_sp_mean", 0.0)
    water_mean = values.get("control_water_sup_intervals_sp_min_mean", 0.0)
    values["derived_heat_drive"] = max(heat_mean - tout_mean, 0.0)
    values["derived_co2_pressure"] = max(co2_mean - 400.0, 0.0)
    values["derived_ventilation_pressure"] = max(window_mean, 0.0)
    values["derived_light_pressure"] = max(assim_mean, 0.0)
    values["derived_irrigation_request"] = max(water_mean, 0.0)

    return {name: float(values.get(name, 0.0)) for name in feature_names}


def _predict_resource(spec: dict, features: dict[str, float], steps: int) -> dict[str, float]:
    scale = float(steps) / 288.0
    out = {}
    for target, model in spec["models"].items():
        value = float(model["intercept"])
        for name, coef in model["coefficients"].items():
            value += float(coef) * float(features.get(name, 0.0))
        out[target] = max(value, 0.0) * scale
    return out


def _electricity_price(trace: dict) -> float:
    timestamps = pd.to_datetime(trace["timestamps"])
    high = ((timestamps.hour >= 7) & (timestamps.hour < 23)).mean()
    return float(high * 0.08 + (1.0 - high) * 0.04)


def _resource_costs(resources: dict[str, float], trace: dict) -> dict[str, float]:
    heat_cost = resources.get("heat_cons_mj_m2", 0.0) * 0.0083
    electricity_cost = resources.get("electricity_kwh_m2", 0.0) * _electricity_price(trace)
    co2_cost = resources.get("co2_cons_kg_m2", 0.0) * 0.08
    return {
        "estimated_heat_cost_eur_m2": heat_cost,
        "estimated_electricity_cost_eur_m2": electricity_cost,
        "estimated_co2_cost_eur_m2": co2_cost,
        "estimated_total_resource_cost_eur_m2": heat_cost + electricity_cost + co2_cost,
    }


def build_rows(suite_paths: list[Path], spec_path: Path, data_root: Path) -> list[dict]:
    spec = _load_json(spec_path)
    weather = _read_weather(data_root / "Weather" / "Weather.csv")
    feature_names = spec["features"]
    rows = []
    for suite_path in suite_paths:
        suite = _load_json(suite_path)
        profile = str(suite.get("economic_profile_name", "") or "tracking_real_resource_w000")
        weight = float(suite.get("economic_resource_weight", 0.0))
        for record in suite.get("records", []):
            trace_path = record.get("trace_path", "")
            if not trace_path:
                raise ValueError(
                    f"Missing trace_path in {suite_path}. Rerun the rollout after trace saving was added."
                )
            trace_file = Path(trace_path)
            if not trace_file.is_absolute():
                trace_file = Path(__file__).resolve().parent / trace_file
            trace = _load_json(trace_file)
            features = _rollout_feature_vector(trace, weather, feature_names)
            resources = _predict_resource(spec, features, int(record["steps"]))
            costs = _resource_costs(resources, trace)
            row = {
                "profile": profile,
                "resource_weight": weight,
                "predictor": record["predictor"],
                "start_idx": int(record["start_idx"]),
                "steps": int(record["steps"]),
                "objective_mean": float(record["objective_mean"]),
                "tair_mae": float(record["target_mae"]["Tair"]),
                "rhair_mae": float(record["target_mae"]["Rhair"]),
                "co2_mae": float(record["target_mae"]["CO2air"]),
                "estimated_heat_mj_m2": resources.get("heat_cons_mj_m2", 0.0),
                "estimated_electricity_kwh_m2": resources.get("electricity_kwh_m2", 0.0),
                "estimated_co2_kg_m2": resources.get("co2_cons_kg_m2", 0.0),
                "estimated_irrigation_l_m2": resources.get("irrigation_l_m2", 0.0),
                **costs,
                "trace_path": str(trace_file),
            }
            rows.append(row)

    baseline = {
        (row["profile"], row["start_idx"]): row
        for row in rows
        if row["predictor"] == "current_hybrid_transformer"
    }
    for row in rows:
        base = baseline.get((row["profile"], row["start_idx"]))
        denom = float(base["estimated_total_resource_cost_eur_m2"]) if base else float("nan")
        row["cost_change_vs_current_hybrid_pct"] = (
            (row["estimated_total_resource_cost_eur_m2"] - denom) / denom * 100.0
            if np.isfinite(denom) and abs(denom) > 1e-12
            else float("nan")
        )
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["profile"], row["resource_weight"], row["predictor"]), []).append(row)
    out = []
    metrics = [
        "objective_mean",
        "tair_mae",
        "rhair_mae",
        "co2_mae",
        "estimated_heat_mj_m2",
        "estimated_electricity_kwh_m2",
        "estimated_co2_kg_m2",
        "estimated_irrigation_l_m2",
        "estimated_total_resource_cost_eur_m2",
    ]
    for (profile, weight, predictor), items in sorted(grouped.items()):
        row = {
            "profile": profile,
            "resource_weight": weight,
            "predictor": predictor,
            "starts": ",".join(str(item["start_idx"]) for item in sorted(items, key=lambda v: v["start_idx"])),
        }
        for metric in metrics:
            values = np.asarray([item[metric] for item in items], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict], detail_rows: list[dict]) -> None:
    lines = [
        "# Mainline Real-Resource Control Comparison",
        "",
        "Selected closed-loop MPC rollouts are evaluated with the calibrated AGC resource estimator. Values are estimated for the 96-step rollout window, not for a season-long greenhouse crop.",
        "",
        "| profile | predictor | starts | objective | CO2 MAE | heat | electricity | CO2 use | irrigation | resource cost |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["profile"],
                    row["predictor"],
                    row["starts"],
                    f"{row['objective_mean_mean']:.4f}",
                    f"{row['co2_mae_mean']:.3f}",
                    f"{row['estimated_heat_mj_m2_mean']:.3f}",
                    f"{row['estimated_electricity_kwh_m2_mean']:.3f}",
                    f"{row['estimated_co2_kg_m2_mean']:.4f}",
                    f"{row['estimated_irrigation_l_m2_mean']:.3f}",
                    f"{row['estimated_total_resource_cost_eur_m2_mean']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary:",
            "",
            "- This comparison uses model-generated action trajectories and a calibrated resource estimator.",
            "- It is valid for tracking/resource trade-off analysis of selected MPC rollouts.",
            "- It does not claim true commercial net-profit improvement because production and quality dynamics are not part of the closed-loop surrogate.",
            "",
            f"Detail records: `{len(detail_rows)}` rollouts.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['profile']}\n{row['predictor'].replace('_', ' ')}" for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(x, [row["estimated_total_resource_cost_eur_m2_mean"] for row in rows], color="tab:green")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("estimated EUR / m2")
    axes[0].set_title("Estimated resource cost")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(
        [row["estimated_total_resource_cost_eur_m2_mean"] for row in rows],
        [row["co2_mae_mean"] for row in rows],
        s=70,
    )
    for row in rows:
        axes[1].annotate(
            row["predictor"].replace("_", " "),
            (row["estimated_total_resource_cost_eur_m2_mean"], row["co2_mae_mean"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[1].set_xlabel("estimated resource cost, EUR/m2")
    axes[1].set_ylabel("CO2air MAE")
    axes[1].set_title("CO2 tracking vs resource cost")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _resolve(path: Path, project_root: Path, original_cwd: Path) -> Path:
    if path.exists():
        return path
    if (original_cwd / path).exists():
        return original_cwd / path
    return project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-json", nargs="+", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, default=Path("results/control/summaries/agc_resource_cost_model.json"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--prefix", default="mainline_real_resource_model_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    cfg = AGCConfig()
    if args.data_root is not None:
        cfg.data_root = str(args.data_root)
    ensure_results_layout(cfg)

    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    suite_paths = [_resolve(path, project_root, original_cwd) for path in args.suite_json]
    spec_path = _resolve(args.model_spec, project_root, original_cwd)

    detail_rows = build_rows(suite_paths, spec_path, data_root)
    rows = aggregate_rows(detail_rows)
    csv_path = project_root / cfg.control_summaries_dir / f"{args.prefix}.csv"
    detail_csv_path = project_root / cfg.control_summaries_dir / f"{args.prefix}_details.csv"
    md_path = project_root / cfg.control_summaries_dir / f"{args.prefix}.md"
    fig_path = project_root / cfg.control_figures_dir / f"{args.prefix}.png"
    write_csv(csv_path, rows)
    write_csv(detail_csv_path, detail_rows)
    write_markdown(md_path, rows, detail_rows)
    plot(fig_path, rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {detail_csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
