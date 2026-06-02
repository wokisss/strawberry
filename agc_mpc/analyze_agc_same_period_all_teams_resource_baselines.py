# -*- coding: utf-8 -*-
"""Build same-period all-team AGC resource baselines and economic context."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_agc_real_economics import (
    COMPARTMENTS,
    _co2_cost,
    _crop_maintenance_cost,
    _estimate_income,
    _read_csv,
)
from config import AGCConfig
from results_utils import ensure_results_layout


def _resolve(path: Path, project_root: Path, original_cwd: Path) -> Path:
    if path.is_absolute():
        return path
    if (original_cwd / path).exists():
        return (original_cwd / path).resolve()
    return (project_root / path).resolve()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _window_from_segments(segment_csv: Path) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    frame = pd.read_csv(segment_csv)
    frame["segment_start_timestamp"] = pd.to_datetime(frame["segment_start_timestamp"])
    frame["segment_end_timestamp"] = pd.to_datetime(frame["segment_end_timestamp"])
    return frame["segment_start_timestamp"].min(), frame["segment_end_timestamp"].max(), frame


def _resource_costs(heat: float, elec_high: float, elec_low: float, co2: float) -> dict[str, float]:
    heat_cost = heat * 0.0083
    electricity_cost = elec_high * 0.08 + elec_low * 0.04
    co2_cost = _co2_cost(co2)
    return {
        "heat_cost_eur_m2": heat_cost,
        "electricity_cost_eur_m2": electricity_cost,
        "co2_cost_eur_m2": co2_cost,
        "resource_cost_eur_m2": heat_cost + electricity_cost + co2_cost,
    }


def _build_real_team_rows(data_root: Path, compartments: list[str], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    date_start = start.normalize()
    date_end = end.normalize()
    rows = []
    for compartment in compartments:
        base = data_root / compartment
        resources = _read_csv(base / "Resources.csv")
        production = _read_csv(base / "Production.csv")
        quality = _read_csv(base / "TomQuality.csv")
        crop = _read_csv(base / "CropParameters.csv")

        res_window = resources[(resources["timestamp"] >= date_start) & (resources["timestamp"] <= date_end)]
        prod_window = production[(production["timestamp"] >= date_start) & (production["timestamp"] <= date_end)]
        qual_window = quality[(quality["timestamp"] >= date_start) & (quality["timestamp"] <= date_end)]

        heat = float(res_window["Heat_cons"].sum())
        elec_high = float(res_window["ElecHigh"].sum())
        elec_low = float(res_window["ElecLow"].sum())
        co2 = float(res_window["CO2_cons"].sum())
        irrigation = float(res_window["Irr"].sum())
        drain = float(res_window["Drain"].sum())
        prod_a = float(prod_window["ProdA"].sum()) if "ProdA" in prod_window.columns else 0.0
        prod_b = float(prod_window["ProdB"].sum()) if "ProdB" in prod_window.columns else 0.0
        total_prod = prod_a + prod_b
        income, mean_price = _estimate_income(prod_window, quality) if len(prod_window) else (0.0, float("nan"))
        crop_cost = _crop_maintenance_cost(crop, date_start, date_end)
        costs = _resource_costs(heat, elec_high, elec_low, co2)
        variable_cost_excl_fixed = costs["resource_cost_eur_m2"] + crop_cost

        rows.append(
            {
                "source_type": "real_agc_executed_resource",
                "compartment": compartment,
                "window_start_timestamp": str(start),
                "window_end_timestamp": str(end),
                "heat_cons_mj_m2": heat,
                "elec_high_kwh_m2": elec_high,
                "elec_low_kwh_m2": elec_low,
                "electricity_kwh_m2": elec_high + elec_low,
                "co2_cons_kg_m2": co2,
                "irrigation_l_m2": irrigation,
                "drain_l_m2": drain,
                **costs,
                "prod_a_kg_m2": prod_a,
                "prod_b_kg_m2": prod_b,
                "total_tomato_kg_m2": total_prod,
                "mean_price_eur_kg": mean_price,
                "estimated_income_eur_m2": income,
                "crop_maintenance_cost_eur_m2": crop_cost,
                "variable_cost_excl_fixed_eur_m2": variable_cost_excl_fixed,
                "same_period_margin_excl_fixed_eur_m2": income - variable_cost_excl_fixed,
                "heat_mj_per_kg_tomato": heat / total_prod if total_prod > 0 else float("nan"),
                "electricity_kwh_per_kg_tomato": (elec_high + elec_low) / total_prod if total_prod > 0 else float("nan"),
                "co2_kg_per_kg_tomato": co2 / total_prod if total_prod > 0 else float("nan"),
                "irrigation_l_per_kg_tomato": irrigation / total_prod if total_prod > 0 else float("nan"),
                "resource_cost_eur_per_kg_tomato": costs["resource_cost_eur_m2"] / total_prod if total_prod > 0 else float("nan"),
            }
        )
    return rows


def _build_mpc_rows(segment_frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    rows = []
    for (predictor, weight), group in segment_frame.groupby(["predictor", "resource_weight"], sort=True):
        heat = float(group["estimated_heat_mj_m2"].sum())
        electricity = float(group["estimated_electricity_kwh_m2"].sum())
        co2 = float(group["estimated_co2_kg_m2"].sum())
        irrigation = float(group["estimated_irrigation_l_m2"].sum())
        rows.append(
            {
                "source_type": "counterfactual_estimated_mpc_resource",
                "compartment": "Reference",
                "predictor": str(predictor),
                "resource_weight": float(weight),
                "window_start_timestamp": str(start),
                "window_end_timestamp": str(end),
                "heat_cons_mj_m2": heat,
                "elec_high_kwh_m2": float("nan"),
                "elec_low_kwh_m2": float("nan"),
                "electricity_kwh_m2": electricity,
                "co2_cons_kg_m2": co2,
                "irrigation_l_m2": irrigation,
                "drain_l_m2": float("nan"),
                "heat_cost_eur_m2": float(group["estimated_heat_cost_eur_m2"].sum()),
                "electricity_cost_eur_m2": float(group["estimated_electricity_cost_eur_m2"].sum()),
                "co2_cost_eur_m2": float(group["estimated_co2_cost_eur_m2"].sum()),
                "resource_cost_eur_m2": float(group["estimated_total_resource_cost_eur_m2"].sum()),
                "objective_mean": float(group["objective_mean"].mean()),
                "co2_mae": float(group["co2_mae"].mean()),
            }
        )
    return rows


def _add_relative_columns(rows: list[dict]) -> None:
    reference = next(row for row in rows if row["source_type"] == "real_agc_executed_resource" and row["compartment"] == "Reference")
    ref_cost = float(reference["resource_cost_eur_m2"])
    aicu = next((row for row in rows if row["source_type"] == "real_agc_executed_resource" and row["compartment"] == "AICU"), None)
    aicu_cost = float(aicu["resource_cost_eur_m2"]) if aicu else float("nan")
    for row in rows:
        cost = float(row["resource_cost_eur_m2"])
        row["resource_cost_vs_reference_pct"] = (cost - ref_cost) / ref_cost * 100.0 if abs(ref_cost) > 1e-12 else float("nan")
        row["resource_cost_vs_aicu_pct"] = (
            (cost - aicu_cost) / aicu_cost * 100.0
            if np.isfinite(aicu_cost) and abs(aicu_cost) > 1e-12
            else float("nan")
        )


def _intensity_rows(real_rows: list[dict]) -> list[dict]:
    fields = [
        "compartment",
        "window_start_timestamp",
        "window_end_timestamp",
        "total_tomato_kg_m2",
        "heat_mj_per_kg_tomato",
        "electricity_kwh_per_kg_tomato",
        "co2_kg_per_kg_tomato",
        "irrigation_l_per_kg_tomato",
        "resource_cost_eur_per_kg_tomato",
    ]
    return [{field: row[field] for field in fields} for row in real_rows]


def _economic_rows(real_rows: list[dict]) -> list[dict]:
    fields = [
        "compartment",
        "window_start_timestamp",
        "window_end_timestamp",
        "prod_a_kg_m2",
        "prod_b_kg_m2",
        "total_tomato_kg_m2",
        "mean_price_eur_kg",
        "estimated_income_eur_m2",
        "resource_cost_eur_m2",
        "crop_maintenance_cost_eur_m2",
        "variable_cost_excl_fixed_eur_m2",
        "same_period_margin_excl_fixed_eur_m2",
    ]
    return [{field: row[field] for field in fields} for row in real_rows]


def _write_resource_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Same-Period All-Team Resource Baselines",
        "",
        "This table directly sums recorded AGC resources over the full-period anchored MPC window. MPC rows are included only as counterfactual estimated-resource references.",
        "",
        "| source | case | heat | electricity | CO2 | irrigation | resource cost | vs Reference | vs AICU | CO2 MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["source_type"] == "real_agc_executed_resource":
            case = row["compartment"]
            co2_mae = ""
        else:
            case = f"{row['predictor']} w={row['resource_weight']:.2f}"
            co2_mae = f"{row['co2_mae']:.3f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["source_type"],
                    case,
                    f"{row['heat_cons_mj_m2']:.3f}",
                    f"{row['electricity_kwh_m2']:.3f}",
                    f"{row['co2_cons_kg_m2']:.4f}",
                    f"{row['irrigation_l_m2']:.3f}",
                    f"{row['resource_cost_eur_m2']:.4f}",
                    f"{row['resource_cost_vs_reference_pct']:.1f}%",
                    f"{row['resource_cost_vs_aicu_pct']:.1f}%",
                    co2_mae,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary:",
            "",
            "- AGC rows are measured resources from recorded competition data.",
            "- MPC rows are counterfactual resource estimates from generated control trajectories.",
            "- This table is a resource baseline, not a net-profit ranking.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_economic_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Same-Period All-Team Economic Context",
        "",
        "This table reports real AGC production and approximate same-period variable-cost margin over the MPC comparison window. Fixed plant cost is omitted because this is a partial-window context table rather than a full-season net-profit calculation.",
        "",
        "| compartment | income | resource cost | crop maintenance | variable cost excl fixed | margin excl fixed | tomato kg/m2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: item["same_period_margin_excl_fixed_eur_m2"], reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["compartment"],
                    f"{row['estimated_income_eur_m2']:.3f}",
                    f"{row['resource_cost_eur_m2']:.4f}",
                    f"{row['crop_maintenance_cost_eur_m2']:.4f}",
                    f"{row['variable_cost_excl_fixed_eur_m2']:.4f}",
                    f"{row['same_period_margin_excl_fixed_eur_m2']:.3f}",
                    f"{row['total_tomato_kg_m2']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary:",
            "",
            "- This is real AGC team context only.",
            "- MPC has no crop/yield/quality dynamic model here, so it is not included in the margin table.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_resource(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    for row in rows:
        labels.append(row["compartment"] if row["source_type"] == "real_agc_executed_resource" else f"{row['predictor'].replace('_', ' ')}\nw={row['resource_weight']:.2f}")
    x = np.arange(len(rows))
    heat = np.asarray([row["heat_cost_eur_m2"] for row in rows])
    elec = np.asarray([row["electricity_cost_eur_m2"] for row in rows])
    co2 = np.asarray([row["co2_cost_eur_m2"] for row in rows])
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.bar(x, heat, label="heat")
    ax.bar(x, elec, bottom=heat, label="electricity")
    ax.bar(x, co2, bottom=heat + elec, label="CO2")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("resource cost, EUR/m2")
    ax.set_title("Same-period all-team real resources and MPC estimated resources")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_intensity(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda item: item["resource_cost_eur_per_kg_tomato"])
    names = [row["compartment"] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar(x, [row["resource_cost_eur_per_kg_tomato"] for row in rows], color="tab:green")
    axes[0].set_xticks(x, names, rotation=30, ha="right")
    axes[0].set_ylabel("EUR / kg tomato")
    axes[0].set_title("Resource cost intensity")
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.25
    axes[1].bar(x - width, [row["heat_mj_per_kg_tomato"] for row in rows], width=width, label="heat MJ/kg")
    axes[1].bar(x, [row["co2_kg_per_kg_tomato"] for row in rows], width=width, label="CO2 kg/kg")
    axes[1].bar(x + width, [row["irrigation_l_per_kg_tomato"] for row in rows], width=width, label="irrigation L/kg")
    axes[1].set_xticks(x, names, rotation=30, ha="right")
    axes[1].set_title("Physical resource intensities")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_economic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda item: item["same_period_margin_excl_fixed_eur_m2"], reverse=True)
    names = [row["compartment"] for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.bar(x - 0.25, [row["estimated_income_eur_m2"] for row in rows], width=0.25, label="income")
    ax.bar(x, [row["variable_cost_excl_fixed_eur_m2"] for row in rows], width=0.25, label="variable cost excl fixed")
    ax.bar(x + 0.25, [row["same_period_margin_excl_fixed_eur_m2"] for row in rows], width=0.25, label="margin excl fixed")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylabel("EUR/m2")
    ax.set_title("Same-period real AGC economic context")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    project_root = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    cfg = AGCConfig()
    if args.data_root is not None:
        cfg.data_root = str(_resolve(args.data_root, project_root, original_cwd))
    ensure_results_layout(cfg)

    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    segment_csv = _resolve(args.segments_csv, project_root, original_cwd)
    start, end, segment_frame = _window_from_segments(segment_csv)

    real_rows = _build_real_team_rows(data_root, args.compartments, start, end)
    resource_rows = list(real_rows)
    if args.include_mpc:
        resource_rows.extend(_build_mpc_rows(segment_frame, start, end))
    _add_relative_columns(resource_rows)
    intensity = _intensity_rows(real_rows)
    economic = _economic_rows(real_rows)

    summaries_dir = project_root / cfg.control_summaries_dir
    figures_dir = project_root / cfg.control_figures_dir
    resource_csv = summaries_dir / f"{args.prefix}_resource_baselines.csv"
    resource_md = summaries_dir / f"{args.prefix}_resource_baselines.md"
    intensity_csv = summaries_dir / f"{args.prefix}_resource_intensity.csv"
    economic_csv = summaries_dir / f"{args.prefix}_economic_context.csv"
    economic_md = summaries_dir / f"{args.prefix}_economic_context.md"
    resource_fig = figures_dir / f"{args.prefix}_resource_baselines.png"
    intensity_fig = figures_dir / f"{args.prefix}_resource_intensity.png"
    economic_fig = figures_dir / f"{args.prefix}_economic_context.png"

    _write_csv(resource_csv, resource_rows)
    _write_resource_markdown(resource_md, resource_rows)
    _write_csv(intensity_csv, intensity)
    _write_csv(economic_csv, economic)
    _write_economic_markdown(economic_md, economic)
    _plot_resource(resource_fig, resource_rows)
    _plot_intensity(intensity_fig, intensity)
    _plot_economic(economic_fig, economic)

    for path in [resource_csv, resource_md, intensity_csv, economic_csv, economic_md, resource_fig, intensity_fig, economic_fig]:
        print(f"Saved: {path}")
    return resource_rows, intensity, economic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=Path("results/control/summaries/full_period_anchored_resource_mpc_segments.csv"),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--compartments", nargs="+", default=COMPARTMENTS)
    parser.add_argument("--prefix", default="agc_same_period_all_teams")
    parser.add_argument("--include-mpc", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
