# -*- coding: utf-8 -*-
"""Compare full-period anchored MPC estimates with same-period AGC resources."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_agc_real_economics import _co2_cost
from config import AGCConfig
from results_utils import ensure_results_layout


DEFAULT_COMPARTMENTS = ["Reference", "Automatoes", "AICU"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().replace("\t", "").replace(" ", "_") for col in df.columns]
    df = df.rename(columns={"%Time": "%time", "%Time_": "%time", "%time_": "%time"})
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig")
    df = pd.read_csv(StringIO(text), low_memory=False)
    df = _normalise_columns(df)
    if "%time" in df.columns:
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["%time"], errors="coerce"), unit="D", origin="1899-12-30")
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resource_cost_row(
    *,
    source_type: str,
    label: str,
    compartment: str,
    predictor: str,
    resource_weight: float,
    start_timestamp: str,
    end_timestamp: str,
    heat: float,
    elec_high: float,
    elec_low: float,
    electricity: float,
    co2: float,
    irrigation: float,
    drain: float,
    objective: float,
    tair_mae: float,
    rhair_mae: float,
    co2_mae: float,
) -> dict:
    heat_cost = heat * 0.0083
    electricity_cost = elec_high * 0.08 + elec_low * 0.04 if elec_high or elec_low else electricity * 0.06
    co2_cost = _co2_cost(co2)
    total_cost = heat_cost + electricity_cost + co2_cost
    return {
        "source_type": source_type,
        "label": label,
        "compartment": compartment,
        "predictor": predictor,
        "resource_weight": resource_weight,
        "window_start_timestamp": start_timestamp,
        "window_end_timestamp": end_timestamp,
        "heat_cons_mj_m2": heat,
        "heat_cost_eur_m2": heat_cost,
        "elec_high_kwh_m2": elec_high,
        "elec_low_kwh_m2": elec_low,
        "electricity_kwh_m2": electricity if electricity else elec_high + elec_low,
        "electricity_cost_eur_m2": electricity_cost,
        "co2_cons_kg_m2": co2,
        "co2_cost_eur_m2": co2_cost,
        "irrigation_l_m2": irrigation,
        "drain_l_m2": drain,
        "estimated_total_resource_cost_eur_m2": total_cost,
        "objective_mean": objective,
        "tair_mae": tair_mae,
        "rhair_mae": rhair_mae,
        "co2_mae": co2_mae,
    }


def _build_agc_rows(data_root: Path, compartments: list[str], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    rows = []
    date_start = start.normalize()
    date_end = end.normalize()
    for compartment in compartments:
        resources = _read_csv(data_root / compartment / "Resources.csv")
        window = resources[(resources["timestamp"] >= date_start) & (resources["timestamp"] <= date_end)]
        heat = float(window["Heat_cons"].sum())
        elec_high = float(window["ElecHigh"].sum())
        elec_low = float(window["ElecLow"].sum())
        co2 = float(window["CO2_cons"].sum())
        irr = float(window["Irr"].sum())
        drain = float(window["Drain"].sum())
        rows.append(
            _resource_cost_row(
                source_type="real_agc_executed_resource",
                label=(
                    "real human-expert greenhouse execution"
                    if compartment == "Reference"
                    else "real AGC AI-team greenhouse execution"
                ),
                compartment=compartment,
                predictor="",
                resource_weight=float("nan"),
                start_timestamp=str(start),
                end_timestamp=str(end),
                heat=heat,
                elec_high=elec_high,
                elec_low=elec_low,
                electricity=elec_high + elec_low,
                co2=co2,
                irrigation=irr,
                drain=drain,
                objective=float("nan"),
                tair_mae=float("nan"),
                rhair_mae=float("nan"),
                co2_mae=float("nan"),
            )
        )
    return rows


def _build_mpc_rows(segment_csv: Path) -> tuple[list[dict], pd.Timestamp, pd.Timestamp]:
    frame = pd.read_csv(segment_csv)
    frame["segment_start_timestamp"] = pd.to_datetime(frame["segment_start_timestamp"])
    frame["segment_end_timestamp"] = pd.to_datetime(frame["segment_end_timestamp"])
    start = frame["segment_start_timestamp"].min()
    end = frame["segment_end_timestamp"].max()

    rows = []
    for (predictor, weight), group in frame.groupby(["predictor", "resource_weight"], sort=True):
        row = _resource_cost_row(
            source_type="counterfactual_estimated_mpc_resource",
            label="counterfactual anchored closed-loop simulation with estimated resource consumption",
            compartment="Reference",
            predictor=str(predictor),
            resource_weight=float(weight),
            start_timestamp=str(start),
            end_timestamp=str(end),
            heat=float(group["estimated_heat_mj_m2"].sum()),
            elec_high=0.0,
            elec_low=0.0,
            electricity=float(group["estimated_electricity_kwh_m2"].sum()),
            co2=float(group["estimated_co2_kg_m2"].sum()),
            irrigation=float(group["estimated_irrigation_l_m2"].sum()),
            drain=float("nan"),
            objective=float(group["objective_mean"].mean()),
            tair_mae=float(group["tair_mae"].mean()),
            rhair_mae=float(group["rhair_mae"].mean()),
            co2_mae=float(group["co2_mae"].mean()),
        )
        row["heat_cost_eur_m2"] = float(group["estimated_heat_cost_eur_m2"].sum())
        row["electricity_cost_eur_m2"] = float(group["estimated_electricity_cost_eur_m2"].sum())
        row["co2_cost_eur_m2"] = float(group["estimated_co2_cost_eur_m2"].sum())
        row["estimated_total_resource_cost_eur_m2"] = float(group["estimated_total_resource_cost_eur_m2"].sum())
        rows.append(row)
    return rows, start, end


def _add_relative_columns(rows: list[dict]) -> None:
    reference = next((row for row in rows if row["source_type"].startswith("real_agc") and row["compartment"] == "Reference"), None)
    ref_cost = float(reference["estimated_total_resource_cost_eur_m2"]) if reference else float("nan")
    for row in rows:
        cost = float(row["estimated_total_resource_cost_eur_m2"])
        row["resource_cost_vs_reference_pct"] = (
            (cost - ref_cost) / ref_cost * 100.0 if np.isfinite(ref_cost) and abs(ref_cost) > 1e-12 else float("nan")
        )

    mpc_baseline = {
        row["predictor"]: row
        for row in rows
        if row["source_type"].startswith("counterfactual") and abs(float(row["resource_weight"])) <= 1e-12
    }
    for row in rows:
        if not row["source_type"].startswith("counterfactual"):
            row["mpc_cost_change_vs_w000_pct"] = float("nan")
            row["mpc_co2_mae_change_vs_w000_pct"] = float("nan")
            continue
        base = mpc_baseline.get(row["predictor"])
        if base is None:
            row["mpc_cost_change_vs_w000_pct"] = float("nan")
            row["mpc_co2_mae_change_vs_w000_pct"] = float("nan")
            continue
        for metric, out_name in [
            ("estimated_total_resource_cost_eur_m2", "mpc_cost_change_vs_w000_pct"),
            ("co2_mae", "mpc_co2_mae_change_vs_w000_pct"),
        ]:
            denom = float(base[metric])
            row[out_name] = (
                (float(row[metric]) - denom) / denom * 100.0
                if np.isfinite(denom) and abs(denom) > 1e-12
                else float("nan")
            )


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Same-Period AGC Resource Baselines",
        "",
        "This comparison uses the same time window as the full-period anchored MPC experiment. AGC rows are real executed resource records. MPC rows are counterfactual resource estimates from generated control trajectories.",
        "",
        "| source | case | heat | electricity | CO2 | irrigation | resource cost | vs Reference | CO2 MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        case = row["compartment"] if row["source_type"].startswith("real_agc") else f"{row['predictor']} w={row['resource_weight']:.2f}"
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
                    f"{row['estimated_total_resource_cost_eur_m2']:.4f}",
                    f"{row['resource_cost_vs_reference_pct']:.1f}%",
                    "" if not np.isfinite(float(row["co2_mae"])) else f"{row['co2_mae']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary:",
            "",
            "- AGC rows are real resource consumption over daily records intersecting the MPC window.",
            "- MPC rows are estimated resource consumption over counterfactual anchored control trajectories.",
            "- Real AGC production or net profit is not ranked against MPC because the MPC rollout has no crop/yield/quality dynamic model.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_baselines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        row["compartment"] if row["source_type"].startswith("real_agc") else f"{row['predictor'].replace('_', ' ')}\nw={row['resource_weight']:.2f}"
        for row in rows
    ]
    x = np.arange(len(rows))
    heat = np.asarray([row["heat_cost_eur_m2"] for row in rows])
    elec = np.asarray([row["electricity_cost_eur_m2"] for row in rows])
    co2 = np.asarray([row["co2_cost_eur_m2"] for row in rows])
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(x, heat, label="heat")
    ax.bar(x, elec, bottom=heat, label="electricity")
    ax.bar(x, co2, bottom=heat + elec, label="CO2")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("resource cost, EUR/m2")
    ax.set_title("Same-period AGC real resources and MPC estimated resources")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cost_vs_control(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mpc_rows = [row for row in rows if row["source_type"].startswith("counterfactual")]
    agc_rows = [row for row in rows if row["source_type"].startswith("real_agc")]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for row in mpc_rows:
        label = f"{row['predictor'].replace('_', ' ')} w={row['resource_weight']:.2f}"
        ax.scatter(row["estimated_total_resource_cost_eur_m2"], row["co2_mae"], s=80)
        ax.annotate(label, (row["estimated_total_resource_cost_eur_m2"], row["co2_mae"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    for row in agc_rows:
        ax.axvline(row["estimated_total_resource_cost_eur_m2"], linestyle="--", alpha=0.35)
        ax.text(row["estimated_total_resource_cost_eur_m2"], ax.get_ylim()[1], row["compartment"], rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("resource cost, EUR/m2")
    ax.set_ylabel("MPC mean CO2air MAE")
    ax.set_title("Estimated MPC control trade-off with real AGC resource-cost references")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(args: argparse.Namespace) -> list[dict]:
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
    mpc_rows, start, end = _build_mpc_rows(segment_csv)
    agc_rows = _build_agc_rows(data_root, args.compartments, start, end)
    rows = agc_rows + mpc_rows
    _add_relative_columns(rows)

    summaries_dir = project_root / cfg.control_summaries_dir
    figures_dir = project_root / cfg.control_figures_dir
    csv_path = summaries_dir / f"{args.prefix}.csv"
    md_path = summaries_dir / f"{args.prefix}.md"
    fig_path = figures_dir / f"{args.prefix}.png"
    control_name = "agc_same_period_resource_cost_vs_control" if args.prefix == "agc_same_period_resource_baselines" else f"{args.prefix}_cost_vs_control"
    control_fig_path = figures_dir / f"{control_name}.png"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _plot_baselines(fig_path, rows)
    _plot_cost_vs_control(control_fig_path, rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {control_fig_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=Path("results/control/summaries/full_period_anchored_resource_mpc_segments.csv"),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--compartments", nargs="+", default=DEFAULT_COMPARTMENTS)
    parser.add_argument("--prefix", default="agc_same_period_resource_baselines")
    return parser.parse_args()


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
