# -*- coding: utf-8 -*-
"""Estimate real AGC resource use and approximate economics by compartment."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

PRICE_TABLE = [
    ("2020-01-01", 5.00, 3.00),
    ("2020-01-14", 5.00, 3.00),
    ("2020-01-15", 5.20, 3.50),
    ("2020-01-28", 5.20, 3.50),
    ("2020-01-29", 4.50, 3.50),
    ("2020-02-11", 4.50, 3.50),
    ("2020-02-12", 4.20, 2.80),
    ("2020-02-25", 4.20, 2.80),
    ("2020-02-26", 3.80, 2.50),
    ("2020-03-10", 3.80, 2.50),
    ("2020-03-11", 3.20, 2.20),
    ("2020-03-24", 3.20, 2.20),
    ("2020-03-25", 2.80, 2.00),
    ("2020-04-07", 2.80, 2.00),
    ("2020-04-08", 2.60, 1.80),
    ("2020-04-21", 2.60, 1.80),
    ("2020-04-22", 2.40, 1.60),
    ("2020-05-05", 2.40, 1.60),
    ("2020-05-06", 2.50, 1.40),
    ("2020-05-19", 2.50, 1.40),
    ("2020-05-20", 2.60, 1.20),
    ("2020-06-02", 2.60, 1.20),
    ("2020-06-03", 2.50, 1.10),
    ("2020-06-16", 2.50, 1.10),
    ("2020-06-17", 2.50, 1.10),
]


def _excel_time_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="D", origin="1899-12-30")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().replace("\t", "").replace(" ", "_") for col in df.columns]
    rename = {
        "%Time": "%time",
        "%Time_": "%time",
        "%time_": "%time",
        "Flavour_": "Flavour",
        "Truss_development_time": "Truss_development_time",
    }
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("Weight\tDMC_fruit", "Weight,DMC_fruit")
    df = pd.read_csv(StringIO(text), low_memory=False)
    df = _normalise_columns(df)
    if "%time" in df.columns:
        df["timestamp"] = _excel_time_to_datetime(df["%time"])
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _price_for(date: pd.Timestamp, brix: float) -> float:
    table = pd.DataFrame(PRICE_TABLE, columns=["date", "brix10", "brix6"])
    table["date"] = pd.to_datetime(table["date"])
    valid = table[table["date"] <= date.normalize()]
    row = valid.iloc[-1] if len(valid) else table.iloc[0]
    brix = float(np.clip(brix, 6.0, 10.0)) if np.isfinite(brix) else 8.0
    return float(row["brix6"] + (brix - 6.0) / 4.0 * (row["brix10"] - row["brix6"]))


def _co2_cost(total_kg_m2: float) -> float:
    total = max(float(total_kg_m2), 0.0)
    return min(total, 12.0) * 0.08 + max(total - 12.0, 0.0) * 0.20


def _estimate_income(production: pd.DataFrame, quality: pd.DataFrame) -> tuple[float, float]:
    prod = production.sort_values("timestamp").copy()
    qual = quality[["timestamp", "TSS"]].dropna().sort_values("timestamp").copy()
    if len(qual) == 0:
        prod["TSS"] = 8.0
    else:
        prod = pd.merge_asof(prod, qual, on="timestamp", direction="nearest")
        prod["TSS"] = prod["TSS"].ffill().bfill().fillna(8.0)

    income = 0.0
    weighted_price = []
    for _, row in prod.iterrows():
        price = _price_for(row["timestamp"], row.get("TSS", 8.0))
        prod_a = max(float(row.get("ProdA", 0.0) or 0.0), 0.0)
        prod_b = max(float(row.get("ProdB", 0.0) or 0.0), 0.0)
        income += prod_a * price + prod_b * 0.5 * price
        if prod_a + prod_b > 0:
            weighted_price.append(price)
    mean_price = float(np.mean(weighted_price)) if weighted_price else float("nan")
    return float(income), mean_price


def _crop_maintenance_cost(crop: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if len(crop) == 0 or "stem_dens" not in crop.columns:
        return 0.0
    frame = crop[["timestamp", "stem_dens"]].dropna().sort_values("timestamp")
    if len(frame) == 0:
        return 0.0
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    daily = pd.DataFrame({"timestamp": days})
    daily = pd.merge_asof(daily, frame, on="timestamp", direction="backward")
    daily["stem_dens"] = daily["stem_dens"].ffill().bfill().fillna(frame["stem_dens"].mean())
    return float((daily["stem_dens"] * 0.0085).sum())


def build_rows(data_root: Path, compartments: list[str]) -> list[dict]:
    rows = []
    for compartment in compartments:
        base = data_root / compartment
        resources = _read_csv(base / "Resources.csv")
        production = _read_csv(base / "Production.csv")
        quality = _read_csv(base / "TomQuality.csv")
        crop = _read_csv(base / "CropParameters.csv")

        heat = float(resources["Heat_cons"].sum())
        elec_high = float(resources["ElecHigh"].sum())
        elec_low = float(resources["ElecLow"].sum())
        co2 = float(resources["CO2_cons"].sum())
        irr = float(resources["Irr"].sum())
        drain = float(resources["Drain"].sum())
        prod_a = float(production["ProdA"].sum())
        prod_b = float(production["ProdB"].sum())
        total_prod = prod_a + prod_b

        heat_cost = heat * 0.0083
        elec_cost = elec_high * 0.08 + elec_low * 0.04
        co2_cost = _co2_cost(co2)
        crop_cost = _crop_maintenance_cost(crop, resources["timestamp"].min(), resources["timestamp"].max())
        income, mean_price = _estimate_income(production, quality)
        variable_cost = heat_cost + elec_cost + co2_cost + crop_cost

        plant_dens = float(crop["plant_dens"].dropna().iloc[0]) if "plant_dens" in crop.columns and crop["plant_dens"].notna().any() else 0.0
        fixed_plant_cost = plant_dens * 2.20
        approximate_net_profit = income - fixed_plant_cost - variable_cost

        rows.append(
            {
                "compartment": compartment,
                "heat_cons_mj_m2": heat,
                "heat_cost_eur_m2": heat_cost,
                "elec_high_kwh_m2": elec_high,
                "elec_low_kwh_m2": elec_low,
                "electricity_cost_eur_m2": elec_cost,
                "co2_cons_kg_m2": co2,
                "co2_cost_eur_m2": co2_cost,
                "irrigation_l_m2": irr,
                "drain_l_m2": drain,
                "prod_a_kg_m2": prod_a,
                "prod_b_kg_m2": prod_b,
                "total_tomato_kg_m2": total_prod,
                "mean_price_eur_kg": mean_price,
                "estimated_income_eur_m2": income,
                "crop_maintenance_cost_eur_m2": crop_cost,
                "fixed_plant_cost_eur_m2": fixed_plant_cost,
                "estimated_variable_cost_eur_m2": variable_cost,
                "approx_net_profit_eur_m2": approximate_net_profit,
                "heat_mj_per_kg_tomato": heat / total_prod if total_prod > 0 else float("nan"),
                "elec_kwh_per_kg_tomato": (elec_high + elec_low) / total_prod if total_prod > 0 else float("nan"),
                "co2_kg_per_kg_tomato": co2 / total_prod if total_prod > 0 else float("nan"),
                "irrigation_l_per_kg_tomato": irr / total_prod if total_prod > 0 else float("nan"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# AGC Real Resource And Economics Baseline",
        "",
        "This table estimates compartment-level resource use and approximate economics from the recorded AGC 2019 data. Net profit is approximate: tomato income uses the official date/Brix price table with nearest available TSS measurements, crop maintenance uses recorded stem density, and plant fixed cost uses the documented two-stem plant price.",
        "",
        "| compartment | income | variable cost | fixed plant cost | approx net profit | tomato kg/m2 | heat MJ/m2 | elec kWh/m2 | CO2 kg/m2 | irrigation L/m2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: item["approx_net_profit_eur_m2"], reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["compartment"],
                    f"{row['estimated_income_eur_m2']:.2f}",
                    f"{row['estimated_variable_cost_eur_m2']:.2f}",
                    f"{row['fixed_plant_cost_eur_m2']:.2f}",
                    f"{row['approx_net_profit_eur_m2']:.2f}",
                    f"{row['total_tomato_kg_m2']:.2f}",
                    f"{row['heat_cons_mj_m2']:.1f}",
                    f"{row['elec_high_kwh_m2'] + row['elec_low_kwh_m2']:.1f}",
                    f"{row['co2_cons_kg_m2']:.2f}",
                    f"{row['irrigation_l_m2']:.1f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Official variable-cost rules encoded here:",
            "",
            "- electricity: `0.08 EUR/kWh` peak and `0.04 EUR/kWh` off-peak",
            "- heat: `0.0083 EUR/MJ`",
            "- CO2: `0.08 EUR/kg` for the first `12 kg/m2`, then `0.20 EUR/kg`",
            "- crop maintenance: `0.0085 EUR per stem/m2 per day`",
            "- Class A tomatoes use full estimated price; Class B uses half price",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda item: item["approx_net_profit_eur_m2"], reverse=True)
    names = [row["compartment"] for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(x - 0.2, [row["estimated_income_eur_m2"] for row in rows], width=0.4, label="income")
    axes[0].bar(x + 0.2, [row["estimated_variable_cost_eur_m2"] for row in rows], width=0.4, label="variable cost")
    axes[0].plot(x, [row["approx_net_profit_eur_m2"] for row in rows], color="black", marker="o", label="approx net")
    axes[0].set_xticks(x, names, rotation=30, ha="right")
    axes[0].set_ylabel("EUR / m2")
    axes[0].set_title("Approximate economics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    width = 0.25
    axes[1].bar(x - width, [row["heat_mj_per_kg_tomato"] for row in rows], width=width, label="heat MJ/kg")
    axes[1].bar(x, [row["elec_kwh_per_kg_tomato"] for row in rows], width=width, label="elec kWh/kg")
    axes[1].bar(x + width, [row["co2_kg_per_kg_tomato"] for row in rows], width=width, label="CO2 kg/kg")
    axes[1].set_xticks(x, names, rotation=30, ha="right")
    axes[1].set_title("Resource intensity per tomato kg")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--compartments", nargs="+", default=COMPARTMENTS)
    parser.add_argument("--prefix", default="agc_real_economics_by_compartment")
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
    rows = build_rows(data_root, args.compartments)

    csv_path = project_root / cfg.control_summaries_dir / f"{args.prefix}.csv"
    md_path = project_root / cfg.control_summaries_dir / f"{args.prefix}.md"
    fig_path = project_root / cfg.control_figures_dir / f"{args.prefix}.png"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    plot(fig_path, rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
