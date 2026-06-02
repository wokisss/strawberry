# -*- coding: utf-8 -*-
"""Generate report figures for anchored MPC and AGC resource baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = PROJECT_ROOT / "results" / "control" / "summaries"
FIGURE_DIR = PROJECT_ROOT / "results" / "control" / "figures"

ANCHOR_SUMMARY = SUMMARY_DIR / "full_period_anchored_resource_mpc_summary.csv"
ALL_TEAM_RESOURCE = SUMMARY_DIR / "agc_same_period_all_teams_resource_baselines.csv"
ALL_TEAM_INTENSITY = SUMMARY_DIR / "agc_same_period_all_teams_resource_intensity.csv"
ALL_TEAM_ECONOMIC = SUMMARY_DIR / "agc_same_period_all_teams_economic_context.csv"

FIGURE_SUMMARY = FIGURE_DIR / "resource_report_fig1_mpc_tradeoff.png"
FIGURE_BASELINE = FIGURE_DIR / "resource_report_fig2_all_team_resource_baseline.png"
FIGURE_INTENSITY = FIGURE_DIR / "resource_report_fig3_real_team_resource_intensity.png"
FIGURE_ECONOMIC = FIGURE_DIR / "resource_report_fig4_real_team_economic_context.png"
FIGURE_DASHBOARD = FIGURE_DIR / "resource_report_fig5_summary_dashboard.png"


MODEL_LABELS = {
    "current_hybrid_transformer": "Balanced MPC",
    "itransformer_co2_residual": "CO2-specialist MPC",
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor = pd.read_csv(ANCHOR_SUMMARY)
    resource = pd.read_csv(ALL_TEAM_RESOURCE)
    intensity = pd.read_csv(ALL_TEAM_INTENSITY)
    economic = pd.read_csv(ALL_TEAM_ECONOMIC)
    return anchor, resource, intensity, economic


def _case_label(row: pd.Series) -> str:
    if row["source_type"] == "real_agc_executed_resource":
        return str(row["compartment"])
    return f"{MODEL_LABELS.get(row['predictor'], row['predictor'])}\nw={row['resource_weight']:.2f}"


def save_mpc_tradeoff(anchor: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    colors = {
        "current_hybrid_transformer": "#1f77b4",
        "itransformer_co2_residual": "#2ca02c",
    }

    ax = axes[0]
    for predictor, group in anchor.groupby("predictor"):
        group = group.sort_values("resource_weight")
        ax.plot(
            group["resource_weight"],
            group["estimated_total_resource_cost_eur_m2"],
            marker="o",
            linewidth=2.5,
            color=colors[predictor],
            label=MODEL_LABELS[predictor],
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['estimated_total_resource_cost_eur_m2']:.4f}",
                (row["resource_weight"], row["estimated_total_resource_cost_eur_m2"]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set_title("Estimated resource cost decreases at w=0.05")
    ax.set_xlabel("Resource weight")
    ax.set_ylabel("Estimated resource cost, EUR/m2")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    for predictor, group in anchor.groupby("predictor"):
        group = group.sort_values("resource_weight")
        ax.plot(
            group["resource_weight"],
            group["co2_mae"],
            marker="o",
            linewidth=2.5,
            color=colors[predictor],
            label=MODEL_LABELS[predictor],
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['co2_mae']:.2f}",
                (row["resource_weight"], row["co2_mae"]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=9,
            )
    ax.set_title("CO2 tracking cost is explicit")
    ax.set_xlabel("Resource weight")
    ax.set_ylabel("CO2air MAE")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("Full-period anchored MPC: resource-control trade-off", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_SUMMARY, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_all_team_baseline(resource: pd.DataFrame) -> None:
    order = resource.sort_values("resource_cost_eur_m2").copy()
    labels = [_case_label(row) for _, row in order.iterrows()]
    colors = ["#8aa6c1" if source == "real_agc_executed_resource" else "#f2a65a" for source in order["source_type"]]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    bars = ax.bar(x, order["resource_cost_eur_m2"], color=colors, edgecolor="white")
    ax.axhline(
        float(resource[(resource["source_type"] == "real_agc_executed_resource") & (resource["compartment"] == "Reference")]["resource_cost_eur_m2"].iloc[0]),
        color="#444444",
        linestyle="--",
        linewidth=1.4,
        label="Real Reference",
    )
    ax.axhline(
        float(resource[(resource["source_type"] == "real_agc_executed_resource") & (resource["compartment"] == "AICU")]["resource_cost_eur_m2"].iloc[0]),
        color="#2ca02c",
        linestyle=":",
        linewidth=1.8,
        label="Real AICU",
    )
    for bar, value in zip(bars, order["resource_cost_eur_m2"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Resource cost, EUR/m2")
    ax.set_title("Same-window AGC real resources and MPC estimated-resource references")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_BASELINE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_intensity(intensity: pd.DataFrame) -> None:
    df = intensity.sort_values("resource_cost_eur_per_kg_tomato")
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    bars = ax.bar(x, df["resource_cost_eur_per_kg_tomato"], color="#4c956c")
    ax.set_xticks(x, df["compartment"], rotation=30, ha="right")
    ax.set_ylabel("EUR/kg tomato")
    ax.set_title("Real AGC resource cost intensity")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, df["resource_cost_eur_per_kg_tomato"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    width = 0.25
    ax.bar(x - width, df["heat_mj_per_kg_tomato"], width=width, label="Heat MJ/kg", color="#c65d3a")
    ax.bar(x, df["co2_kg_per_kg_tomato"], width=width, label="CO2 kg/kg", color="#5f8dd3")
    ax.bar(x + width, df["irrigation_l_per_kg_tomato"], width=width, label="Irrigation L/kg", color="#78a55a")
    ax.set_xticks(x, df["compartment"], rotation=30, ha="right")
    ax.set_title("Physical resource intensities")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Real AGC resource intensity per kg tomato", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_INTENSITY, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_economic(economic: pd.DataFrame) -> None:
    df = economic.sort_values("same_period_margin_excl_fixed_eur_m2", ascending=False)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    width = 0.26
    ax.bar(x - width, df["estimated_income_eur_m2"], width=width, label="Income", color="#4c78a8")
    ax.bar(x, df["variable_cost_excl_fixed_eur_m2"], width=width, label="Variable cost excl. fixed", color="#f58518")
    ax.bar(x + width, df["same_period_margin_excl_fixed_eur_m2"], width=width, label="Margin excl. fixed", color="#54a24b")
    ax.set_xticks(x, df["compartment"], rotation=30, ha="right")
    ax.set_ylabel("EUR/m2")
    ax.set_title("Same-window real AGC economic context")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ECONOMIC, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_dashboard(anchor: pd.DataFrame, resource: pd.DataFrame, economic: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    axes = axes.reshape(-1)

    # Panel 1: MPC cost reduction.
    ax = axes[0]
    rows = anchor[anchor["resource_weight"].isin([0.0, 0.05])].copy()
    piv = rows.pivot(index="predictor", columns="resource_weight", values="estimated_total_resource_cost_eur_m2")
    labels = [MODEL_LABELS[idx] for idx in piv.index]
    x = np.arange(len(piv))
    width = 0.35
    ax.bar(x - width / 2, piv[0.0], width=width, label="w=0.00", color="#9aa0a6")
    ax.bar(x + width / 2, piv[0.05], width=width, label="w=0.05", color="#f2a65a")
    for idx, predictor in enumerate(piv.index):
        change = (piv.loc[predictor, 0.05] - piv.loc[predictor, 0.0]) / piv.loc[predictor, 0.0] * 100.0
        ax.text(idx, max(piv.loc[predictor]) + 0.008, f"{change:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("EUR/m2")
    ax.set_title("MPC estimated resource reduction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    # Panel 2: resource baseline ranking.
    ax = axes[1]
    selected = resource[
        (resource["source_type"] == "real_agc_executed_resource")
        | ((resource["source_type"] == "counterfactual_estimated_mpc_resource") & np.isclose(resource["resource_weight"], 0.05))
    ].copy()
    selected = selected.sort_values("resource_cost_eur_m2")
    labels = [_case_label(row) for _, row in selected.iterrows()]
    colors = ["#8aa6c1" if source == "real_agc_executed_resource" else "#f2a65a" for source in selected["source_type"]]
    x = np.arange(len(selected))
    ax.bar(x, selected["resource_cost_eur_m2"], color=colors)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("EUR/m2")
    ax.set_title("Same-window resource baseline")
    ax.grid(axis="y", alpha=0.25)

    # Panel 3: CO2 tracking for MPC.
    ax = axes[2]
    for predictor, group in anchor.groupby("predictor"):
        group = group.sort_values("resource_weight")
        ax.plot(group["resource_weight"], group["co2_mae"], marker="o", linewidth=2.3, label=MODEL_LABELS[predictor])
    ax.set_xlabel("Resource weight")
    ax.set_ylabel("CO2air MAE")
    ax.set_title("CO2 tracking trade-off")
    ax.grid(alpha=0.25)
    ax.legend()

    # Panel 4: real economic context.
    ax = axes[3]
    econ = economic.sort_values("same_period_margin_excl_fixed_eur_m2", ascending=False)
    x = np.arange(len(econ))
    ax.bar(x, econ["same_period_margin_excl_fixed_eur_m2"], color="#54a24b")
    ax.set_xticks(x, econ["compartment"], rotation=30, ha="right")
    ax.set_ylabel("EUR/m2")
    ax.set_title("Real AGC margin context, excl. fixed cost")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Resource-aware MPC and same-window AGC baselines", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DASHBOARD, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    anchor, resource, intensity, economic = _load()
    save_mpc_tradeoff(anchor)
    save_all_team_baseline(resource)
    save_intensity(intensity)
    save_economic(economic)
    save_dashboard(anchor, resource, economic)
    for path in [FIGURE_SUMMARY, FIGURE_BASELINE, FIGURE_INTENSITY, FIGURE_ECONOMIC, FIGURE_DASHBOARD]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
