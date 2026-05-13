# -*- coding: utf-8 -*-
"""Generate Chinese supervisor-report figures from real experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = PROJECT_ROOT / "results" / "control" / "summaries"
FORECAST_ANALYSIS_DIR = PROJECT_ROOT / "results" / "forecasting" / "analysis"
FIGURE_DIR = PROJECT_ROOT / "results" / "control" / "figures"

MODEL_SELECTION_CSV = FORECAST_ANALYSIS_DIR / "fctv_final_multistart_model_rankings_reference.csv"
RESOURCE_SENSITIVITY_CSV = SUMMARY_DIR / "mainline_real_resource_sensitivity.csv"

FIGURE_1 = FIGURE_DIR / "supervisor_fig1_model_selection_cn.png"
FIGURE_2 = FIGURE_DIR / "supervisor_fig2_resource_economic_cn.png"
FIGURE_2_TRADEOFF = FIGURE_DIR / "supervisor_fig2_w005_tradeoff_cn.png"
FIGURE_2_COMBINED = FIGURE_DIR / "supervisor_fig2_combined_tradeoff_resource_cn.png"

MODEL_NAME_NOTE = "均衡模型：current_hybrid_transformer\nCO2专项模型：itransformer_co2_residual"


DISPLAY_NAMES = {
    "dlinear_forecaster": "线性基准",
    "gru_forecaster": "GRU",
    "segrnn_forecaster": "SegRNN",
    "transformer_forecaster": "普通Transformer",
    "current_hybrid_transformer": "均衡模型",
    "transformer_hybrid_residual": "混合残差",
    "patchtst_residual": "PatchTST残差",
    "itransformer_residual": "iTransformer残差",
    "itransformer_co2_residual": "CO2专项模型",
    "itransformer_co2_late_residual": "CO2后段残差",
    "itransformer_co2_control_aware_fusion": "CO2控制感知融合",
}

REPRESENTATIVE_MODELS = list(DISPLAY_NAMES.keys())
MAIN_MODELS = ["current_hybrid_transformer", "itransformer_co2_residual"]


def configure_font() -> None:
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def save_model_selection_figure() -> None:
    df = pd.read_csv(MODEL_SELECTION_CSV)
    agg = (
        df.groupby("predictor", as_index=False)
        .agg(
            objective_mean=("objective_mean", "mean"),
            objective_std=("objective_mean", "std"),
            co2_mae_mean=("co2_mae", "mean"),
            co2_mae_std=("co2_mae", "std"),
            tair_mae_mean=("tair_mae", "mean"),
            rhair_mae_mean=("rhair_mae", "mean"),
        )
    )
    plot_df = agg[agg["predictor"].isin(REPRESENTATIVE_MODELS)].copy()
    plot_df["display"] = plot_df["predictor"].map(DISPLAY_NAMES)
    plot_df = plot_df.sort_values("objective_mean")
    main_df = agg[agg["predictor"].isin(MAIN_MODELS)].copy()
    main_df["display"] = main_df["predictor"].map(DISPLAY_NAMES)

    colors = {
        "current_hybrid_transformer": "#1f77b4",
        "itransformer_co2_residual": "#2ca02c",
    }

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.0))

    ax = axes[0]
    for _, row in plot_df.iterrows():
        predictor = row["predictor"]
        highlight = predictor in MAIN_MODELS
        ax.scatter(
            row["objective_mean"],
            row["co2_mae_mean"],
            s=150 if highlight else 70,
            color=colors.get(predictor, "#9aa0a6"),
            edgecolor="black" if highlight else "white",
            linewidth=1.4 if highlight else 0.5,
            zorder=3 if highlight else 2,
        )
        ax.annotate(
            row["display"],
            (row["objective_mean"], row["co2_mae_mean"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9 if highlight else 8,
            fontweight="bold" if highlight else "normal",
        )
    ax.set_title("闭环验证后的代表模型位置（越靠左下越好）", fontsize=13)
    ax.set_xlabel("平均综合控制目标（越低越好）")
    ax.set_ylabel("平均二氧化碳误差（越低越好）")
    ax.grid(alpha=0.25)

    ax = axes[1]
    x = np.arange(len(main_df))
    width = 0.36
    main_df = main_df.set_index("predictor").loc[MAIN_MODELS].reset_index()
    bars1 = ax.bar(
        x - width / 2,
        main_df["objective_mean"],
        width=width,
        color="#1f77b4",
        label="综合控制目标",
    )
    ax2 = ax.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        main_df["co2_mae_mean"],
        width=width,
        color="#2ca02c",
        label="二氧化碳误差",
    )
    ax.set_xticks(x, main_df["display"])
    ax.set_title("最终保留的两个主模型对比", fontsize=13)
    ax.set_ylabel("综合控制目标（越低越好）")
    ax2.set_ylabel("二氧化碳误差（越低越好）")
    ax.grid(axis="y", alpha=0.25)
    for bars, axis in [(bars1, ax), (bars2, ax2)]:
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}" if height < 1 else f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    lines = [bars1, bars2]
    labels = [item.get_label() for item in lines]
    ax.legend(lines, labels, loc="upper left")

    fig.suptitle(
        "图一：闭环控制验证后，从代表模型中收束到两个主模型",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.985,
        MODEL_NAME_NOTE,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7fbff", "edgecolor": "#4a6fa5"},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_1, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_resource_economic_figure() -> None:
    df = pd.read_csv(RESOURCE_SENSITIVITY_CSV)
    df = df[df["predictor"].isin(MAIN_MODELS)].copy()
    df["display"] = df["predictor"].map(DISPLAY_NAMES)
    df = df.sort_values(["predictor", "resource_weight"])

    colors = {
        "current_hybrid_transformer": "#1f77b4",
        "itransformer_co2_residual": "#2ca02c",
    }
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.0))
    axes = axes.reshape(-1)

    ax = axes[0]
    for predictor, group in df.groupby("predictor"):
        ax.plot(
            group["resource_weight"],
            group["estimated_total_resource_cost_eur_m2_mean"],
            marker="o",
            linewidth=2.4,
            color=colors[predictor],
            label=DISPLAY_NAMES[predictor],
        )
    ax.set_title("资源惩罚权重越大，估计资源成本越低")
    ax.set_xlabel("资源消耗惩罚权重")
    ax.set_ylabel("估计资源成本（欧元/平方米）")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    for predictor, group in df.groupby("predictor"):
        ax.plot(
            group["resource_weight"],
            group["co2_mae_mean"],
            marker="o",
            linewidth=2.4,
            color=colors[predictor],
            label=DISPLAY_NAMES[predictor],
        )
    ax.set_title("节约资源会带来二氧化碳控制代价")
    ax.set_xlabel("资源消耗惩罚权重")
    ax.set_ylabel("二氧化碳误差（越低越好）")
    ax.grid(alpha=0.25)

    ax = axes[2]
    w005 = df[np.isclose(df["resource_weight"], 0.05)].set_index("predictor").loc[MAIN_MODELS]
    metrics = [
        ("estimated_heat_mj_m2_mean", "热量"),
        ("estimated_electricity_kwh_m2_mean", "电力"),
        ("estimated_co2_kg_m2_mean", "二氧化碳"),
        ("estimated_irrigation_l_m2_mean", "灌溉水"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    for idx, predictor in enumerate(MAIN_MODELS):
        values = [float(w005.loc[predictor, key]) for key, _ in metrics]
        base = [float(w005.loc["current_hybrid_transformer", key]) for key, _ in metrics]
        ratio = [v / b if abs(b) > 1e-12 else np.nan for v, b in zip(values, base)]
        ax.bar(
            x + (idx - 0.5) * width,
            ratio,
            width=width,
            color=colors[predictor],
            label=DISPLAY_NAMES[predictor],
        )
    ax.axhline(1.0, color="black", linewidth=1.1, linestyle="--")
    ax.set_xticks(x, [name for _, name in metrics])
    ax.set_title("w=0.05 下的资源消耗对比（相对均衡模型）")
    ax.set_ylabel("相对消耗比例")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[3]
    for predictor, group in df.groupby("predictor"):
        ax.plot(
            group["estimated_total_resource_cost_eur_m2_mean"],
            group["co2_mae_mean"],
            marker="o",
            linewidth=2.2,
            color=colors[predictor],
            label=DISPLAY_NAMES[predictor],
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"w={row['resource_weight']:.2f}",
                (row["estimated_total_resource_cost_eur_m2_mean"], row["co2_mae_mean"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title("二氧化碳控制与资源成本的权衡")
    ax.set_xlabel("估计资源成本（越低越好）")
    ax.set_ylabel("二氧化碳误差（越低越好）")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "图二：加入官方经济规则后，两个主模型的估计资源成本与控制代价",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_2, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_w005_tradeoff_figure() -> None:
    df = pd.read_csv(RESOURCE_SENSITIVITY_CSV)
    df = df[df["predictor"].isin(MAIN_MODELS)].copy()
    df = df.sort_values(["predictor", "resource_weight"])
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "current_hybrid_transformer": "#1f77b4",
        "itransformer_co2_residual": "#2ca02c",
    }

    fig, ax_cost = plt.subplots(figsize=(13.8, 7.2))
    ax_co2 = ax_cost.twinx()

    for predictor, group in df.groupby("predictor"):
        label = DISPLAY_NAMES[predictor]
        color = colors[predictor]
        ax_cost.plot(
            group["resource_weight"],
            group["estimated_total_resource_cost_eur_m2_mean"],
            marker="o",
            markersize=9,
            linewidth=3.0,
            color=color,
            linestyle="-",
            label=f"{label}：估计资源成本",
        )
        ax_co2.plot(
            group["resource_weight"],
            group["co2_mae_mean"],
            marker="s",
            markersize=8,
            linewidth=2.6,
            color=color,
            linestyle="--",
            alpha=0.82,
            label=f"{label}：二氧化碳误差",
        )

    ax_cost.axvspan(0.045, 0.055, color="#f2c94c", alpha=0.28)
    ax_cost.axvline(0.05, color="#b7791f", linewidth=2.0, linestyle=":")

    co2_model = df[df["predictor"] == "itransformer_co2_residual"].copy()
    base = co2_model[np.isclose(co2_model["resource_weight"], 0.0)].iloc[0]
    w005 = co2_model[np.isclose(co2_model["resource_weight"], 0.05)].iloc[0]
    w008 = co2_model[np.isclose(co2_model["resource_weight"], 0.08)].iloc[0]
    cost_drop_005 = (base["estimated_total_resource_cost_eur_m2_mean"] - w005["estimated_total_resource_cost_eur_m2_mean"]) / base["estimated_total_resource_cost_eur_m2_mean"] * 100.0
    co2_increase_005 = (w005["co2_mae_mean"] - base["co2_mae_mean"]) / base["co2_mae_mean"] * 100.0
    cost_drop_008 = (base["estimated_total_resource_cost_eur_m2_mean"] - w008["estimated_total_resource_cost_eur_m2_mean"]) / base["estimated_total_resource_cost_eur_m2_mean"] * 100.0
    co2_increase_008 = (w008["co2_mae_mean"] - base["co2_mae_mean"]) / base["co2_mae_mean"] * 100.0

    ax_cost.annotate(
        "推荐折中点：w=0.05\n"
        f"CO2专项模型成本下降 {cost_drop_005:.1f}%\n"
        f"CO2误差上升 {co2_increase_005:.1f}%",
        xy=(0.05, w005["estimated_total_resource_cost_eur_m2_mean"]),
        xytext=(0.032, w005["estimated_total_resource_cost_eur_m2_mean"] + 0.00125),
        arrowprops={"arrowstyle": "->", "color": "#7a4b00", "linewidth": 1.5},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7d6", "edgecolor": "#b7791f"},
        fontsize=12,
    )
    ax_co2.annotate(
        "w=0.08 更省成本，\n但控制代价更大\n"
        f"成本下降 {cost_drop_008:.1f}%\n"
        f"CO2误差上升 {co2_increase_008:.1f}%",
        xy=(0.08, w008["co2_mae_mean"]),
        xytext=(0.058, w008["co2_mae_mean"] + 2.2),
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.4},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f4f4", "edgecolor": "#777777"},
        fontsize=11,
    )

    ax_cost.set_title(
        "为什么选择 w=0.05：资源成本下降与二氧化碳控制代价的折中",
        fontsize=16,
        fontweight="bold",
    )
    ax_cost.set_xlabel("资源消耗惩罚权重 w（越大越重视节约资源）", fontsize=12)
    ax_cost.set_ylabel("估计资源成本（欧元/平方米，越低越好）", fontsize=12)
    ax_co2.set_ylabel("二氧化碳误差（越低越好）", fontsize=12)
    ax_cost.grid(alpha=0.25)
    ax_cost.set_xticks([0.00, 0.02, 0.05, 0.08])
    ax_cost.set_xlim(-0.005, 0.087)

    lines1, labels1 = ax_cost.get_legend_handles_labels()
    lines2, labels2 = ax_co2.get_legend_handles_labels()
    ax_cost.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.23)
    fig.savefig(FIGURE_2_TRADEOFF, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_combined_tradeoff_resource_figure() -> None:
    df = pd.read_csv(RESOURCE_SENSITIVITY_CSV)
    df = df[df["predictor"].isin(MAIN_MODELS)].copy()
    df = df.sort_values(["predictor", "resource_weight"])

    colors = {
        "current_hybrid_transformer": "#1f77b4",
        "itransformer_co2_residual": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(17.2, 7.2))
    ax_cost = axes[0]
    ax_co2 = ax_cost.twinx()

    for predictor, group in df.groupby("predictor"):
        label = DISPLAY_NAMES[predictor]
        color = colors[predictor]
        ax_cost.plot(
            group["resource_weight"],
            group["estimated_total_resource_cost_eur_m2_mean"],
            marker="o",
            markersize=8,
            linewidth=2.8,
            color=color,
            linestyle="-",
            label=f"{label}：估计资源成本",
        )
        ax_co2.plot(
            group["resource_weight"],
            group["co2_mae_mean"],
            marker="s",
            markersize=7,
            linewidth=2.3,
            color=color,
            linestyle="--",
            alpha=0.82,
            label=f"{label}：二氧化碳误差",
        )

    ax_cost.axvspan(0.045, 0.055, color="#f2c94c", alpha=0.26)
    ax_cost.axvline(0.05, color="#b7791f", linewidth=1.8, linestyle=":")

    co2_model = df[df["predictor"] == "itransformer_co2_residual"].copy()
    base = co2_model[np.isclose(co2_model["resource_weight"], 0.0)].iloc[0]
    w005 = co2_model[np.isclose(co2_model["resource_weight"], 0.05)].iloc[0]
    w008 = co2_model[np.isclose(co2_model["resource_weight"], 0.08)].iloc[0]
    cost_drop_005 = (base["estimated_total_resource_cost_eur_m2_mean"] - w005["estimated_total_resource_cost_eur_m2_mean"]) / base["estimated_total_resource_cost_eur_m2_mean"] * 100.0
    co2_increase_005 = (w005["co2_mae_mean"] - base["co2_mae_mean"]) / base["co2_mae_mean"] * 100.0
    cost_drop_008 = (base["estimated_total_resource_cost_eur_m2_mean"] - w008["estimated_total_resource_cost_eur_m2_mean"]) / base["estimated_total_resource_cost_eur_m2_mean"] * 100.0
    co2_increase_008 = (w008["co2_mae_mean"] - base["co2_mae_mean"]) / base["co2_mae_mean"] * 100.0

    ax_cost.annotate(
        "推荐折中点：w=0.05\n"
        f"CO2专项模型成本下降 {cost_drop_005:.1f}%\n"
        f"CO2误差上升 {co2_increase_005:.1f}%",
        xy=(0.05, w005["estimated_total_resource_cost_eur_m2_mean"]),
        xytext=(0.029, w005["estimated_total_resource_cost_eur_m2_mean"] + 0.0010),
        arrowprops={"arrowstyle": "->", "color": "#7a4b00", "linewidth": 1.3},
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#fff7d6", "edgecolor": "#b7791f"},
        fontsize=10,
    )
    ax_co2.annotate(
        "w=0.08 更省成本，\n但控制代价更大\n"
        f"成本下降 {cost_drop_008:.1f}%\n"
        f"CO2误差上升 {co2_increase_008:.1f}%",
        xy=(0.08, w008["co2_mae_mean"]),
        xytext=(0.058, w008["co2_mae_mean"] + 2.0),
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.2},
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#f4f4f4", "edgecolor": "#777777"},
        fontsize=10,
    )
    ax_cost.set_title("子图一：为什么选择 w=0.05", fontsize=13)
    ax_cost.set_xlabel("资源消耗惩罚权重 w")
    ax_cost.set_ylabel("估计资源成本（欧元/平方米，越低越好）")
    ax_co2.set_ylabel("二氧化碳误差（越低越好）")
    ax_cost.grid(alpha=0.25)
    ax_cost.set_xticks([0.00, 0.02, 0.05, 0.08])
    ax_cost.set_xlim(-0.005, 0.087)

    lines1, labels1 = ax_cost.get_legend_handles_labels()
    lines2, labels2 = ax_co2.get_legend_handles_labels()
    ax_cost.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=8, ncol=1)

    ax = axes[1]
    w005_df = df[np.isclose(df["resource_weight"], 0.05)].set_index("predictor").loc[MAIN_MODELS]
    metrics = [
        ("estimated_heat_mj_m2_mean", "热量"),
        ("estimated_electricity_kwh_m2_mean", "电力"),
        ("estimated_co2_kg_m2_mean", "二氧化碳"),
        ("estimated_irrigation_l_m2_mean", "灌溉水"),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    for idx, predictor in enumerate(MAIN_MODELS):
        values = [float(w005_df.loc[predictor, key]) for key, _ in metrics]
        base_values = [float(w005_df.loc["current_hybrid_transformer", key]) for key, _ in metrics]
        ratio = [v / b if abs(b) > 1e-12 else np.nan for v, b in zip(values, base_values)]
        bars = ax.bar(
            x + (idx - 0.5) * width,
            ratio,
            width=width,
            color=colors[predictor],
            label=DISPLAY_NAMES[predictor],
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.axhline(1.0, color="black", linewidth=1.1, linestyle="--")
    ax.set_xticks(x, [name for _, name in metrics])
    ax.set_ylim(0.0, 1.15)
    ax.set_title("子图二：w=0.05 下两个模型的估计资源消耗对比", fontsize=13)
    ax.set_ylabel("相对消耗比例（均衡模型 = 1.00）")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "图二：资源惩罚权重选择与 w=0.05 下的资源消耗对比",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.985,
        MODEL_NAME_NOTE,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7fbff", "edgecolor": "#4a6fa5"},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_2_COMBINED, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_font()
    save_model_selection_figure()
    save_resource_economic_figure()
    save_w005_tradeoff_figure()
    save_combined_tradeoff_resource_figure()
    print(f"Saved: {FIGURE_1}")
    print(f"Saved: {FIGURE_2}")
    print(f"Saved: {FIGURE_2_TRADEOFF}")
    print(f"Saved: {FIGURE_2_COMBINED}")


if __name__ == "__main__":
    main()
