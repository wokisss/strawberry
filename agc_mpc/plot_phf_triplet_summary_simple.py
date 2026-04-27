# -*- coding: utf-8 -*-
"""Build a simple paper-ready PHF triplet comparison figure from saved summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    "itransformer_co2_control_aware_fusion",
    "itransformer_co2_late_frozen_expert",
    "itransformer_co2_horizon_mixture",
]

LABELS = {
    "itransformer_co2_control_aware_fusion": "Control-aware\nfusion",
    "itransformer_co2_late_frozen_expert": "Late frozen\nexpert",
    "itransformer_co2_horizon_mixture": "Horizon\nmixture",
}

COLORS = {
    "itransformer_co2_control_aware_fusion": "#2c6fbb",
    "itransformer_co2_late_frozen_expert": "#1f7a4d",
    "itransformer_co2_horizon_mixture": "#c0392b",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ranked_record(project_root: Path, model: str, compartment: str) -> dict:
    payload = _load_json(
        project_root / "results" / "forecasting" / "analysis" / f"control_relevant_validation_{compartment.lower()}.json"
    )
    for record in payload.get("ranked_summary", []):
        if record.get("predictor") == model:
            return record
    raise KeyError(f"Missing ranked control-relevant record for {model}")


def _load_forecast_summary(project_root: Path, model: str, compartment: str) -> dict:
    return _load_json(
        project_root / "results" / "forecasting" / "analysis" / f"{model}_joint_all_{compartment.lower()}_summary.json"
    )


def _load_control_summary(project_root: Path, model: str) -> dict:
    return _load_json(project_root / "results" / "control" / "summaries" / f"{model}_gradient_mpc_summary.json")


def _annotate(ax, bars, fmt: str = "{:.2f}", fontsize: int = 8) -> None:
    for bar in bars:
        h = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _lighten_color(hex_color: str, amount: float = 0.45) -> tuple[float, float, float]:
    color = hex_color.lstrip("#")
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    return tuple(channel + (1.0 - channel) * amount for channel in (r, g, b))


def build_figure(project_root: Path, compartment: str, output: Path, summary_json: Path) -> None:
    forecast = {m: _load_forecast_summary(project_root, m, compartment) for m in MODELS}
    control = {m: _load_control_summary(project_root, m) for m in MODELS}
    ranked = {m: _load_ranked_record(project_root, m, compartment) for m in MODELS}

    full_mae = [forecast[m]["metrics_by_target"]["CO2air"]["full_mae"] for m in MODELS]
    final_mae = [forecast[m]["metrics_by_target"]["CO2air"]["final_mae"] for m in MODELS]
    first_step_mae = [ranked[m]["co2_first_step_mae"] for m in MODELS]
    first6_mae = [ranked[m]["co2_control_horizon_mae"] for m in MODELS]
    closed_loop_co2 = [control[m]["target_mae"]["CO2air"] for m in MODELS]
    objective = [control[m]["objective_mean"] for m in MODELS]
    mean_rank = [ranked[m]["control_relevant_mean_rank"] for m in MODELS]

    late_ctrl = control["itransformer_co2_late_frozen_expert"]["target_mae"]["CO2air"]
    horizon_final = forecast["itransformer_co2_horizon_mixture"]["metrics_by_target"]["CO2air"]["final_mae"]
    gap_to_late_ctrl = [value - late_ctrl for value in closed_loop_co2]
    gap_to_horizon_final = [value - horizon_final for value in final_mae]

    x = np.arange(len(MODELS))
    model_labels = [LABELS[m] for m in MODELS]
    model_colors = [COLORS[m] for m in MODELS]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    width = 0.34
    ax = axes[0, 0]
    bars1 = ax.bar(
        x - width / 2,
        first_step_mae,
        width,
        color=[_lighten_color(COLORS[m], 0.45) for m in MODELS],
        edgecolor=model_colors,
        linewidth=1.0,
        label="CO2 first-step MAE",
    )
    bars2 = ax.bar(
        x + width / 2,
        first6_mae,
        width,
        color=model_colors,
        hatch="//",
        alpha=0.95,
        label="CO2 first-6-step MAE",
    )
    _annotate(ax, bars1)
    _annotate(ax, bars2)
    ax.set_xticks(x, model_labels)
    ax.set_title("Short-horizon control-relevant forecast")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.03,
        0.96,
        "Lower is better.\nLight bar = first-step MAE; hatched bar = first-6-step MAE.\nFusion should match Late frozen here.",
        transform=ax.transAxes,
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "alpha": 0.58, "edgecolor": "#cccccc"},
    )

    ax = axes[0, 1]
    bars1 = ax.bar(
        x - width / 2,
        full_mae,
        width,
        color=[_lighten_color(COLORS[m], 0.45) for m in MODELS],
        edgecolor=model_colors,
        linewidth=1.0,
        label="CO2 full MAE",
    )
    bars2 = ax.bar(
        x + width / 2,
        final_mae,
        width,
        color=model_colors,
        hatch="//",
        alpha=0.95,
        label="CO2 final MAE",
    )
    _annotate(ax, bars1)
    _annotate(ax, bars2)
    ax.set_xticks(x, model_labels)
    ax.set_title("Offline CO2 forecasting")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.03,
        0.96,
        "Lower is better.\nLight bar = full MAE; hatched bar = final MAE.\nFusion should match Horizon mix here.",
        transform=ax.transAxes,
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "alpha": 0.58, "edgecolor": "#cccccc"},
    )

    ax = axes[1, 0]
    bars = ax.bar(x, closed_loop_co2, color=model_colors, width=0.55, label="GradientMPC CO2 MAE")
    _annotate(ax, bars)
    ax.set_xticks(x, model_labels)
    ax.set_title("Closed-loop control transfer")
    ax.set_ylabel("GradientMPC CO2 MAE")
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        x,
        objective,
        color="#222222",
        marker="D",
        linewidth=1.8,
        markersize=6,
        label="Closed-loop objective",
    )
    for idx, value in enumerate(objective):
        ax2.text(
            x[idx] + 0.03,
            value + max(objective) * 0.015,
            f"{value:.3f}",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#222222",
        )
    ax2.set_ylabel("Objective")
    ax2.set_ylim(0.0, max(objective) * 1.18)
    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.8, bar.get_height() * 0.08),
            f"rank={mean_rank[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white" if idx != 1 else "black",
            fontweight="bold",
        )
    ax.text(
        0.03,
        0.96,
        "Lower is better.\nBar = GradientMPC CO2 MAE; black line = objective; bar text = rank.\nThis is the main control transfer check.",
        transform=ax.transAxes,
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "alpha": 0.52, "edgecolor": "#cccccc"},
    )

    ax = axes[1, 1]
    bars1 = ax.bar(
        x - width / 2,
        gap_to_late_ctrl,
        width,
        color=[_lighten_color(COLORS[m], 0.45) for m in MODELS],
        edgecolor=model_colors,
        linewidth=1.0,
        label="Gap to Late frozen control",
    )
    bars2 = ax.bar(
        x + width / 2,
        gap_to_horizon_final,
        width,
        color=model_colors,
        hatch="//",
        alpha=0.95,
        label="Gap to Horizon final forecast",
    )
    _annotate(ax, bars1)
    _annotate(ax, bars2)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x, model_labels)
    ax.set_title("Best-of-both check")
    ax.set_ylabel("Gap to leader")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.03,
        0.96,
        "Lower is better.\nLight bar = gap to Late frozen control; hatched bar = gap to Horizon final forecast.\nFusion should keep both gaps small.",
        transform=ax.transAxes,
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "alpha": 0.58, "edgecolor": "#cccccc"},
    )

    fig.suptitle(
        "PHF Triplet Summary: why Control-aware fusion is the report model",
        fontsize=16,
        y=0.99,
    )
    legend_handles = [
        plt.Line2D([0], [0], color=COLORS[m], linewidth=8, label=LABELS[m].replace("\n", " "))
        for m in MODELS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
        fontsize=10,
        title="Model colors",
        title_fontsize=10,
    )
    fig.text(
        0.5,
        0.005,
        "Takeaway: Control-aware fusion keeps the short-horizon control behavior of Late frozen expert, "
        "while recovering most of the offline CO2 forecasting benefit of Horizon mixture.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.965))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_payload = {
        "compartment": compartment,
        "models": MODELS,
        "metrics": {
            model: {
                "co2_first_step_mae": ranked[model]["co2_first_step_mae"],
                "co2_first_6_step_mae": ranked[model]["co2_control_horizon_mae"],
                "co2_full_mae": forecast[model]["metrics_by_target"]["CO2air"]["full_mae"],
                "co2_final_mae": forecast[model]["metrics_by_target"]["CO2air"]["final_mae"],
                "closed_loop_co2_mae": control[model]["target_mae"]["CO2air"],
                "objective_mean": control[model]["objective_mean"],
                "control_relevant_mean_rank": ranked[model]["control_relevant_mean_rank"],
                "gap_to_late_frozen_control": control[model]["target_mae"]["CO2air"] - late_ctrl,
                "gap_to_horizon_final_forecast": forecast[model]["metrics_by_target"]["CO2air"]["final_mae"] - horizon_final,
            }
            for model in MODELS
        },
        "figure_path": str(output.relative_to(project_root)),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument(
        "--output",
        default="results/forecasting/figures/comparisons/phf_triplet_summary_simple.png",
    )
    parser.add_argument(
        "--summary-json",
        default="results/forecasting/analysis/phf_triplet_summary_simple.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    build_figure(project_root, args.compartment, project_root / args.output, project_root / args.summary_json)
    print(f"Saved figure: {project_root / args.output}")
    print(f"Saved summary: {project_root / args.summary_json}")


if __name__ == "__main__":
    main()
