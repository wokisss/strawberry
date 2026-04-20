# -*- coding: utf-8 -*-
"""Build trace-based control comparison dashboards for two MPC predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LEFT = "itransformer_co2_horizon_mixture"
DEFAULT_RIGHT = "itransformer_co2_late_frozen_expert"
DEFAULT_CONTROLLER = "gradient_mpc"

PREDICTOR_LABELS = {
    "itransformer_co2_horizon_mixture": "Offline CO2 Leader\nHorizon Mix",
    "itransformer_co2_late_frozen_expert": "Best CO2 Control\nLate Frozen Expert",
    "itransformer_co2_recoupled_expert": "Best Objective Control\nRecoupled Expert",
    "itransformer_co2_late_residual": "Late Residual",
    "itransformer_co2_frozen_backbone_horizon_mixture": "Safe Horizon Mix",
}

COLORS = {
    "itransformer_co2_horizon_mixture": "#c0392b",
    "itransformer_co2_late_frozen_expert": "#1f7a4d",
    "itransformer_co2_recoupled_expert": "#2c6fbb",
    "itransformer_co2_late_residual": "#7f8c8d",
    "itransformer_co2_frozen_backbone_horizon_mixture": "#8e44ad",
}


def _load_summary(project_root: Path, predictor: str, controller: str) -> dict:
    path = project_root / "results" / "control" / "summaries" / f"{predictor}_{controller}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing control summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_existing_figure(project_root: Path, summary: dict) -> np.ndarray:
    figure_path = Path(summary["figure_path"])
    if not figure_path.is_absolute():
        figure_path = project_root / figure_path
    if not figure_path.exists():
        raise FileNotFoundError(f"Missing control figure: {figure_path}")
    return mpimg.imread(figure_path)


def _annotate_bars(ax, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def build_comparison(left: str, right: str, controller: str, output: Path | None = None) -> Path:
    project_root = Path(__file__).resolve().parent
    left_summary = _load_summary(project_root, left, controller)
    right_summary = _load_summary(project_root, right, controller)
    left_img = _load_existing_figure(project_root, left_summary)
    right_img = _load_existing_figure(project_root, right_summary)

    left_label = PREDICTOR_LABELS.get(left, left)
    right_label = PREDICTOR_LABELS.get(right, right)
    left_color = COLORS.get(left, "#c0392b")
    right_color = COLORS.get(right, "#1f7a4d")

    targets = list(left_summary["target_mae"].keys())
    left_target = [left_summary["target_mae"][target] for target in targets]
    right_target = [right_summary["target_mae"][target] for target in targets]

    control_metrics = ["objective_mean", "control_delta_mae", "action_tv"]
    control_labels = ["Objective", "|u - logged|", "Action TV"]
    left_control = [left_summary[name] for name in control_metrics]
    right_control = [right_summary[name] for name in control_metrics]

    fig = plt.figure(figsize=(22, 18))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 0.55, 4.8],
        width_ratios=[1, 1],
        hspace=0.25,
        wspace=0.08,
    )

    ax_targets = fig.add_subplot(grid[0, 0])
    x = np.arange(len(targets))
    width = 0.36
    bars_left = ax_targets.bar(x - width / 2, left_target, width, color=left_color, label=left_label)
    bars_right = ax_targets.bar(x + width / 2, right_target, width, color=right_color, label=right_label)
    _annotate_bars(ax_targets, bars_left)
    _annotate_bars(ax_targets, bars_right)
    ax_targets.set_xticks(x, targets)
    ax_targets.set_ylabel("Closed-loop MAE")
    ax_targets.set_title("Target Tracking Error, Lower Is Better")
    ax_targets.grid(axis="y", alpha=0.25)
    ax_targets.legend(fontsize=9)

    ax_control = fig.add_subplot(grid[0, 1])
    x2 = np.arange(len(control_labels))
    bars_left = ax_control.bar(x2 - width / 2, left_control, width, color=left_color, label=left_label)
    bars_right = ax_control.bar(x2 + width / 2, right_control, width, color=right_color, label=right_label)
    _annotate_bars(ax_control, bars_left)
    _annotate_bars(ax_control, bars_right)
    ax_control.set_xticks(x2, control_labels)
    ax_control.set_ylabel("Raw metric value")
    ax_control.set_title("MPC Behavior Metrics, Lower Is Usually Better")
    ax_control.grid(axis="y", alpha=0.25)

    ax_delta = fig.add_subplot(grid[1, :])
    deltas = np.asarray(left_target, dtype=np.float32) - np.asarray(right_target, dtype=np.float32)
    delta_colors = ["#b03a2e" if value > 0 else "#2874a6" for value in deltas]
    delta_bars = ax_delta.bar(targets, deltas, color=delta_colors)
    _annotate_bars(ax_delta, delta_bars)
    ax_delta.axhline(0.0, color="#333333", linewidth=1.0)
    ax_delta.set_ylabel("Left - Right MAE")
    ax_delta.set_title("Positive Means Horizon Mix Is Worse Than The Control Leader")
    ax_delta.grid(axis="y", alpha=0.25)

    ax_left = fig.add_subplot(grid[2, 0])
    ax_left.imshow(left_img)
    ax_left.set_title(left_label.replace("\n", " | "), fontsize=12)
    ax_left.axis("off")

    ax_right = fig.add_subplot(grid[2, 1])
    ax_right.imshow(right_img)
    ax_right.set_title(right_label.replace("\n", " | "), fontsize=12)
    ax_right.axis("off")

    subtitle = (
        f"{controller} | {left}: objective={left_summary['objective_mean']:.4f}, "
        f"CO2 MAE={left_summary['target_mae']['CO2air']:.3f} | "
        f"{right}: objective={right_summary['objective_mean']:.4f}, "
        f"CO2 MAE={right_summary['target_mae']['CO2air']:.3f}"
    )
    fig.suptitle(
        "Closed-Loop MPC Comparison: Offline Forecasting Leader vs Control Leader\n" + subtitle,
        fontsize=16,
        y=0.99,
    )

    if output is None:
        output = (
            project_root
            / "results"
            / "control"
            / "figures"
            / f"comparison_{left}_vs_{right}_{controller}.png"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a pairwise control comparison dashboard.")
    parser.add_argument("--left", default=DEFAULT_LEFT, help="Left predictor name.")
    parser.add_argument("--right", default=DEFAULT_RIGHT, help="Right predictor name.")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER, help="Controller name.")
    parser.add_argument("--output", default=None, help="Optional output figure path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output is not None else None
    saved = build_comparison(args.left, args.right, args.controller, output)
    print(f"Saved comparison figure: {saved}")


if __name__ == "__main__":
    main()
