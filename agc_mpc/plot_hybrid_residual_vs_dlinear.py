# -*- coding: utf-8 -*-
"""Plot a direct comparison between DLinear and hybrid residual forecasters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_ROOT / "results" / "forecasting" / "analysis"
FIGURES_DIR = PROJECT_ROOT / "results" / "forecasting" / "figures"

DLINEAR_SUMMARY = ANALYSIS_DIR / "dlinear_forecaster_joint_all_reference_summary.json"
HYBRID_SUMMARY = ANALYSIS_DIR / "hybrid_residual_forecaster_joint_all_reference_summary.json"

TARGETS = ["Tair", "Rhair", "CO2air"]
DISPLAY_NAMES = {
    "Tair": "温度",
    "Rhair": "湿度",
    "CO2air": "CO2",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metric(summary: dict, metric_name: str) -> list[float]:
    return [summary["metrics_by_target"][target][metric_name] for target in TARGETS]


def _plot_metric_bars(ax, dlinear_summary: dict, hybrid_summary: dict, metric_name: str, title: str) -> None:
    x = np.arange(len(TARGETS))
    width = 0.36
    dlinear_values = _extract_metric(dlinear_summary, metric_name)
    hybrid_values = _extract_metric(hybrid_summary, metric_name)

    ax.bar(x - width / 2, dlinear_values, width=width, label="DLinear", color="#4e79a7")
    ax.bar(x + width / 2, hybrid_values, width=width, label="Hybrid Residual", color="#e15759")
    ax.set_xticks(x, [DISPLAY_NAMES[target] for target in TARGETS])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()


def _plot_representative_windows(ax_row, dlinear_summary: dict, hybrid_summary: dict, target: str, target_idx: int) -> None:
    true_window = np.asarray(dlinear_summary["representative_window"]["true"], dtype=np.float32)[:, target_idx]
    dlinear_pred = np.asarray(dlinear_summary["representative_window"]["pred"], dtype=np.float32)[:, target_idx]
    hybrid_pred = np.asarray(hybrid_summary["representative_window"]["pred"], dtype=np.float32)[:, target_idx]
    steps = np.arange(1, len(true_window) + 1)

    ax_row.plot(steps, true_window, color="black", linewidth=2.0, label="真实轨迹")
    ax_row.plot(steps, dlinear_pred, color="#4e79a7", linestyle="--", linewidth=2.0, label="DLinear")
    ax_row.plot(steps, hybrid_pred, color="#e15759", linestyle="-.", linewidth=2.0, label="Hybrid Residual")
    ax_row.set_title(f"{DISPLAY_NAMES[target]} 代表性预测窗")
    ax_row.set_xlabel("预测步")
    ax_row.set_ylabel(DISPLAY_NAMES[target])
    ax_row.grid(True, alpha=0.25)


def main() -> None:
    dlinear_summary = _load_json(DLINEAR_SUMMARY)
    hybrid_summary = _load_json(HYBRID_SUMMARY)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "hybrid_residual_vs_dlinear_joint_all_reference.png"

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.2, 1.2])

    _plot_metric_bars(fig.add_subplot(gs[0, 0]), dlinear_summary, hybrid_summary, "full_mae", "Full MAE 对比")
    _plot_metric_bars(fig.add_subplot(gs[0, 1]), dlinear_summary, hybrid_summary, "final_mae", "Final MAE 对比")
    _plot_metric_bars(fig.add_subplot(gs[1, 0]), dlinear_summary, hybrid_summary, "full_r2", "Full R2 对比")
    _plot_metric_bars(fig.add_subplot(gs[1, 1]), dlinear_summary, hybrid_summary, "final_r2", "Final R2 对比")

    for idx, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[2 + idx // 2, idx % 2])
        _plot_representative_windows(ax, dlinear_summary, hybrid_summary, target, idx)
        if idx == 0:
            ax.legend()

    fig.suptitle("Hybrid Residual vs DLinear | joint_all + Reference", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
