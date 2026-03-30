# -*- coding: utf-8 -*-
"""Plot old DiffMPC-style Transformer results across Strawberry and AGC regimes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_layout import comparison_figures_dir


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_ROOT / "results" / "forecasting" / "analysis"
FIGURES_DIR = comparison_figures_dir(PROJECT_ROOT / "results" / "forecasting" / "figures")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    old_summary = _load_json(ANALYSIS_DIR / "strawberry_vs_agc_dataset_switch_summary.json")
    single = _load_json(ANALYSIS_DIR / "diffmpc_style_transformer_single_reference_summary.json")
    joint_all = _load_json(ANALYSIS_DIR / "diffmpc_style_transformer_joint_all_reference_summary.json")
    leave_one_out = _load_json(ANALYSIS_DIR / "diffmpc_style_transformer_leave_one_out_reference_summary.json")

    targets_old = old_summary["old_project"]["variables"]
    old_mae = old_summary["old_project"]["final_mae"]
    old_r2 = old_summary["old_project"]["final_r2"]

    agc_targets = ["Tair", "Rhair", "CO2air"]
    agc_labels = {
        "Tair": "Temperature",
        "Rhair": "Humidity",
        "CO2air": "CO2",
    }

    series = [
        {
            "label": "Strawberry / old Transformer-hybrid",
            "mae": dict(zip(targets_old, old_mae)),
            "r2": dict(zip(targets_old, old_r2)),
            "color": "#9c6ade",
        },
        {
            "label": "AGC / diffmpc-style / single",
            "mae": {agc_labels[k]: v["final_mae"] for k, v in single["metrics_by_target"].items()},
            "r2": {agc_labels[k]: v["final_r2"] for k, v in single["metrics_by_target"].items()},
            "color": "#5b8ff9",
        },
        {
            "label": "AGC / diffmpc-style / joint_all",
            "mae": {agc_labels[k]: v["final_mae"] for k, v in joint_all["metrics_by_target"].items()},
            "r2": {agc_labels[k]: v["final_r2"] for k, v in joint_all["metrics_by_target"].items()},
            "color": "#5ad8a6",
        },
        {
            "label": "AGC / diffmpc-style / leave_one_out",
            "mae": {agc_labels[k]: v["final_mae"] for k, v in leave_one_out["metrics_by_target"].items()},
            "r2": {agc_labels[k]: v["final_r2"] for k, v in leave_one_out["metrics_by_target"].items()},
            "color": "#f6bd16",
        },
    ]

    target_order = ["Temperature", "Humidity", "CO2"]
    x = np.arange(len(target_order))
    width = 0.2

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), squeeze=False)

    for col_idx, target in enumerate(target_order):
        ax_mae = axes[0, col_idx]
        ax_r2 = axes[1, col_idx]

        for series_idx, item in enumerate(series):
            offset = (series_idx - 1.5) * width
            mae_value = item["mae"][target]
            r2_value = item["r2"][target]

            ax_mae.bar(
                x=[0 + offset],
                height=[mae_value],
                width=width,
                color=item["color"],
                label=item["label"] if col_idx == 0 else None,
            )
            ax_r2.bar(
                x=[0 + offset],
                height=[r2_value],
                width=width,
                color=item["color"],
            )
            ax_mae.text(0 + offset, mae_value, f"{mae_value:.2f}", ha="center", va="bottom", fontsize=8)
            ax_r2.text(0 + offset, r2_value, f"{r2_value:.3f}", ha="center", va="bottom", fontsize=8)

        ax_mae.set_title(f"{target} final MAE")
        ax_mae.set_xticks([0], [target])
        ax_mae.grid(axis="y", alpha=0.25)

        ax_r2.set_title(f"{target} final R2")
        ax_r2.set_xticks([0], [target])
        ax_r2.grid(axis="y", alpha=0.25)
        ax_r2.set_ylim(bottom=min(0.0, ax_r2.get_ylim()[0]))

    fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "DiffMPC-style Transformer across datasets and AGC training regimes\n"
        "Protocol focus: equivalent 2-hour forecasting task, old-transformer-style training budget",
        fontsize=14,
        y=1.08,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = FIGURES_DIR / "diffmpc_style_transformer_dataset_suitability.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
