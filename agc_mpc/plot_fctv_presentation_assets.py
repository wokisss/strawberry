# -*- coding: utf-8 -*-
"""Build paper-facing FCTV presentation assets from the transfer analysis JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import AGCConfig
from results_utils import ensure_results_layout


ROLE_ROWS = [
    ("Rhair", "rhair_first_step_mae", "mpc_rhair_mae"),
    ("CO2air", "co2_first_step_mae", "mpc_co2_mae"),
    ("CO2air", "co2_constraint_near_mae_proxy", "mpc_co2_mae"),
    ("Tair", "tair_first_step_mae", "mpc_tair_mae"),
    ("Objective", "multiobjective_transfer_selection_score", "mpc_objective"),
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _robustness_lookup(payload: dict) -> dict[tuple[str, str], dict]:
    return {(row["control_target"], row["metric"]): row for row in payload["robustness_rows"]}


def _role(payload: dict, target: str, metric: str) -> str:
    return payload.get("metric_roles", {}).get(target, {}).get(metric, "missing")


def _plot_logic_chain(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    boxes = [
        ("Model / module", "Backbone, residual path,\nPHF expert, frequency prior"),
        ("Forecast behavior", "Timing error, bias,\nconstraint-near error,\ncontrol sensitivity"),
        ("FCTV metric", "First-step MAE,\ncontrol-horizon MAE,\nbias, near-boundary MAE,\ngradient diagnostics"),
        ("Closed-loop target", "Tair MAE, Rhair MAE,\nCO2air MAE, objective"),
    ]
    x_positions = np.linspace(0.08, 0.86, len(boxes))
    for idx, ((title, body), x) in enumerate(zip(boxes, x_positions)):
        ax.text(
            x,
            0.58,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#f7f3e8" if idx % 2 == 0 else "#e8f1f7",
                "edgecolor": "#263238",
                "linewidth": 1.2,
            },
        )
        if idx < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.095, 0.58),
                xytext=(x + 0.095, 0.58),
                arrowprops={"arrowstyle": "->", "linewidth": 1.8, "color": "#263238"},
            )
    ax.text(
        0.5,
        0.13,
        "Derive candidate metrics from MPC mechanics first, then validate which metrics predict closed-loop outcomes across models.",
        ha="center",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )
    fig.suptitle("Forecast-to-Control Transfer Validation logic chain", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_role_table(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = _robustness_lookup(payload)
    table = []
    for label, metric, control_target in ROLE_ROWS:
        row = lookup[(control_target, metric)]
        table.append(
            [
                label,
                metric,
                _role(payload, control_target, metric),
                f"{row['full_spearman']:.3f}",
                f"{row['full_pairwise_consistency']:.3f}",
            ]
        )
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    tab = ax.table(
        cellText=table,
        colLabels=["Scope", "Metric", "Role", "Spearman", "Pairwise"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1.0, 1.8)
    for (row_idx, col_idx), cell in tab.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold")
        elif col_idx == 2:
            role = table[row_idx - 1][2]
            if "primary" in role:
                cell.set_facecolor("#d7ebff")
            elif "secondary" in role:
                cell.set_facecolor("#dff1df")
            elif "weak" in role:
                cell.set_facecolor("#fff0c2")
            else:
                cell.set_facecolor("#f5d6d6")
    fig.suptitle(f"Current FCTV metric roles in the {payload['model_count']}-model pool", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_counterexample(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = payload["ranked_records"]
    x = np.asarray([record["co2_final_step_mae"] for record in records], dtype=np.float64)
    y = np.asarray([record["mpc_co2_mae"] for record in records], dtype=np.float64)
    first = np.asarray([record["co2_first_step_mae"] for record in records], dtype=np.float64)
    labels = [record["predictor"] for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    axes[0].scatter(x, y, s=70, color="#d69c2f", edgecolor="#263238")
    axes[0].set_xlabel("CO2 final-step MAE")
    axes[0].set_ylabel("Closed-loop CO2 MAE")
    axes[0].set_title("Terminal forecast rank is not enough")
    axes[0].grid(alpha=0.25)
    axes[1].scatter(first, y, s=70, color="#2c6fbb", edgecolor="#263238")
    axes[1].set_xlabel("CO2 first-step MAE")
    axes[1].set_ylabel("Closed-loop CO2 MAE")
    axes[1].set_title("First-step metric is more control-relevant")
    axes[1].grid(alpha=0.25)
    highlight = {
        "segrnn_forecaster",
        "frequency_forecaster",
        "itransformer_co2_horizon_mixture",
        "itransformer_co2_late_frozen_expert",
    }
    for idx, label in enumerate(labels):
        if label not in highlight:
            continue
        short = label.replace("itransformer_co2_", "").replace("_forecaster", "").replace("_", "\n")
        axes[0].annotate(short, (x[idx], y[idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axes[1].annotate(short, (first[idx], y[idx]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    fig.suptitle("Counterexample: better final-step CO2 forecasting does not guarantee better control", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_assets(transfer_json: Path, prefix: str) -> list[Path]:
    cfg = AGCConfig()
    ensure_results_layout(cfg)
    payload = _load(transfer_json)
    figure_dir = Path(cfg.forecast_figures_dir) / "comparisons"
    outputs = [
        figure_dir / f"{prefix}_logic_chain.png",
        figure_dir / f"{prefix}_metric_roles.png",
        figure_dir / f"{prefix}_co2_counterexample.png",
    ]
    _plot_logic_chain(outputs[0])
    _plot_role_table(outputs[1], payload)
    _plot_counterexample(outputs[2], payload)
    return outputs


def parse_args() -> argparse.Namespace:
    cfg = AGCConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-json",
        type=Path,
        default=Path(cfg.forecast_analysis_dir) / "forecast_to_control_transfer_reference.json",
    )
    parser.add_argument("--prefix", default="fctv_presentation_reference")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    args = parse_args()
    outputs = build_assets(args.transfer_json, args.prefix)
    for output in outputs:
        print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
