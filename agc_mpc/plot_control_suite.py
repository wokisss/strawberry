# -*- coding: utf-8 -*-
"""Plot aggregate control benchmark summaries for a predictor suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREDICTOR_LABELS = {
    "current_hybrid_transformer": "Current Hybrid",
    "transformer_hybrid_residual": "Hybrid Residual",
    "itransformer_residual": "iTransformer Residual",
    "itransformer_co2_residual": "iTransformer + CO2",
    "itransformer_co2_late_residual": "iTransformer + CO2 v2",
    "itransformer_co2_frozen_expert": "iTransformer + Frozen CO2",
    "itransformer_co2_late_frozen_expert": "iTransformer + Late Frozen CO2",
    "itransformer_co2_teacher_distill": "iTransformer + CO2 Distill",
    "itransformer_co2_recoupled_expert": "iTransformer + Recoupled CO2",
    "itransformer_co2_protected_expert": "iTransformer + Protected CO2",
    "itransformer_co2_protected_terminal": "iTransformer + Protected CO2 v2",
    "itransformer_co2_horizon_mixture": "iTransformer + Horizon Mix",
    "itransformer_co2_frozen_backbone_horizon_mixture": "iTransformer + Safe Horizon Mix",
    "itransformer_co2_wavelet_residual": "iTransformer + CO2 Wavelet",
    "itransformer_co2_wavelet_blend": "iTransformer + CO2 Blend",
    "patchtst_residual": "PatchTST Residual",
    "dlinear_baseline": "DLinear",
    "transformer_hybrid_baseline": "Transformer-hybrid",
    "transformer_baseline": "Transformer",
}

CONTROLLER_LABELS = {
    "recorded": "Recorded",
    "gradient_mpc": "GradientMPC",
    "cem_mpc": "CEMMPC",
}

CONTROLLER_COLORS = {
    "recorded": "#7f8c8d",
    "gradient_mpc": "#1f9d73",
    "cem_mpc": "#d95f02",
}


def _setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _record_lookup(records: list[dict]) -> dict[tuple[str, str], dict]:
    return {(record["predictor"], record["controller"]): record for record in records}


def _plot_grouped_bars(ax, predictors: list[str], controllers: list[str], values_by_controller: dict[str, list[float]], title: str, ylabel: str) -> None:
    x = np.arange(len(predictors))
    width = 0.24
    offsets = np.linspace(-width, width, num=len(controllers))
    for offset, controller in zip(offsets, controllers):
        ax.bar(
            x + offset,
            values_by_controller[controller],
            width=width,
            color=CONTROLLER_COLORS[controller],
            label=CONTROLLER_LABELS[controller],
        )
    ax.set_xticks(x, [PREDICTOR_LABELS.get(name, name) for name in predictors], rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def build_suite_plot(suite: dict, out_path: Path) -> Path:
    predictors = suite["predictors"]
    controllers = suite["controllers"]
    target_cols = suite["target_cols"]
    lookup = _record_lookup(suite["records"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, target in zip(axes[0], target_cols):
        values = {
            controller: [lookup[(predictor, controller)]["target_mae"][target] for predictor in predictors]
            for controller in controllers
        }
        _plot_grouped_bars(ax, predictors, controllers, values, f"{target} closed-loop MAE", "MAE")
    metric_specs = [
        ("objective_mean", "Objective Mean"),
        ("control_delta_mae", "|u-u_log| mean"),
        ("action_tv", "Action TV"),
    ]
    for ax, (metric_name, title) in zip(axes[1], metric_specs):
        values = {
            controller: [lookup[(predictor, controller)][metric_name] for predictor in predictors]
            for controller in controllers
        }
        _plot_grouped_bars(ax, predictors, controllers, values, title, metric_name)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(controllers), frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"AGC closed-loop control suite | {suite['compartment']} | {suite['steps']} steps | {suite['rollout_mode']}",
        fontsize=14,
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-json", required=True, help="Path to the combined suite summary JSON.")
    parser.add_argument("--out", default=None, help="Optional explicit output figure path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_matplotlib()
    suite_path = Path(args.suite_json)
    suite = _load_json(suite_path)
    if args.out is None:
        out_path = suite_path.with_name(suite_path.stem + "_metrics.png").with_suffix(".png")
        out_path = Path(str(out_path).replace("\\summaries\\", "\\figures\\"))
    else:
        out_path = Path(args.out)
    result = build_suite_plot(suite, out_path)
    print(f"Saved control suite plot: {result}")


if __name__ == "__main__":
    main()
