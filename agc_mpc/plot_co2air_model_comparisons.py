# -*- coding: utf-8 -*-
"""Generate CO2-focused comparison figures across selected forecasting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_layout import comparison_figures_dir


ITRANSFORMER_SUITE = [
    "itransformer_residual",
    "itransformer_co2_residual",
    "itransformer_co2_late_residual",
    "itransformer_co2_frozen_expert",
    "itransformer_co2_late_frozen_expert",
    "itransformer_co2_teacher_distill",
    "itransformer_co2_recoupled_expert",
    "itransformer_co2_protected_expert",
    "itransformer_co2_protected_terminal",
    "itransformer_co2_horizon_mixture",
    "itransformer_co2_frozen_backbone_horizon_mixture",
]

CO2_SPECIALIST_SUITE = [
    "co2_env_lstm",
    "co2_vmd_lstm_fusion",
    "co2_wavelet_gru_attn",
]

MODEL_LABELS = {
    "itransformer_residual": "iTransformer Residual",
    "itransformer_co2_residual": "iTransformer + CO2 v1",
    "itransformer_co2_late_residual": "iTransformer + CO2 v2",
    "itransformer_co2_frozen_expert": "iTransformer + Frozen CO2",
    "itransformer_co2_late_frozen_expert": "iTransformer + Late Frozen CO2",
    "itransformer_co2_teacher_distill": "iTransformer + CO2 Distill",
    "itransformer_co2_recoupled_expert": "iTransformer + Recoupled CO2",
    "itransformer_co2_protected_expert": "iTransformer + Protected CO2",
    "itransformer_co2_protected_terminal": "iTransformer + Protected CO2 v2",
    "itransformer_co2_horizon_mixture": "iTransformer + Horizon Mix",
    "itransformer_co2_frozen_backbone_horizon_mixture": "iTransformer + Safe Horizon Mix",
    "co2_env_lstm": "CO2 LSTM",
    "co2_vmd_lstm_fusion": "CO2 VMD-LSTM",
    "co2_wavelet_gru_attn": "CO2 Wavelet-GRU",
}

MODEL_COLORS = {
    "itransformer_residual": "#1f77b4",
    "itransformer_co2_residual": "#ff7f0e",
    "itransformer_co2_late_residual": "#2ca02c",
    "itransformer_co2_frozen_expert": "#9467bd",
    "itransformer_co2_late_frozen_expert": "#8c564b",
    "itransformer_co2_teacher_distill": "#e377c2",
    "itransformer_co2_recoupled_expert": "#bcbd22",
    "itransformer_co2_protected_expert": "#17a589",
    "itransformer_co2_protected_terminal": "#f39c12",
    "itransformer_co2_horizon_mixture": "#34495e",
    "itransformer_co2_frozen_backbone_horizon_mixture": "#c0392b",
    "co2_env_lstm": "#7f7f7f",
    "co2_vmd_lstm_fusion": "#d62728",
    "co2_wavelet_gru_attn": "#17becf",
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


def _summary_path(analysis_dir: Path, run_name: str) -> Path:
    return analysis_dir / f"{run_name}_summary.json"


def _load_suite(analysis_dir: Path, models: list[str], regime: str, compartment: str) -> list[dict]:
    suite = []
    for model_name in models:
        run_name = f"{model_name}_{regime}_{compartment.lower()}"
        suite.append(_load_json(_summary_path(analysis_dir, run_name)))
    return suite


def _annotate_bars(ax, bars, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def _plot_itransformer_metrics(summaries: list[dict], out_path: Path) -> Path:
    targets = ["Tair", "Rhair", "CO2air"]
    metric_specs = [
        ("full_mae", "Full-Horizon MAE"),
        ("final_mae", "Final-Step MAE"),
    ]
    x = np.arange(len(targets))
    width = 0.075

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes = np.atleast_1d(axes)
    for ax, (metric_name, title) in zip(axes, metric_specs):
        for idx, summary in enumerate(summaries):
            model_name = summary["protocol"]["variant"]
            offset = (idx - (len(summaries) - 1) / 2) * width
            values = [summary["metrics_by_target"][target][metric_name] for target in targets]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            _annotate_bars(ax, bars)
        ax.set_xticks(x, targets)
        ax.set_title(title)
        ax.set_ylabel("MAE")
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.suptitle("iTransformer CO2 Branch Comparison", fontsize=14, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_co2_specialist_metrics(summaries: list[dict], out_path: Path) -> Path:
    metric_specs = [
        ("full_mae", "Full-Horizon MAE"),
        ("final_mae", "Final-Step MAE"),
    ]
    x = np.arange(len(summaries))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes = np.atleast_1d(axes)
    for ax, (metric_name, title) in zip(axes, metric_specs):
        values = [summary["metrics_by_target"]["CO2air"][metric_name] for summary in summaries]
        labels = [MODEL_LABELS[summary["protocol"]["variant"]] for summary in summaries]
        colors = [MODEL_COLORS[summary["protocol"]["variant"]] for summary in summaries]
        bars = ax.bar(x, values, color=colors, width=0.6)
        _annotate_bars(ax, bars)
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.set_title(title)
        ax.set_ylabel("MAE")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Standalone CO2 Specialist Comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default="results/forecasting/analysis")
    parser.add_argument("--figures-dir", default="results/forecasting/figures")
    parser.add_argument("--regime", default="joint_all")
    parser.add_argument("--target-compartment", default="Reference")
    return parser.parse_args()


def main() -> None:
    _setup_matplotlib()
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    analysis_dir = project_root / args.analysis_dir
    output_dir = comparison_figures_dir(project_root / args.figures_dir)

    itransformer_summaries = _load_suite(analysis_dir, ITRANSFORMER_SUITE, args.regime, args.target_compartment)
    specialist_summaries = _load_suite(analysis_dir, CO2_SPECIALIST_SUITE, args.regime, args.target_compartment)

    itransformer_metrics = _plot_itransformer_metrics(
        itransformer_summaries,
        output_dir / "itransformer_co2_branch_comparison_metrics.png",
    )
    specialist_metrics = _plot_co2_specialist_metrics(
        specialist_summaries,
        output_dir / "co2_specialists_comparison_metrics.png",
    )

    print(f"Saved: {itransformer_metrics}")
    print(f"Saved: {specialist_metrics}")


if __name__ == "__main__":
    main()
