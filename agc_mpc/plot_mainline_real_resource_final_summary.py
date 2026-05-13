# -*- coding: utf-8 -*-
"""Create a presentation-ready summary figure for real-resource MPC validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import AGCConfig
from results_utils import ensure_results_layout


LABELS = {
    "current_hybrid_transformer": "Current hybrid Transformer",
    "itransformer_co2_residual": "CO2-aware iTransformer residual",
}


def plot(input_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(input_csv)
    df = df.sort_values(["predictor", "resource_weight"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    colors = {
        "current_hybrid_transformer": "tab:blue",
        "itransformer_co2_residual": "tab:green",
    }

    for predictor, group in df.groupby("predictor"):
        label = LABELS.get(predictor, predictor.replace("_", " "))
        color = colors.get(predictor)
        axes[0].plot(
            group["resource_weight"],
            group["co2_mae_mean"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=label,
        )
        axes[1].plot(
            group["resource_weight"],
            group["estimated_total_resource_cost_eur_m2_mean"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=label,
        )
        axes[2].plot(
            group["estimated_total_resource_cost_eur_m2_mean"],
            group["co2_mae_mean"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
        for _, row in group.iterrows():
            axes[2].annotate(
                f"w={row['resource_weight']:.2f}",
                (row["estimated_total_resource_cost_eur_m2_mean"], row["co2_mae_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    axes[0].set_title("CO2 tracking under resource weight")
    axes[0].set_xlabel("resource weight")
    axes[0].set_ylabel("CO2air MAE, lower is better")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Estimated resource cost")
    axes[1].set_xlabel("resource weight")
    axes[1].set_ylabel("EUR/m2 over 96-step rollout")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Tracking-resource trade-off")
    axes[2].set_xlabel("estimated resource cost, lower is better")
    axes[2].set_ylabel("CO2air MAE, lower is better")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        "Real-AGC-resource-calibrated MPC validation: selected model comparison",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/control/summaries/mainline_real_resource_sensitivity.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/control/figures/mainline_real_resource_final_summary.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = AGCConfig()
    ensure_results_layout(cfg)
    input_csv = args.input_csv if args.input_csv.is_absolute() else project_root / args.input_csv
    output = args.output if args.output.is_absolute() else project_root / args.output
    plot(input_csv, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
