# -*- coding: utf-8 -*-
"""Plot multi-start closed-loop objective and CO2 rankings."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import AGCConfig
from results_utils import ensure_results_layout


LABELS = {
    "current_hybrid_transformer": "Hybrid",
    "transformer_hybrid_residual": "Hybrid residual",
    "segrnn_forecaster": "SegRNN",
    "frequency_forecaster": "Frequency",
    "itransformer_co2_residual": "iT CO2 residual",
    "itransformer_co2_protected_expert": "Protected expert",
    "itransformer_co2_late_residual": "Late residual",
    "itransformer_co2_late_frozen_expert": "Late frozen",
    "itransformer_co2_control_aware_fusion": "Control-aware",
    "itransformer_co2_horizon_mixture": "Horizon mix",
    "itransformer_residual": "iT residual",
    "patchtst_residual": "PatchTST",
    "transformer_forecaster": "Transformer",
    "nlinear_forecaster": "NLinear",
    "dlinear_forecaster": "DLinear",
    "itransformer_co2_wavelet_residual": "Wavelet residual",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank(values: list[float]) -> list[int]:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    ranks = np.empty(len(values), dtype=np.int64)
    for idx, ordered_idx in enumerate(order, start=1):
        ranks[ordered_idx] = idx
    return ranks.tolist()


def _records_by_start(payload: dict) -> dict[int, list[dict]]:
    by_start: dict[int, list[dict]] = {}
    for record in payload["records"]:
        if record.get("controller") != "gradient_mpc":
            continue
        by_start.setdefault(int(record["start_idx"]), []).append(record)
    return {start: sorted(records, key=lambda item: item["predictor"]) for start, records in by_start.items()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "start_idx",
        "predictor",
        "objective_mean",
        "objective_rank",
        "co2_mae",
        "co2_rank",
        "tair_mae",
        "rhair_mae",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Multi-Start Closed-Loop Ranking",
        "",
        "| start_idx | predictor | objective | objective_rank | CO2 MAE | CO2_rank | Tair MAE | Rhair MAE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["start_idx"]),
                    row["predictor"],
                    f"{row['objective_mean']:.4f}",
                    str(row["objective_rank"]),
                    f"{row['co2_mae']:.3f}",
                    str(row["co2_rank"]),
                    f"{row['tair_mae']:.3f}",
                    f"{row['rhair_mae']:.3f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path: Path, payload: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    starts = sorted({row["start_idx"] for row in rows})
    predictors = payload["predictors"]
    labels = [LABELS.get(predictor, predictor) for predictor in predictors]
    x = np.arange(len(predictors))

    fig, axes = plt.subplots(len(starts), 2, figsize=(18, 4.2 * len(starts)), sharex=True)
    if len(starts) == 1:
        axes = np.asarray([axes])
    highlight = {
        "itransformer_co2_residual": "#d95f02",
        "current_hybrid_transformer": "#1f77b4",
        "transformer_hybrid_residual": "#2ca02c",
    }
    default_color = "#9aa6b2"

    for row_idx, start in enumerate(starts):
        start_rows = [row for row in rows if row["start_idx"] == start]
        lookup = {row["predictor"]: row for row in start_rows}
        objective_values = [lookup[predictor]["objective_mean"] for predictor in predictors]
        co2_values = [lookup[predictor]["co2_mae"] for predictor in predictors]
        colors = [highlight.get(predictor, default_color) for predictor in predictors]

        axes[row_idx, 0].bar(x, objective_values, color=colors)
        axes[row_idx, 0].set_title(f"start {start}: objective")
        axes[row_idx, 0].set_ylabel("objective, lower is better")
        axes[row_idx, 0].grid(axis="y", alpha=0.25)

        axes[row_idx, 1].bar(x, co2_values, color=colors)
        axes[row_idx, 1].set_title(f"start {start}: CO2air MAE")
        axes[row_idx, 1].set_ylabel("CO2air MAE, lower is better")
        axes[row_idx, 1].grid(axis="y", alpha=0.25)

        for col, values in enumerate([objective_values, co2_values]):
            best_idx = int(np.argmin(values))
            axes[row_idx, col].scatter([best_idx], [values[best_idx]], color="black", s=45, zorder=4)

    for ax in axes[-1, :]:
        ax.set_xticks(x, labels, rotation=35, ha="right")
    fig.suptitle("Multi-start closed-loop rankings: objective vs CO2 tracking", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_outputs(suite_path: Path, prefix: str) -> list[Path]:
    cfg = AGCConfig()
    ensure_results_layout(cfg)
    payload = _load(suite_path)
    by_start = _records_by_start(payload)
    rows = []
    for start, records in by_start.items():
        objectives = [float(record["objective_mean"]) for record in records]
        co2_values = [float(record["target_mae"]["CO2air"]) for record in records]
        objective_ranks = _rank(objectives)
        co2_ranks = _rank(co2_values)
        for idx, record in enumerate(records):
            rows.append(
                {
                    "start_idx": start,
                    "predictor": record["predictor"],
                    "objective_mean": float(record["objective_mean"]),
                    "objective_rank": objective_ranks[idx],
                    "co2_mae": float(record["target_mae"]["CO2air"]),
                    "co2_rank": co2_ranks[idx],
                    "tair_mae": float(record["target_mae"]["Tair"]),
                    "rhair_mae": float(record["target_mae"]["Rhair"]),
                }
            )
    order = {predictor: idx for idx, predictor in enumerate(payload["predictors"])}
    rows = sorted(rows, key=lambda item: (item["start_idx"], order.get(item["predictor"], 10**9)))
    analysis_dir = Path(cfg.forecast_analysis_dir)
    figure_dir = Path(cfg.forecast_figures_dir) / "comparisons"
    csv_path = analysis_dir / f"{prefix}.csv"
    md_path = analysis_dir / f"{prefix}.md"
    figure_path = figure_dir / f"{prefix}.png"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _plot(figure_path, payload, rows)
    return [csv_path, md_path, figure_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-json", type=Path, required=True)
    parser.add_argument("--prefix", default="fctv_multistart_model_rankings_reference")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    os.chdir(project_root)
    args = parse_args()
    if not args.suite_json.exists():
        candidate = original_cwd / args.suite_json
        if candidate.exists():
            args.suite_json = candidate
    for output in build_outputs(args.suite_json, args.prefix):
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
