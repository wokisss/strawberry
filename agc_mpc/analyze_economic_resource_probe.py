# -*- coding: utf-8 -*-
"""Compare tracking-only and economic/resource-aware MPC probe suites."""

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


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_by_predictor(payload: dict) -> dict[str, dict]:
    return {record["predictor"]: record for record in payload.get("records", [])}


def _pct_change(new: float, old: float) -> float:
    if abs(old) <= 1e-12:
        return 0.0
    return (new - old) / old * 100.0


def build_rows(tracking_suite: Path, economic_suite: Path) -> list[dict]:
    tracking = _records_by_predictor(_load(tracking_suite))
    economic = _records_by_predictor(_load(economic_suite))
    rows = []
    for predictor in sorted(set(tracking) & set(economic)):
        base = tracking[predictor]
        econ = economic[predictor]
        base_targets = base["target_mae"]
        econ_targets = econ["target_mae"]
        rows.append(
            {
                "predictor": predictor,
                "tracking_objective": float(base["objective_mean"]),
                "economic_objective": float(econ["objective_mean"]),
                "objective_change_pct": _pct_change(float(econ["objective_mean"]), float(base["objective_mean"])),
                "tracking_resource_proxy": float(base.get("resource_proxy_mean", 0.0)),
                "economic_resource_proxy": float(econ.get("resource_proxy_mean", 0.0)),
                "resource_change_pct": _pct_change(
                    float(econ.get("resource_proxy_mean", 0.0)),
                    float(base.get("resource_proxy_mean", 0.0)),
                ),
                "tracking_tair_mae": float(base_targets["Tair"]),
                "economic_tair_mae": float(econ_targets["Tair"]),
                "tracking_rhair_mae": float(base_targets["Rhair"]),
                "economic_rhair_mae": float(econ_targets["Rhair"]),
                "tracking_co2_mae": float(base_targets["CO2air"]),
                "economic_co2_mae": float(econ_targets["CO2air"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Economic/Resource-Aware MPC Probe Comparison",
        "",
        "Note: the reported objective values are not directly comparable because the economic run includes the resource term. Use target MAE and resource proxy changes to judge the trade-off.",
        "",
        "| predictor | tracking objective | economic objective | tracking resource | economic resource | resource change | tracking Tair MAE | economic Tair MAE | tracking Rhair MAE | economic Rhair MAE | tracking CO2 MAE | economic CO2 MAE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["predictor"],
                    f"{row['tracking_objective']:.4f}",
                    f"{row['economic_objective']:.4f}",
                    f"{row['tracking_resource_proxy']:.3f}",
                    f"{row['economic_resource_proxy']:.3f}",
                    f"{row['resource_change_pct']:+.1f}%",
                    f"{row['tracking_tair_mae']:.3f}",
                    f"{row['economic_tair_mae']:.3f}",
                    f"{row['tracking_rhair_mae']:.3f}",
                    f"{row['economic_rhair_mae']:.3f}",
                    f"{row['tracking_co2_mae']:.3f}",
                    f"{row['economic_co2_mae']:.3f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["predictor"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, key_a, key_b, title, ylabel in [
        (axes[0], "tracking_objective", "economic_objective", "Objective", "lower is better"),
        (axes[1], "tracking_resource_proxy", "economic_resource_proxy", "Resource proxy", "lower is better"),
        (axes[2], "tracking_co2_mae", "economic_co2_mae", "CO2air MAE", "lower is better"),
    ]:
        ax.bar(x - width / 2, [row[key_a] for row in rows], width, label="tracking-only", color="#6c8ebf")
        ax.bar(x + width / 2, [row[key_b] for row in rows], width, label="economic/resource", color="#c47f2e")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-suite", type=Path, required=True)
    parser.add_argument("--economic-suite", type=Path, required=True)
    parser.add_argument("--prefix", default="economic_resource_probe_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    os.chdir(project_root)
    if not args.tracking_suite.exists():
        candidate = original_cwd / args.tracking_suite
        args.tracking_suite = candidate if candidate.exists() else project_root / args.tracking_suite
    if not args.economic_suite.exists():
        candidate = original_cwd / args.economic_suite
        args.economic_suite = candidate if candidate.exists() else project_root / args.economic_suite

    cfg = AGCConfig()
    ensure_results_layout(cfg)
    rows = build_rows(args.tracking_suite, args.economic_suite)
    if not rows:
        raise ValueError("No shared predictors found between the two suites.")

    summary_dir = Path(cfg.control_summaries_dir)
    figure_dir = Path(cfg.control_figures_dir)
    csv_path = summary_dir / f"{args.prefix}.csv"
    md_path = summary_dir / f"{args.prefix}.md"
    fig_path = figure_dir / f"{args.prefix}.png"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    plot(fig_path, rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
