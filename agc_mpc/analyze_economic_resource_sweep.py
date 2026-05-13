# -*- coding: utf-8 -*-
"""Summarize economic/resource-aware MPC weight sweeps."""

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


def _resolve(path: Path, project_root: Path, original_cwd: Path) -> Path:
    if path.exists():
        return path
    candidate = original_cwd / path
    if candidate.exists():
        return candidate
    return project_root / path


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else float("nan")


def build_rows(suite_paths: list[Path]) -> list[dict]:
    rows = []
    for path in suite_paths:
        payload = _load(path)
        weight = float(payload.get("economic_resource_weight", 0.0))
        profile = str(payload.get("economic_profile_name", "tracking"))
        by_predictor: dict[str, list[dict]] = {}
        for record in payload.get("records", []):
            by_predictor.setdefault(record["predictor"], []).append(record)
        for predictor, records in sorted(by_predictor.items()):
            rows.append(
                {
                    "profile": profile,
                    "resource_weight": weight,
                    "predictor": predictor,
                    "starts": ",".join(str(record["start_idx"]) for record in records),
                    "objective_mean": _mean([float(record["objective_mean"]) for record in records]),
                    "objective_std": _std([float(record["objective_mean"]) for record in records]),
                    "resource_proxy_mean": _mean([float(record.get("resource_proxy_mean", 0.0)) for record in records]),
                    "resource_proxy_std": _std([float(record.get("resource_proxy_mean", 0.0)) for record in records]),
                    "tair_mae_mean": _mean([float(record["target_mae"]["Tair"]) for record in records]),
                    "rhair_mae_mean": _mean([float(record["target_mae"]["Rhair"]) for record in records]),
                    "co2_mae_mean": _mean([float(record["target_mae"]["CO2air"]) for record in records]),
                    "control_delta_mean": _mean([float(record["control_delta_mae"]) for record in records]),
                    "action_tv_mean": _mean([float(record["action_tv"]) for record in records]),
                }
            )

    baseline = {
        row["predictor"]: row for row in rows if abs(float(row["resource_weight"])) <= 1e-12
    }
    for row in rows:
        base = baseline.get(row["predictor"])
        if not base:
            row["resource_change_pct"] = float("nan")
            row["co2_change_pct"] = float("nan")
            row["rhair_change_pct"] = float("nan")
            row["tair_change_pct"] = float("nan")
            continue
        for metric, out_key in [
            ("resource_proxy_mean", "resource_change_pct"),
            ("co2_mae_mean", "co2_change_pct"),
            ("rhair_mae_mean", "rhair_change_pct"),
            ("tair_mae_mean", "tair_change_pct"),
        ]:
            denom = float(base[metric])
            row[out_key] = 0.0 if abs(denom) <= 1e-12 else (float(row[metric]) - denom) / denom * 100.0
    return sorted(rows, key=lambda item: (item["predictor"], item["resource_weight"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Economic Resource MPC Sweep",
        "",
        "Mean values are computed across the requested rollout starts. Objective values are not directly comparable across weights because the resource term changes the optimized objective; use tracking errors and resource proxy changes for the trade-off.",
        "",
        "| predictor | weight | resource proxy | resource change | Tair MAE | Rhair MAE | CO2 MAE | CO2 change | action TV |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["predictor"],
                    f"{float(row['resource_weight']):.2f}",
                    f"{float(row['resource_proxy_mean']):.3f}",
                    f"{float(row['resource_change_pct']):+.1f}%",
                    f"{float(row['tair_mae_mean']):.3f}",
                    f"{float(row['rhair_mae_mean']):.3f}",
                    f"{float(row['co2_mae_mean']):.3f}",
                    f"{float(row['co2_change_pct']):+.1f}%",
                    f"{float(row['action_tv_mean']):.3f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictors = sorted({row["predictor"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, predictor in enumerate(predictors):
        pred_rows = sorted(
            [row for row in rows if row["predictor"] == predictor],
            key=lambda item: float(item["resource_weight"]),
        )
        weights = [float(row["resource_weight"]) for row in pred_rows]
        resources = [float(row["resource_proxy_mean"]) for row in pred_rows]
        co2 = [float(row["co2_mae_mean"]) for row in pred_rows]
        label = predictor.replace("_", " ")
        color = colors[idx % len(colors)]
        axes[0].plot(weights, resources, marker="o", linewidth=2.0, label=label, color=color)
        axes[1].plot(resources, co2, marker="o", linewidth=2.0, label=label, color=color)

    axes[0].set_title("Resource proxy vs weight")
    axes[0].set_xlabel("resource weight")
    axes[0].set_ylabel("resource proxy, lower is better")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Tracking-resource trade-off")
    axes[1].set_xlabel("resource proxy, lower is better")
    axes[1].set_ylabel("CO2air MAE, lower is better")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-json", nargs="+", type=Path, required=True)
    parser.add_argument("--prefix", default="economic_resource_sweep_top3_reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    os.chdir(project_root)
    suite_paths = [_resolve(path, project_root, original_cwd) for path in args.suite_json]

    cfg = AGCConfig()
    ensure_results_layout(cfg)
    rows = build_rows(suite_paths)
    csv_path = Path(cfg.control_summaries_dir) / f"{args.prefix}.csv"
    md_path = Path(cfg.control_summaries_dir) / f"{args.prefix}.md"
    fig_path = Path(cfg.control_figures_dir) / f"{args.prefix}.png"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    plot(fig_path, rows)
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()

