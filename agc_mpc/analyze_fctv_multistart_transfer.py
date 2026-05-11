# -*- coding: utf-8 -*-
"""Analyze FCTV transfer robustness across repeated closed-loop start indices."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_forecast_to_control_transfer import (
    _analyze,
    _json_ready,
    _plot,
    _plot_robustness,
    _plot_summary_dashboard,
    _write_csv,
    _write_markdown,
    _write_robustness_csv,
)
from config import AGCConfig
from results_utils import ensure_results_layout


KEY_METRICS = [
    ("mpc_tair_mae", "tair_first_step_mae"),
    ("mpc_rhair_mae", "rhair_first_step_mae"),
    ("mpc_co2_mae", "co2_first_step_mae"),
    ("mpc_co2_mae", "co2_constraint_near_mae_proxy"),
    ("mpc_objective", "multiobjective_transfer_selection_score"),
]


def _load_base_records(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("ranked_summary", [])
    if not records:
        raise ValueError(f"No ranked_summary records found in {path}")
    return {record["predictor"]: record for record in records}


def _load_control_records(path: Path) -> dict[int, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_start: dict[int, dict[str, dict]] = {}
    for record in payload.get("records", []):
        if record.get("controller") != "gradient_mpc":
            continue
        start_idx = int(record.get("start_idx", payload.get("start_idx", 0)))
        by_start.setdefault(start_idx, {})[record["predictor"]] = record
    if not by_start:
        raise ValueError(f"No gradient_mpc records found in {path}")
    return by_start


def _records_for_start(base_by_predictor: dict[str, dict], control_by_predictor: dict[str, dict]) -> list[dict]:
    records = []
    for predictor, base in base_by_predictor.items():
        control = control_by_predictor.get(predictor)
        if control is None:
            continue
        record = copy.deepcopy(base)
        target_mae = control.get("target_mae", {})
        record["mpc_objective"] = control.get("objective_mean", np.nan)
        record["mpc_control_delta"] = control.get("control_delta_mae", np.nan)
        record["mpc_action_tv"] = control.get("action_tv", np.nan)
        record["mpc_tair_mae"] = target_mae.get("Tair", np.nan)
        record["mpc_rhair_mae"] = target_mae.get("Rhair", np.nan)
        record["mpc_co2_mae"] = target_mae.get("CO2air", np.nan)
        records.append(record)
    return records


def _role_for(analysis: dict, target: str, metric: str) -> str:
    return analysis.get("metric_roles", {}).get(target, {}).get(metric, "missing")


def _robustness_row(analysis: dict, target: str, metric: str) -> dict:
    for row in analysis["robustness_rows"]:
        if row["control_target"] == target and row["metric"] == metric:
            return row
    return {}


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "start_idx",
        "model_count",
        "control_target",
        "metric",
        "role",
        "full_spearman",
        "full_pairwise_consistency",
        "leave_one_model_spearman_min",
        "leave_one_model_pairwise_min",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, analyses: dict[int, dict], rows: list[dict]) -> None:
    lines = [
        "# FCTV Multi-Start Transfer Robustness",
        "",
        "This report reuses the same forecast-side FCTV metrics and replaces closed-loop outcomes with repeated `GradientMPC` 96-step rollouts from multiple test-set start indices.",
        "",
        "| start_idx | model_count | control_target | metric | role | spearman | pairwise | leave-model spearman min |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["start_idx"]),
                    str(row["model_count"]),
                    row["control_target"],
                    row["metric"],
                    row["role"],
                    f"{float(row['full_spearman']):.3f}",
                    f"{float(row['full_pairwise_consistency']):.3f}",
                    f"{float(row['leave_one_model_spearman_min']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- A metric is reusable only if its role and rank/pairwise statistics remain stable across start indices.",
            "- If a metric changes role across start indices, report it as segment-dependent rather than as a universal selector.",
            "- Whole-objective screening still requires final closed-loop validation even when per-target metrics are stable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_multistart(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{target}\n{metric}" for target, metric in KEY_METRICS]
    starts = sorted({int(row["start_idx"]) for row in rows})
    x = np.arange(len(labels))
    width = 0.8 / max(len(starts), 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for idx, start in enumerate(starts):
        subset = [
            next(
                row
                for row in rows
                if int(row["start_idx"]) == start and row["control_target"] == target and row["metric"] == metric
            )
            for target, metric in KEY_METRICS
        ]
        offset = (idx - (len(starts) - 1) / 2.0) * width
        axes[0].bar(
            x + offset,
            [row["full_spearman"] for row in subset],
            width=width,
            label=f"start {start}",
        )
        axes[1].bar(
            x + offset,
            [row["full_pairwise_consistency"] for row in subset],
            width=width,
            label=f"start {start}",
        )
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_title("Spearman by start index")
    axes[0].set_ylim(-1.0, 1.0)
    axes[1].set_title("Pairwise consistency by start index")
    axes[1].set_ylim(0.0, 1.0)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="lower right")
    fig.suptitle("FCTV multi-start transfer robustness", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_multistart(base_path: Path, suite_path: Path, prefix: str) -> dict:
    cfg = AGCConfig()
    ensure_results_layout(cfg)
    base_by_predictor = _load_base_records(base_path)
    controls_by_start = _load_control_records(suite_path)
    analysis_dir = Path(cfg.forecast_analysis_dir)
    figure_dir = Path(cfg.forecast_figures_dir) / "comparisons"

    analyses = {}
    summary_rows = []
    for start_idx, control_by_predictor in sorted(controls_by_start.items()):
        records = _records_for_start(base_by_predictor, control_by_predictor)
        if len(records) < 4:
            continue
        analysis = _analyze(records)
        analysis["start_idx"] = start_idx
        analyses[start_idx] = analysis
        start_prefix = f"{prefix}_start{start_idx:05d}"
        (analysis_dir / f"{start_prefix}.json").write_text(
            json.dumps(_json_ready(analysis), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        _write_csv(analysis_dir / f"{start_prefix}.csv", analysis["metric_transfer_rows"])
        _write_robustness_csv(analysis_dir / f"{start_prefix}_robustness.csv", analysis["robustness_rows"])
        _write_markdown(analysis_dir / f"{start_prefix}.md", analysis)
        _plot(figure_dir / f"{start_prefix}.png", analysis["metric_transfer_rows"])
        _plot_robustness(figure_dir / f"{start_prefix}_robustness.png", analysis["robustness_rows"])
        _plot_summary_dashboard(figure_dir / f"{start_prefix}_summary.png", analysis)

        for target, metric in KEY_METRICS:
            row = _robustness_row(analysis, target, metric)
            if not row:
                continue
            summary_rows.append(
                {
                    "start_idx": start_idx,
                    "model_count": analysis["model_count"],
                    "control_target": target,
                    "metric": metric,
                    "role": _role_for(analysis, target, metric),
                    "full_spearman": row["full_spearman"],
                    "full_pairwise_consistency": row["full_pairwise_consistency"],
                    "leave_one_model_spearman_min": row["leave_one_model_spearman_min"],
                    "leave_one_model_pairwise_min": row["leave_one_model_pairwise_min"],
                }
            )

    payload = {
        "suite_path": str(suite_path),
        "base_path": str(base_path),
        "start_indices": sorted(analyses.keys()),
        "key_metrics": [{"control_target": target, "metric": metric} for target, metric in KEY_METRICS],
        "summary_rows": summary_rows,
    }
    (analysis_dir / f"{prefix}.json").write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_summary_csv(analysis_dir / f"{prefix}.csv", summary_rows)
    _write_summary_md(analysis_dir / f"{prefix}.md", analyses, summary_rows)
    _plot_multistart(figure_dir / f"{prefix}.png", summary_rows)
    return payload


def parse_args() -> argparse.Namespace:
    cfg = AGCConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-validation",
        type=Path,
        default=Path(cfg.forecast_analysis_dir) / "control_relevant_validation_reference.json",
    )
    parser.add_argument("--suite-json", type=Path, required=True)
    parser.add_argument("--prefix", default="forecast_to_control_transfer_multistart_reference")
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
    if not args.base_validation.exists():
        candidate = original_cwd / args.base_validation
        if candidate.exists():
            args.base_validation = candidate
    payload = analyze_multistart(args.base_validation, args.suite_json, args.prefix)
    print(f"Analyzed start indices: {payload['start_indices']}")


if __name__ == "__main__":
    main()
