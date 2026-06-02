# -*- coding: utf-8 -*-
"""Run full-period anchored resource-aware MPC over the Reference test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import AGCConfig
from control.controller import GradientMPCController
from control.simulator import AGCClosedLoopSimulator
from control_main import _apply_three_target_control_protocol, _set_global_seed
from data_processing.processor import AGCDataProcessor
from evaluate_mainline_real_resource_control import (
    _load_json,
    _predict_resource,
    _read_weather,
    _resource_costs,
    _rollout_feature_vector,
)
from results_utils import ensure_results_layout
from run_fctv_multistart_control import _load_adapter


DEFAULT_PREDICTORS = [
    "current_hybrid_transformer",
    "itransformer_co2_residual",
]
DEFAULT_RESOURCE_WEIGHTS = [0.0, 0.05]


def _profile_name(prefix: str, weight: float) -> str:
    return f"{prefix}_w{int(round(weight * 1000)):03d}"


def _generate_anchor_starts(total_samples: int, execution_steps: int, max_segments: int | None) -> list[int]:
    starts = list(range(0, max(total_samples - execution_steps, 0) + 1, execution_steps))
    if max_segments is not None:
        starts = starts[: max(0, int(max_segments))]
    return starts


def _resolve(path: Path, project_root: Path, original_cwd: Path) -> Path:
    if path.is_absolute():
        return path
    if (original_cwd / path).exists():
        return (original_cwd / path).resolve()
    return (project_root / path).resolve()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _remove_if_exists(path: Path) -> None:
    if path and path.is_file():
        path.unlink()


def _add_segment_deltas(rows: list[dict]) -> None:
    baseline = {
        (row["predictor"], row["start_idx"]): row
        for row in rows
        if abs(float(row["resource_weight"])) <= 1e-12
    }
    for row in rows:
        base = baseline.get((row["predictor"], row["start_idx"]))
        if base is None:
            row["cost_change_vs_w000_pct"] = float("nan")
            row["co2_mae_change_vs_w000_pct"] = float("nan")
            row["objective_change_vs_w000_pct"] = float("nan")
            continue
        for metric, out_name in [
            ("estimated_total_resource_cost_eur_m2", "cost_change_vs_w000_pct"),
            ("co2_mae", "co2_mae_change_vs_w000_pct"),
            ("objective_mean", "objective_change_vs_w000_pct"),
        ]:
            denom = float(base[metric])
            row[out_name] = (
                (float(row[metric]) - denom) / denom * 100.0
                if np.isfinite(denom) and abs(denom) > 1e-12
                else float("nan")
            )


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["predictor"], float(row["resource_weight"])), []).append(row)

    out = []
    for (predictor, weight), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["start_idx"]))
        total_steps = sum(int(item["steps"]) for item in items)
        summary = {
            "predictor": predictor,
            "resource_weight": weight,
            "profile": items[0]["profile"],
            "segments": len(items),
            "total_steps": total_steps,
            "coverage_start_timestamp": items[0]["segment_start_timestamp"],
            "coverage_end_timestamp": items[-1]["segment_end_timestamp"],
            "objective_mean": float(np.mean([item["objective_mean"] for item in items])),
            "tair_mae": float(np.mean([item["tair_mae"] for item in items])),
            "rhair_mae": float(np.mean([item["rhair_mae"] for item in items])),
            "co2_mae": float(np.mean([item["co2_mae"] for item in items])),
            "estimated_heat_mj_m2": float(sum(item["estimated_heat_mj_m2"] for item in items)),
            "estimated_electricity_kwh_m2": float(sum(item["estimated_electricity_kwh_m2"] for item in items)),
            "estimated_co2_kg_m2": float(sum(item["estimated_co2_kg_m2"] for item in items)),
            "estimated_irrigation_l_m2": float(sum(item["estimated_irrigation_l_m2"] for item in items)),
            "estimated_heat_cost_eur_m2": float(sum(item["estimated_heat_cost_eur_m2"] for item in items)),
            "estimated_electricity_cost_eur_m2": float(sum(item["estimated_electricity_cost_eur_m2"] for item in items)),
            "estimated_co2_cost_eur_m2": float(sum(item["estimated_co2_cost_eur_m2"] for item in items)),
            "estimated_total_resource_cost_eur_m2": float(
                sum(item["estimated_total_resource_cost_eur_m2"] for item in items)
            ),
        }
        out.append(summary)

    baseline = {
        row["predictor"]: row
        for row in out
        if abs(float(row["resource_weight"])) <= 1e-12
    }
    for row in out:
        base = baseline.get(row["predictor"])
        if base is None:
            row["cost_change_vs_w000_pct"] = float("nan")
            row["co2_mae_change_vs_w000_pct"] = float("nan")
            row["objective_change_vs_w000_pct"] = float("nan")
            continue
        for metric, out_name in [
            ("estimated_total_resource_cost_eur_m2", "cost_change_vs_w000_pct"),
            ("co2_mae", "co2_mae_change_vs_w000_pct"),
            ("objective_mean", "objective_change_vs_w000_pct"),
        ]:
            denom = float(base[metric])
            row[out_name] = (
                (float(row[metric]) - denom) / denom * 100.0
                if np.isfinite(denom) and abs(denom) > 1e-12
                else float("nan")
            )
    return out


def _write_markdown(path: Path, rows: list[dict], detail_rows: list[dict]) -> None:
    lines = [
        "# Full-Period Anchored Resource MPC",
        "",
        "This table summarizes repeated anchored closed-loop MPC segments over the Reference test split. Each segment is re-anchored to true observed AGC history before optimizing the next control window.",
        "",
        "| predictor | w | segments | period | objective | CO2 MAE | heat | electricity | CO2 use | irrigation | resource cost | cost vs w=0 | CO2 vs w=0 |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        period = f"{row['coverage_start_timestamp']} to {row['coverage_end_timestamp']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["predictor"],
                    f"{row['resource_weight']:.2f}",
                    str(row["segments"]),
                    period,
                    f"{row['objective_mean']:.4f}",
                    f"{row['co2_mae']:.3f}",
                    f"{row['estimated_heat_mj_m2']:.3f}",
                    f"{row['estimated_electricity_kwh_m2']:.3f}",
                    f"{row['estimated_co2_kg_m2']:.4f}",
                    f"{row['estimated_irrigation_l_m2']:.3f}",
                    f"{row['estimated_total_resource_cost_eur_m2']:.4f}",
                    f"{row['cost_change_vs_w000_pct']:.1f}%",
                    f"{row['co2_mae_change_vs_w000_pct']:.1f}%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Boundary:",
            "",
            "- MPC resource values are counterfactual estimates from the calibrated AGC resource estimator.",
            "- The comparison supports resource-cost and climate-control trade-off claims only.",
            "- It does not claim true net-profit, yield, or quality improvement.",
            "",
            f"Segment records: `{len(detail_rows)}`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_tradeoff(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for row in rows:
        label = f"{row['predictor'].replace('_', ' ')} w={row['resource_weight']:.2f}"
        ax.scatter(row["estimated_total_resource_cost_eur_m2"], row["co2_mae"], s=80)
        ax.annotate(label, (row["estimated_total_resource_cost_eur_m2"], row["co2_mae"]), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("cumulative estimated resource cost, EUR/m2")
    ax.set_ylabel("mean CO2air MAE")
    ax.set_title("Full-period anchored MPC trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative_cost(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["predictor", "resource_weight", "start_idx"])
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for (predictor, weight), group in frame.groupby(["predictor", "resource_weight"], sort=True):
        group = group.sort_values("start_idx")
        label = f"{predictor.replace('_', ' ')} w={weight:.2f}"
        ax.plot(
            np.arange(1, len(group) + 1),
            group["estimated_total_resource_cost_eur_m2"].cumsum(),
            marker="o" if len(group) <= 12 else None,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("anchored segment")
    ax.set_ylabel("cumulative estimated resource cost, EUR/m2")
    ax.set_title("Cumulative full-period estimated resource cost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_full_period(args: argparse.Namespace) -> tuple[list[dict], list[dict], Path]:
    project_root = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    os.chdir(project_root)

    cfg = _apply_three_target_control_protocol(AGCConfig())
    if args.data_root is not None:
        cfg.data_root = str(_resolve(args.data_root, project_root, original_cwd))
    cfg.control_compartment = args.compartment
    cfg.control_eval_steps = int(args.execution_steps)
    cfg.seed = int(args.seed)
    cfg.control_save_rollout_figures = bool(args.save_segment_figures)
    ensure_results_layout(cfg)
    _set_global_seed(cfg.seed)

    spec_path = _resolve(args.model_spec, project_root, original_cwd)
    spec = _load_json(spec_path)
    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    weather = _read_weather(data_root / "Weather" / "Weather.csv")

    processor = AGCDataProcessor(cfg)
    raw_bundle = processor.build_compartment_raw_bundle(cfg.control_compartment)
    scaled_bundle = processor.build_compartment_bundle(cfg.control_compartment)
    starts = _generate_anchor_starts(len(raw_bundle["X_past_test"]), cfg.control_eval_steps, args.max_segments)
    if not starts:
        raise ValueError("No valid anchor starts were generated.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    suite_records = []

    for weight in args.resource_weights:
        cfg.economic_resource_weight = float(weight)
        cfg.economic_profile_name = _profile_name(args.profile_prefix, float(weight))
        for predictor in args.predictors:
            print(f"Running {predictor} w={weight:.3f}: {len(starts)} anchored segments")
            adapter = _load_adapter(predictor, cfg, scaled_bundle, raw_bundle, device)
            for start_idx in starts:
                cfg.control_start_idx = int(start_idx)
                cfg.control_output_tag = (
                    f"{cfg.economic_profile_name}_anchored_start{int(start_idx):05d}_"
                    f"{cfg.control_eval_steps}steps"
                )
                simulator = AGCClosedLoopSimulator(adapter, raw_bundle, cfg)
                controller = GradientMPCController(adapter, cfg)
                summary = simulator.run(controller, predictor_name=predictor)
                record = asdict(summary)
                record["profile"] = cfg.economic_profile_name
                record["resource_weight"] = float(weight)
                record["output_tag"] = cfg.control_output_tag
                trace_path = Path(record["trace_path"])
                if not trace_path.is_absolute():
                    trace_path = project_root / trace_path
                trace = _load_json(trace_path)
                features = _rollout_feature_vector(trace, weather, spec["features"])
                resources = _predict_resource(spec, features, int(record["steps"]))
                costs = _resource_costs(resources, trace)
                row = {
                    "profile": cfg.economic_profile_name,
                    "predictor": predictor,
                    "resource_weight": float(weight),
                    "start_idx": int(record["start_idx"]),
                    "steps": int(record["steps"]),
                    "segment_start_timestamp": str(trace["timestamps"][0]),
                    "segment_end_timestamp": str(trace["timestamps"][-1]),
                    "objective_mean": float(record["objective_mean"]),
                    "tair_mae": float(record["target_mae"]["Tair"]),
                    "rhair_mae": float(record["target_mae"]["Rhair"]),
                    "co2_mae": float(record["target_mae"]["CO2air"]),
                    "estimated_heat_mj_m2": resources.get("heat_cons_mj_m2", 0.0),
                    "estimated_electricity_kwh_m2": resources.get("electricity_kwh_m2", 0.0),
                    "estimated_co2_kg_m2": resources.get("co2_cons_kg_m2", 0.0),
                    "estimated_irrigation_l_m2": resources.get("irrigation_l_m2", 0.0),
                    **costs,
                    "trace_path": str(trace_path) if args.save_segment_traces else "",
                }
                if not args.save_segment_traces:
                    summary_path_value = str(record.get("summary_path", "")).strip()
                    summary_file = Path(summary_path_value) if summary_path_value else Path()
                    if summary_path_value and not summary_file.is_absolute():
                        summary_file = project_root / summary_file
                    _remove_if_exists(trace_path)
                    if summary_path_value:
                        _remove_if_exists(summary_file)
                    record["trace_path"] = ""
                    record["summary_path"] = ""

                suite_records.append(record)
                rows.append(row)

    _add_segment_deltas(rows)
    summary_rows = _aggregate_rows(rows)

    summaries_dir = project_root / cfg.control_summaries_dir
    figures_dir = project_root / cfg.control_figures_dir
    suffix = "" if args.prefix == "full_period_anchored_resource_mpc" else f"_{args.prefix}"
    segments_path = summaries_dir / f"full_period_anchored_resource_mpc_segments{suffix}.csv"
    summary_path = summaries_dir / f"full_period_anchored_resource_mpc_summary{suffix}.csv"
    md_path = summaries_dir / f"full_period_anchored_resource_mpc_summary{suffix}.md"
    tradeoff_path = figures_dir / f"full_period_anchored_resource_mpc_tradeoff{suffix}.png"
    cumulative_path = figures_dir / f"full_period_anchored_resource_mpc_cumulative_cost{suffix}.png"
    suite_path = summaries_dir / f"full_period_anchored_resource_mpc_suite{suffix}.json"

    _write_csv(segments_path, rows)
    _write_csv(summary_path, summary_rows)
    _write_markdown(md_path, summary_rows, rows)
    _plot_tradeoff(tradeoff_path, summary_rows)
    _plot_cumulative_cost(cumulative_path, rows)

    suite = {
        "predictors": args.predictors,
        "resource_weights": [float(value) for value in args.resource_weights],
        "compartment": cfg.control_compartment,
        "execution_steps": cfg.control_eval_steps,
        "anchor_starts": starts,
        "coverage_start_timestamp": rows[0]["segment_start_timestamp"],
        "coverage_end_timestamp": rows[-1]["segment_end_timestamp"],
        "model_spec": str(spec_path),
        "segments_csv": str(segments_path),
        "summary_csv": str(summary_path),
        "records": suite_records,
    }
    suite_path.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {segments_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {tradeoff_path}")
    print(f"Saved: {cumulative_path}")
    print(f"Saved: {suite_path}")
    return rows, summary_rows, segments_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictors", nargs="+", default=DEFAULT_PREDICTORS)
    parser.add_argument("--resource-weights", nargs="+", type=float, default=DEFAULT_RESOURCE_WEIGHTS)
    parser.add_argument("--execution-steps", type=int, default=24)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-prefix", default="full_period_anchored")
    parser.add_argument("--prefix", default="full_period_anchored_resource_mpc")
    parser.add_argument("--model-spec", type=Path, default=Path("results/control/summaries/agc_resource_cost_model.json"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--save-segment-figures", action="store_true")
    parser.add_argument(
        "--save-segment-traces",
        action="store_true",
        help="Keep per-anchor summary/trace JSON files. By default they are temporary and only aggregate outputs are retained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    digest = hashlib.sha1(
        ("_".join(args.predictors) + "_" + "_".join(str(v) for v in args.resource_weights)).encode("utf-8")
    ).hexdigest()[:8]
    if args.max_segments is not None and args.prefix == "full_period_anchored_resource_mpc":
        args.prefix = f"smoke_{args.max_segments}segments_{digest}"
    run_full_period(args)


if __name__ == "__main__":
    main()
