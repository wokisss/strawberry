# -*- coding: utf-8 -*-
"""Analyze whether forecast-side validation metrics predict closed-loop control benefit."""

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


TARGET_PREFIXES = {
    "Tair": "tair",
    "Rhair": "rhair",
    "CO2air": "co2",
}

CONTROL_TARGETS_BY_PREFIX = {
    "tair": "mpc_tair_mae",
    "rhair": "mpc_rhair_mae",
    "co2": "mpc_co2_mae",
}

SELECTION_METRIC_SUFFIXES = [
    "first_step_mae",
    "control_horizon_mae",
    "weighted_horizon_mae",
    "full_horizon_mae",
    "final_step_mae",
    "control_horizon_abs_bias",
    "constraint_near_mae_proxy",
]

SELECTION_METRICS = [
    f"{prefix}_{suffix}"
    for prefix in TARGET_PREFIXES.values()
    for suffix in SELECTION_METRIC_SUFFIXES
]

TRANSFER_SCORE_SUFFIX_WEIGHTS = {
    "first_step_mae": 3.0,
    "control_horizon_mae": 2.0,
    "constraint_near_mae_proxy": 1.5,
    "control_horizon_abs_bias": 1.5,
}

TRANSFER_SCORE_WEIGHTS = {
    f"{prefix}_{suffix}": weight
    for prefix in TARGET_PREFIXES.values()
    for suffix, weight in TRANSFER_SCORE_SUFFIX_WEIGHTS.items()
}

DIAGNOSTIC_METRICS = [
    "cost_grad_mean_abs",
    "tair_first_grad_mean_abs",
    "tair_t_heat_sp_first_grad",
    "tair_t_vent_sp_first_grad",
    "tair_window_pos_lee_sp_first_grad",
    "rhair_first_grad_mean_abs",
    "rhair_dx_sp_first_grad",
    "rhair_t_vent_sp_first_grad",
    "rhair_window_pos_lee_sp_first_grad",
    "rhair_water_sup_intervals_sp_min_first_grad",
    "co2_first_grad_mean_abs",
    "co2_sp_first_grad",
    "co2_sp_first_grad_positive_fraction",
    "co2_sp_first_grad_flat_fraction",
    "t_vent_sp_first_grad",
    "assim_sp_first_grad",
]

CONTROL_TARGETS = [
    "mpc_tair_mae",
    "mpc_rhair_mae",
    "mpc_co2_mae",
    "mpc_objective",
]

MODEL_FAMILIES = {
    "dlinear_forecaster": "linear_baseline",
    "frequency_forecaster": "frequency_baseline",
    "nlinear_forecaster": "linear_baseline",
    "gru_forecaster": "rnn_baseline",
    "lstm_forecaster": "rnn_baseline",
    "segrnn_forecaster": "rnn_baseline",
    "transformer_forecaster": "transformer_baseline",
    "current_hybrid_transformer": "generic_hybrid",
    "transformer_hybrid_residual": "residual_baseline",
    "itransformer_residual": "residual_baseline",
    "patchtst_residual": "residual_baseline",
    "itransformer_co2_late_residual": "co2_residual",
    "itransformer_co2_late_frozen_expert": "phf_expert",
    "itransformer_co2_recoupled_expert": "phf_expert",
    "itransformer_co2_horizon_mixture": "phf_mixture",
    "itransformer_co2_frozen_backbone_horizon_mixture": "phf_mixture",
    "itransformer_co2_control_aware_fusion": "control_aware_fusion",
}


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("ranked_summary", [])
    if not records:
        raise ValueError(f"No ranked_summary records found in {path}")
    return records


def _rank_average(values: np.ndarray) -> np.ndarray:
    ranks = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return ranks
    finite_indices = np.where(finite)[0]
    order = finite_indices[np.argsort(values[finite])]
    sorted_values = values[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x_valid = x[mask]
    y_valid = y[mask]
    if np.std(x_valid) <= 1e-12 or np.std(y_valid) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return _corr(_rank_average(x[mask]), _rank_average(y[mask]))


def _pairwise_consistency(metric_values: np.ndarray, target_values: np.ndarray) -> dict:
    agree = 0
    total = 0
    ties = 0
    n = len(metric_values)
    for i in range(n):
        for j in range(i + 1, n):
            if not (
                np.isfinite(metric_values[i])
                and np.isfinite(metric_values[j])
                and np.isfinite(target_values[i])
                and np.isfinite(target_values[j])
            ):
                continue
            metric_delta = metric_values[i] - metric_values[j]
            target_delta = target_values[i] - target_values[j]
            if abs(metric_delta) <= 1e-12 or abs(target_delta) <= 1e-12:
                ties += 1
                continue
            total += 1
            if np.sign(metric_delta) == np.sign(target_delta):
                agree += 1
    return {
        "agree": agree,
        "total": total,
        "ties_skipped": ties,
        "consistency": float(agree / total) if total else float("nan"),
    }


def _topk_stats(metric_values: np.ndarray, target_values: np.ndarray, k: int) -> dict:
    mask = np.isfinite(metric_values) & np.isfinite(target_values)
    if mask.sum() == 0:
        return {"hit_best": None, "overlap_fraction": None}
    finite_indices = np.where(mask)[0]
    k_eff = min(k, len(finite_indices))
    metric_top = set(finite_indices[np.argsort(metric_values[mask])[:k_eff]].tolist())
    target_top = set(finite_indices[np.argsort(target_values[mask])[:k_eff]].tolist())
    target_best = finite_indices[np.argsort(target_values[mask])[:1]][0]
    return {
        "k": k_eff,
        "hit_best": bool(target_best in metric_top),
        "overlap_fraction": float(len(metric_top & target_top) / max(k_eff, 1)),
    }


def _add_forecast_only_ranks(records: list[dict]) -> None:
    for metric in SELECTION_METRICS:
        values = np.asarray([float(record.get(metric, np.nan)) for record in records], dtype=np.float64)
        ranks = _rank_average(values)
        for idx, record in enumerate(records):
            record[f"{metric}_forecast_only_rank"] = float(ranks[idx]) if np.isfinite(ranks[idx]) else np.nan
    for record in records:
        rank_values = [
            float(record[f"{metric}_forecast_only_rank"])
            for metric in SELECTION_METRICS
            if np.isfinite(float(record.get(f"{metric}_forecast_only_rank", np.nan)))
        ]
        record["forecast_only_transfer_rank"] = float(np.mean(rank_values)) if rank_values else np.nan
        target_scores = []
        for prefix in TARGET_PREFIXES.values():
            weighted_sum = 0.0
            weight_sum = 0.0
            for suffix, weight in TRANSFER_SCORE_SUFFIX_WEIGHTS.items():
                metric = f"{prefix}_{suffix}"
                value = float(record.get(f"{metric}_forecast_only_rank", np.nan))
                if np.isfinite(value):
                    weighted_sum += weight * value
                    weight_sum += weight
            score = float(weighted_sum / weight_sum) if weight_sum > 0 else np.nan
            record[f"{prefix}_transfer_selection_score"] = score
            if np.isfinite(score):
                target_scores.append(score)
        record["multiobjective_transfer_selection_score"] = (
            float(np.mean(target_scores)) if target_scores else np.nan
        )


def _analyze(records: list[dict]) -> dict:
    _add_forecast_only_ranks(records)
    composite_metrics = [
        "forecast_only_transfer_rank",
        "tair_transfer_selection_score",
        "rhair_transfer_selection_score",
        "co2_transfer_selection_score",
        "multiobjective_transfer_selection_score",
    ]
    metrics = SELECTION_METRICS + composite_metrics + DIAGNOSTIC_METRICS
    rows = []
    for metric in metrics:
        metric_values = np.asarray([float(record.get(metric, np.nan)) for record in records], dtype=np.float64)
        metric_kind = "selection" if metric in SELECTION_METRICS or metric in composite_metrics else "diagnostic"
        for target in CONTROL_TARGETS:
            target_values = np.asarray([float(record.get(target, np.nan)) for record in records], dtype=np.float64)
            pairwise = (
                _pairwise_consistency(metric_values, target_values)
                if metric_kind == "selection"
                else {"agree": None, "total": None, "ties_skipped": None, "consistency": None}
            )
            top1 = _topk_stats(metric_values, target_values, 1) if metric_kind == "selection" else {}
            top3 = _topk_stats(metric_values, target_values, 3) if metric_kind == "selection" else {}
            rows.append(
                {
                    "metric": metric,
                    "metric_kind": metric_kind,
                    "control_target": target,
                    "pearson": _corr(metric_values, target_values),
                    "spearman": _spearman(metric_values, target_values),
                    "pairwise_consistency": pairwise.get("consistency"),
                    "pairwise_agree": pairwise.get("agree"),
                    "pairwise_total": pairwise.get("total"),
                    "top1_hits_best": top1.get("hit_best") if top1 else None,
                    "top3_hits_best": top3.get("hit_best") if top3 else None,
                    "top3_overlap_fraction": top3.get("overlap_fraction") if top3 else None,
                }
            )
    robustness = _robustness(records, rows)
    return {
        "model_count": len(records),
        "models": [record["predictor"] for record in records],
        "model_families": {record["predictor"]: _model_family(record["predictor"]) for record in records},
        "targets": TARGET_PREFIXES,
        "selection_metrics": SELECTION_METRICS,
        "transfer_score_weights": TRANSFER_SCORE_WEIGHTS,
        "transfer_score_suffix_weights": TRANSFER_SCORE_SUFFIX_WEIGHTS,
        "diagnostic_metrics": DIAGNOSTIC_METRICS,
        "control_targets": CONTROL_TARGETS,
        "ranked_records": sorted(
            records,
            key=lambda item: item.get("multiobjective_transfer_selection_score", np.inf),
        ),
        "metric_transfer_rows": rows,
        "robustness_rows": robustness,
        "metric_roles": _classify_metric_roles(rows, robustness),
        "summary": _summarize(rows),
    }


def _model_family(predictor: str) -> str:
    return MODEL_FAMILIES.get(predictor, "other")


def _metric_row(records: list[dict], metric: str, target: str) -> dict:
    working = [record.copy() for record in records]
    _add_forecast_only_ranks(working)
    metric_values = np.asarray([float(record.get(metric, np.nan)) for record in working], dtype=np.float64)
    target_values = np.asarray([float(record.get(target, np.nan)) for record in working], dtype=np.float64)
    pairwise = _pairwise_consistency(metric_values, target_values)
    top1 = _topk_stats(metric_values, target_values, 1)
    top3 = _topk_stats(metric_values, target_values, 3)
    return {
        "pearson": _corr(metric_values, target_values),
        "spearman": _spearman(metric_values, target_values),
        "pairwise_consistency": pairwise.get("consistency"),
        "pairwise_agree": pairwise.get("agree"),
        "pairwise_total": pairwise.get("total"),
        "top1_hits_best": top1.get("hit_best"),
        "top3_hits_best": top3.get("hit_best"),
        "top3_overlap_fraction": top3.get("overlap_fraction"),
    }


def _finite_summary(values: list[float]) -> dict:
    arr = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


def _robustness(records: list[dict], full_rows: list[dict]) -> list[dict]:
    metrics = SELECTION_METRICS + [
        "forecast_only_transfer_rank",
        "tair_transfer_selection_score",
        "rhair_transfer_selection_score",
        "co2_transfer_selection_score",
        "multiobjective_transfer_selection_score",
    ]
    rows = []
    full_lookup = {
        (row["metric"], row["control_target"]): row
        for row in full_rows
        if row["metric_kind"] == "selection"
    }
    families = sorted({_model_family(record["predictor"]) for record in records})
    for metric in metrics:
        for target in CONTROL_TARGETS:
            leave_model_spearman = []
            leave_model_pairwise = []
            leave_family_spearman = []
            leave_family_pairwise = []

            for excluded in records:
                subset = [record for record in records if record["predictor"] != excluded["predictor"]]
                row = _metric_row(subset, metric, target)
                leave_model_spearman.append(row["spearman"])
                leave_model_pairwise.append(row["pairwise_consistency"])

            for family in families:
                subset = [record for record in records if _model_family(record["predictor"]) != family]
                if len(subset) < 4:
                    continue
                row = _metric_row(subset, metric, target)
                leave_family_spearman.append(row["spearman"])
                leave_family_pairwise.append(row["pairwise_consistency"])

            full = full_lookup[(metric, target)]
            model_spearman = _finite_summary(leave_model_spearman)
            model_pairwise = _finite_summary(leave_model_pairwise)
            family_spearman = _finite_summary(leave_family_spearman)
            family_pairwise = _finite_summary(leave_family_pairwise)
            rows.append(
                {
                    "metric": metric,
                    "control_target": target,
                    "full_spearman": full["spearman"],
                    "full_pairwise_consistency": full["pairwise_consistency"],
                    "leave_one_model_spearman_min": model_spearman["min"],
                    "leave_one_model_spearman_mean": model_spearman["mean"],
                    "leave_one_model_spearman_max": model_spearman["max"],
                    "leave_one_model_pairwise_min": model_pairwise["min"],
                    "leave_one_model_pairwise_mean": model_pairwise["mean"],
                    "leave_one_model_pairwise_max": model_pairwise["max"],
                    "leave_one_family_spearman_min": family_spearman["min"],
                    "leave_one_family_spearman_mean": family_spearman["mean"],
                    "leave_one_family_spearman_max": family_spearman["max"],
                    "leave_one_family_pairwise_min": family_pairwise["min"],
                    "leave_one_family_pairwise_mean": family_pairwise["mean"],
                    "leave_one_family_pairwise_max": family_pairwise["max"],
                    "role": _classify_one_metric(
                        target,
                        full["spearman"],
                        full["pairwise_consistency"],
                        model_spearman["min"],
                        model_pairwise["min"],
                    ),
                }
            )
    return rows


def _classify_one_metric(
    target: str,
    spearman: float,
    pairwise: float,
    leave_model_spearman_min: float,
    leave_model_pairwise_min: float,
) -> str:
    role_prefix = "objective_" if target == "mpc_objective" else ""
    if spearman >= 0.70 and pairwise >= 0.80 and leave_model_spearman_min >= 0.60:
        return f"{role_prefix}primary_selection"
    if not all(np.isfinite(v) for v in [spearman, pairwise, leave_model_spearman_min, leave_model_pairwise_min]):
        return "insufficient"
    if pairwise >= 0.75 and spearman >= 0.60 and leave_model_pairwise_min >= 0.70 and leave_model_spearman_min >= 0.50:
        return f"{role_prefix}primary_selection"
    if pairwise >= 0.60 and spearman >= 0.35 and leave_model_pairwise_min >= 0.55:
        return f"{role_prefix}secondary_selection"
    if pairwise < 0.55 or abs(spearman) < 0.25:
        return "offline_or_diagnostic_only"
    return "weak_selection"


def _classify_metric_roles(full_rows: list[dict], robustness_rows: list[dict]) -> dict:
    roles = {target: {} for target in CONTROL_TARGETS}
    for row in robustness_rows:
        roles.setdefault(row["control_target"], {})[row["metric"]] = row["role"]
    for row in full_rows:
        if row["metric_kind"] == "diagnostic":
            roles.setdefault(row["control_target"], {})[row["metric"]] = "diagnostic_only"
    return roles


def _summarize(rows: list[dict]) -> dict:
    summary = {}
    for target in CONTROL_TARGETS:
        selection_rows = [
            row for row in rows if row["control_target"] == target and row["metric_kind"] == "selection"
        ]
        best_pairwise = sorted(
            selection_rows,
            key=lambda item: (
                -float(item["pairwise_consistency"] if item["pairwise_consistency"] is not None else -np.inf),
                -abs(float(item["spearman"] if np.isfinite(item["spearman"]) else 0.0)),
            ),
        )[:5]
        best_spearman = sorted(
            selection_rows,
            key=lambda item: -abs(float(item["spearman"] if np.isfinite(item["spearman"]) else 0.0)),
        )[:5]
        summary[target] = {
            "best_pairwise_selection_metrics": best_pairwise,
            "best_spearman_selection_metrics": best_spearman,
        }
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric",
        "metric_kind",
        "control_target",
        "pearson",
        "spearman",
        "pairwise_consistency",
        "pairwise_agree",
        "pairwise_total",
        "top1_hits_best",
        "top3_hits_best",
        "top3_overlap_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_robustness_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric",
        "control_target",
        "role",
        "full_spearman",
        "full_pairwise_consistency",
        "leave_one_model_spearman_min",
        "leave_one_model_spearman_mean",
        "leave_one_model_spearman_max",
        "leave_one_model_pairwise_min",
        "leave_one_model_pairwise_mean",
        "leave_one_model_pairwise_max",
        "leave_one_family_spearman_min",
        "leave_one_family_spearman_mean",
        "leave_one_family_spearman_max",
        "leave_one_family_pairwise_min",
        "leave_one_family_pairwise_mean",
        "leave_one_family_pairwise_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, analysis: dict) -> None:
    rows = analysis["metric_transfer_rows"]
    ranked = analysis["ranked_records"]
    lines = [
        "# Forecast-To-Control Transfer Analysis",
        "",
        f"Model count: `{analysis['model_count']}`.",
        "",
        "This report tests whether forecast-side validation metrics predict `GradientMPC` closed-loop outcomes.",
        "For selection metrics, lower values are treated as better. Gradient metrics are diagnostic only.",
        "",
        "## Metric Roles",
        "",
        "| control_target | metric | role |",
        "| --- | --- | --- |",
    ]
    for target, role_map in analysis["metric_roles"].items():
        for metric, role in role_map.items():
            lines.append(f"| {target} | {metric} | {role} |")
    lines.extend(
        [
            "",
            "## FCTV Transfer Selection Scores",
            "",
            "Each target-specific transfer score is a weighted average of forecast-only metric ranks. Lower is better.",
            "`multiobjective_transfer_selection_score` averages the three target-specific scores.",
            "",
            "| metric suffix | weight |",
            "| --- | --- |",
        ]
    )
    for suffix, weight in TRANSFER_SCORE_SUFFIX_WEIGHTS.items():
        lines.append(f"| {suffix} | {weight:.1f} |")
    lines.extend(
        [
            "",
            "Role definitions:",
            "",
            "- `primary_selection`: stable enough for closed-loop target-specific model selection in the current pool.",
            "- `secondary_selection`: useful supporting selection signal.",
            "- `weak_selection`: directionally useful but not strong enough alone.",
            "- `objective_primary_selection` / `objective_secondary_selection`: useful for whole-objective screening.",
            "- `offline_or_diagnostic_only`: not suitable for control selection by itself.",
            "- `diagnostic_only`: useful for interpretation, not direct ranking.",
            "",
        "## Forecast-Only Transfer Rank",
        "",
        "| rank | predictor | multiobjective_score | tair_score | rhair_score | co2_score | control_relevant_mean_rank | mpc_tair_mae | mpc_rhair_mae | mpc_co2_mae | mpc_objective |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for idx, record in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    record["predictor"],
                    f"{float(record['multiobjective_transfer_selection_score']):.3f}",
                    f"{float(record['tair_transfer_selection_score']):.3f}",
                    f"{float(record['rhair_transfer_selection_score']):.3f}",
                    f"{float(record['co2_transfer_selection_score']):.3f}",
                    f"{float(record['control_relevant_mean_rank']):.3f}",
                    f"{float(record['mpc_tair_mae']):.3f}",
                    f"{float(record['mpc_rhair_mae']):.3f}",
                    f"{float(record['mpc_co2_mae']):.3f}",
                    f"{float(record['mpc_objective']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Metric Transfer Quality", ""])
    for target in CONTROL_TARGETS:
        lines.extend(
            [
                f"### Target: `{target}`",
                "",
                "| metric | kind | pearson | spearman | pairwise | top1_hit | top3_hit | top3_overlap |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        target_rows = [row for row in rows if row["control_target"] == target]
        target_rows = sorted(
            target_rows,
            key=lambda item: (
                item["metric_kind"] != "selection",
                -float(item["pairwise_consistency"] if item["pairwise_consistency"] is not None else -1),
                -abs(float(item["spearman"] if np.isfinite(item["spearman"]) else 0.0)),
            ),
        )
        for row in target_rows:
            def fmt(value):
                if value is None:
                    return ""
                if isinstance(value, bool):
                    return "yes" if value else "no"
                if isinstance(value, (float, np.floating)):
                    return "" if not np.isfinite(value) else f"{float(value):.3f}"
                return str(value)

            lines.append(
                "| "
                + " | ".join(
                    [
                        row["metric"],
                        row["metric_kind"],
                        fmt(row["pearson"]),
                        fmt(row["spearman"]),
                        fmt(row["pairwise_consistency"]),
                        fmt(row["top1_hits_best"]),
                        fmt(row["top3_hits_best"]),
                        fmt(row["top3_overlap_fraction"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Robustness Summary", ""])
    for target in CONTROL_TARGETS:
        lines.extend(
            [
                f"### Target: `{target}`",
                "",
                "| metric | role | full_spearman | leave-model spearman range | leave-family spearman range | leave-model pairwise min |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        target_rows = [
            row for row in analysis["robustness_rows"] if row["control_target"] == target
        ]
        for row in target_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["metric"],
                        row["role"],
                        f"{float(row['full_spearman']):.3f}",
                        f"{float(row['leave_one_model_spearman_min']):.3f} .. {float(row['leave_one_model_spearman_max']):.3f}",
                        f"{float(row['leave_one_family_spearman_min']):.3f} .. {float(row['leave_one_family_spearman_max']):.3f}",
                        f"{float(row['leave_one_model_pairwise_min']):.3f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selection_rows = [row for row in rows if row["metric_kind"] == "selection"]
    metric_order = []
    for prefix in TARGET_PREFIXES.values():
        metric_order.extend(
            [
                f"{prefix}_first_step_mae",
                f"{prefix}_control_horizon_mae",
                f"{prefix}_final_step_mae",
                f"{prefix}_transfer_selection_score",
            ]
        )
    metric_order.append("multiobjective_transfer_selection_score")
    fig, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
    colors = {
        "mpc_tair_mae": "#2c6fbb",
        "mpc_rhair_mae": "#2e8b57",
        "mpc_co2_mae": "#d69c2f",
        "mpc_objective": "#c43c2f",
    }
    y = np.arange(len(metric_order))
    for ax, score_key, title in [
        (axes[0], "spearman", "Spearman rank correlation"),
        (axes[1], "pairwise_consistency", "Pairwise ordering consistency"),
    ]:
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(CONTROL_TARGETS))
        for offset, target in zip(offsets, CONTROL_TARGETS):
            values = []
            for metric in metric_order:
                match = next(
                    row
                    for row in selection_rows
                    if row["metric"] == metric and row["control_target"] == target
                )
                values.append(match[score_key])
            ax.barh(y + offset, values, height=width, color=colors[target], label=target)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(-1.0 if score_key == "spearman" else 0.0, 1.0)
    axes[0].set_yticks(y, metric_order)
    axes[1].legend(loc="lower right")
    fig.suptitle("Forecast-side metrics vs closed-loop control outcomes", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharex=True)
    role_colors = {
        "primary_selection": "#2c6fbb",
        "secondary_selection": "#2e8b57",
        "objective_primary_selection": "#2c6fbb",
        "objective_secondary_selection": "#2e8b57",
        "weak_selection": "#d69c2f",
        "offline_or_diagnostic_only": "#c43c2f",
        "insufficient": "#777777",
    }
    for ax, (target_name, prefix) in zip(axes, TARGET_PREFIXES.items()):
        control_target = CONTROL_TARGETS_BY_PREFIX[prefix]
        metric_order = [
            f"{prefix}_first_step_mae",
            f"{prefix}_control_horizon_mae",
            f"{prefix}_constraint_near_mae_proxy",
            f"{prefix}_control_horizon_abs_bias",
            f"{prefix}_final_step_mae",
            f"{prefix}_transfer_selection_score",
        ]
        target_rows = [
            row
            for row in rows
            if row["control_target"] == control_target and row["metric"] in metric_order
        ]
        target_rows = sorted(target_rows, key=lambda row: metric_order.index(row["metric"]))
        y = np.arange(len(target_rows))
        for idx, row in enumerate(target_rows):
            left = float(row["leave_one_model_spearman_min"])
            right = float(row["leave_one_model_spearman_max"])
            full = float(row["full_spearman"])
            color = role_colors.get(row["role"], "#777777")
            ax.plot([left, right], [idx, idx], color=color, linewidth=5, alpha=0.65)
            ax.scatter([full], [idx], color=color, edgecolor="#222222", zorder=3)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_yticks(y, [row["metric"].replace(f"{prefix}_", "") for row in target_rows])
        ax.set_xlim(-1.0, 1.0)
        ax.set_title(f"{target_name} robustness")
        ax.grid(axis="x", alpha=0.25)
    axes[1].set_xlabel("Spearman correlation range after leaving one model out")
    fig.suptitle("Leave-one-model robustness of FCTV metrics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_dashboard(path: Path, analysis: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = analysis["ranked_records"]
    robustness = []
    for target_name, prefix in TARGET_PREFIXES.items():
        metric = f"{prefix}_first_step_mae"
        match = next(
            (
                row
                for row in analysis["robustness_rows"]
                if row["control_target"] == CONTROL_TARGETS_BY_PREFIX[prefix]
                and row["metric"] == metric
            ),
            None,
        )
        if match is not None:
            robustness.append((target_name, match))
    role_colors = {
        "primary_selection": "#2c6fbb",
        "secondary_selection": "#2e8b57",
        "weak_selection": "#d69c2f",
        "offline_or_diagnostic_only": "#c43c2f",
        "diagnostic_only": "#777777",
        "not_co2_selection": "#777777",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Forecast-to-control validation: current evidence", fontsize=16, y=0.98)

    top = ranked[:6]
    labels = [record["predictor"].replace("itransformer_co2_", "").replace("_", "\n") for record in top]
    scores = [record["multiobjective_transfer_selection_score"] for record in top]
    bar_colors = ["#2c6fbb" if idx == 0 else "#7aa6d9" for idx in range(len(top))]
    axes[0].bar(np.arange(len(top)), scores, color=bar_colors)
    axes[0].set_title("Composite score ranking")
    axes[0].set_ylabel("score rank, lower is better")
    axes[0].set_xticks(np.arange(len(top)), labels, rotation=0, ha="center", fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(scores):
        axes[0].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    y = np.arange(len(robustness))
    for idx, (_, row) in enumerate(robustness):
        left = float(row["leave_one_model_spearman_min"])
        right = float(row["leave_one_model_spearman_max"])
        full = float(row["full_spearman"])
        color = role_colors.get(row["role"], "#777777")
        axes[1].plot([left, right], [idx, idx], color=color, linewidth=5, alpha=0.7)
        axes[1].scatter([full], [idx], color=color, edgecolor="#222222", zorder=3)
    axes[1].axvline(0, color="#222222", linewidth=0.8)
    axes[1].set_yticks(y, [target for target, _ in robustness], fontsize=9)
    axes[1].set_xlim(-0.6, 1.0)
    axes[1].set_title("First-step robustness by target")
    axes[1].set_xlabel("Spearman range after leaving one model out")
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].axis("off")
    roles = analysis["metric_roles"]
    role_lines = [
        ("Tair first-step role", [roles.get("mpc_tair_mae", {}).get("tair_first_step_mae", "unknown")]),
        ("Rhair first-step role", [roles.get("mpc_rhair_mae", {}).get("rhair_first_step_mae", "unknown")]),
        ("CO2 first-step role", [roles.get("mpc_co2_mae", {}).get("co2_first_step_mae", "unknown")]),
        ("Whole objective", [roles.get("mpc_objective", {}).get("multiobjective_transfer_selection_score", "unknown")]),
    ]
    y_text = 0.95
    for title, items in role_lines:
        axes[2].text(0.0, y_text, title, fontsize=11, fontweight="bold", va="top")
        y_text -= 0.08
        for item in items:
            axes[2].text(0.03, y_text, f"- {item}", fontsize=9, va="top")
            y_text -= 0.06
        y_text -= 0.05
    axes[2].text(
        0.0,
        0.08,
        "Conclusion: FCTV is now evaluated per target.\n"
        "CO2 remains the clearest stress case, while Tair\n"
        "and Rhair roles are reported separately instead of\n"
        "being folded into a universal claim.",
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    cfg = AGCConfig()
    default_input = Path(cfg.forecast_analysis_dir) / f"control_relevant_validation_{cfg.control_compartment.lower()}.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--prefix", default="forecast_to_control_transfer_reference")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    cfg = AGCConfig()
    ensure_results_layout(cfg)
    args = parse_args()
    records = _load_records(args.input)
    analysis = _analyze(records)
    analysis_dir = Path(cfg.forecast_analysis_dir)
    figure_dir = Path(cfg.forecast_figures_dir) / "comparisons"

    json_path = analysis_dir / f"{args.prefix}.json"
    csv_path = analysis_dir / f"{args.prefix}.csv"
    md_path = analysis_dir / f"{args.prefix}.md"
    figure_path = figure_dir / f"{args.prefix}.png"
    robustness_prefix = args.prefix.replace(
        "forecast_to_control_transfer",
        "forecast_to_control_transfer_robustness",
        1,
    )
    robustness_csv_path = analysis_dir / f"{robustness_prefix}.csv"
    robustness_figure_path = figure_dir / f"{robustness_prefix}.png"
    summary_prefix = args.prefix.replace(
        "forecast_to_control_transfer",
        "forecast_to_control_transfer_summary",
        1,
    )
    summary_figure_path = figure_dir / f"{summary_prefix}.png"

    json_path.write_text(
        json.dumps(_json_ready(analysis), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(csv_path, analysis["metric_transfer_rows"])
    _write_robustness_csv(robustness_csv_path, analysis["robustness_rows"])
    _write_markdown(md_path, analysis)
    _plot(figure_path, analysis["metric_transfer_rows"])
    _plot_robustness(robustness_figure_path, analysis["robustness_rows"])
    _plot_summary_dashboard(summary_figure_path, analysis)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved robustness CSV: {robustness_csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved robustness figure: {robustness_figure_path}")
    print(f"Saved summary figure: {summary_figure_path}")


if __name__ == "__main__":
    main()
