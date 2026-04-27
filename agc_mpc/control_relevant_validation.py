# -*- coding: utf-8 -*-
"""Control-relevant validation for AGC forecasting models used by MPC."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import AGCConfig
from control.controller import PredictiveControlAdapter
from control_main import (
    _apply_three_target_control_protocol,
    _build_model_specs,
    _load_checkpoint,
    _load_frozen_expert_if_needed,
    _load_main_if_needed,
    _set_global_seed,
)
from data_processing.processor import AGCDataProcessor
from results_utils import ensure_results_layout


DEFAULT_PREDICTORS = [
    "itransformer_residual",
    "itransformer_co2_late_residual",
    "itransformer_co2_late_frozen_expert",
    "itransformer_co2_recoupled_expert",
    "itransformer_co2_horizon_mixture",
    "itransformer_co2_frozen_backbone_horizon_mixture",
    "itransformer_co2_control_aware_fusion",
]

TARGETS_FOR_RANKING = [
    "co2_first_step_mae",
    "co2_control_horizon_mae",
    "co2_weighted_horizon_mae",
    "co2_final_step_mae",
    "co2_control_horizon_abs_bias",
    "co2_constraint_near_mae_proxy",
    "mpc_objective",
    "mpc_co2_mae",
]

PREDICTOR_LABELS = {
    "itransformer_residual": "Residual",
    "itransformer_co2_late_residual": "Late residual",
    "itransformer_co2_late_frozen_expert": "Late frozen expert",
    "itransformer_co2_recoupled_expert": "Recoupled expert",
    "itransformer_co2_horizon_mixture": "Horizon mixture",
    "itransformer_co2_frozen_backbone_horizon_mixture": "Frozen-backbone mix",
    "itransformer_co2_control_aware_fusion": "Control-aware fusion",
}


def _inverse_targets(y_scaler, arr: np.ndarray) -> np.ndarray:
    shape = arr.shape
    flat = arr.reshape(-1, shape[-1])
    inv = y_scaler.inverse_transform(flat)
    return inv.reshape(shape)


def _sample_indices(start_idx: int, available: int, count: int, stride: int) -> list[int]:
    indices = list(range(start_idx, available, max(stride, 1)))
    return indices[:count]


def _load_adapter(
    predictor: str,
    cfg: AGCConfig,
    scaled_bundle,
    raw_bundle,
    device: torch.device,
) -> PredictiveControlAdapter:
    model_specs = _build_model_specs(scaled_bundle, cfg)
    if predictor not in model_specs:
        raise ValueError(f"Unsupported predictor: {predictor}")

    model = model_specs[predictor]["builder"]()
    _load_frozen_expert_if_needed(model, predictor, cfg, device)
    _load_main_if_needed(model, predictor, cfg, device)
    _load_checkpoint(
        model,
        Path(cfg.forecast_checkpoints_dir) / model_specs[predictor]["checkpoint"],
        device,
    )
    return PredictiveControlAdapter(
        model=model,
        scalers=scaled_bundle["scalers"],
        feature_groups=scaled_bundle["feature_groups"],
        cfg=cfg,
        raw_bundle=raw_bundle,
        device=device,
    )


def _logged_forecast_metrics(
    adapter: PredictiveControlAdapter,
    cfg: AGCConfig,
    scaled_bundle,
    indices: list[int],
) -> dict:
    x = torch.tensor(scaled_bundle["X_past_test"][indices], dtype=torch.float32, device=adapter.device)
    w = torch.tensor(scaled_bundle["W_future_test"][indices], dtype=torch.float32, device=adapter.device)
    u = torch.tensor(scaled_bundle["U_future_test"][indices], dtype=torch.float32, device=adapter.device)
    with torch.no_grad():
        pred_scaled = adapter.predict_scaled(x, w, u).detach().cpu().numpy()
    true_scaled = scaled_bundle["Y_future_test"][indices]
    pred_real = _inverse_targets(scaled_bundle["scalers"]["y"], pred_scaled)
    true_real = _inverse_targets(scaled_bundle["scalers"]["y"], true_scaled)
    signed_err = pred_real - true_real
    abs_err = np.abs(pred_real - true_real)

    control_horizon = min(cfg.control_horizon, abs_err.shape[1])
    horizon_weights = np.power(cfg.horizon_decay, np.arange(abs_err.shape[1], dtype=np.float32))
    horizon_weights = horizon_weights / np.maximum(horizon_weights.sum(), 1e-6)

    def by_target(values: np.ndarray) -> dict[str, float]:
        return {target: float(values[idx]) for idx, target in enumerate(adapter.y_cols)}

    first = abs_err[:, 0, :].mean(axis=0)
    control = abs_err[:, :control_horizon, :].mean(axis=(0, 1))
    full = abs_err.mean(axis=(0, 1))
    final = abs_err[:, -1, :].mean(axis=0)
    weighted = (abs_err * horizon_weights.reshape(1, -1, 1)).sum(axis=1).mean(axis=0)

    first_bias = signed_err[:, 0, :].mean(axis=0)
    control_bias = signed_err[:, :control_horizon, :].mean(axis=(0, 1))
    full_bias = signed_err.mean(axis=(0, 1))
    final_bias = signed_err[:, -1, :].mean(axis=0)
    weighted_bias = (signed_err * horizon_weights.reshape(1, -1, 1)).sum(axis=1).mean(axis=0)

    constraint_near_proxy = []
    constraint_near_counts = {}
    for target_idx, target in enumerate(adapter.y_cols):
        true_target = true_real[..., target_idx]
        low = np.nanquantile(true_target, 0.15)
        high = np.nanquantile(true_target, 0.85)
        mask = (true_target <= low) | (true_target >= high)
        constraint_near_counts[target] = int(mask.sum())
        if np.any(mask):
            constraint_near_proxy.append(float(abs_err[..., target_idx][mask].mean()))
        else:
            constraint_near_proxy.append(float("nan"))

    co2_idx = adapter.y_index["CO2air"]
    early = abs_err[:, :control_horizon, co2_idx].mean()
    mid_start = control_horizon
    mid_end = max(mid_start + 1, int(abs_err.shape[1] * 0.75))
    mid = abs_err[:, mid_start:mid_end, co2_idx].mean()
    late = abs_err[:, mid_end:, co2_idx].mean() if mid_end < abs_err.shape[1] else abs_err[:, -1:, co2_idx].mean()
    early_bias = signed_err[:, :control_horizon, co2_idx].mean()
    mid_bias = signed_err[:, mid_start:mid_end, co2_idx].mean()
    late_bias = (
        signed_err[:, mid_end:, co2_idx].mean()
        if mid_end < signed_err.shape[1]
        else signed_err[:, -1:, co2_idx].mean()
    )

    return {
        "first_step_mae": by_target(first),
        "control_horizon_mae": by_target(control),
        "full_horizon_mae": by_target(full),
        "final_step_mae": by_target(final),
        "weighted_horizon_mae": by_target(weighted),
        "first_step_bias": by_target(first_bias),
        "control_horizon_bias": by_target(control_bias),
        "full_horizon_bias": by_target(full_bias),
        "final_step_bias": by_target(final_bias),
        "weighted_horizon_bias": by_target(weighted_bias),
        "constraint_near_mae_proxy": by_target(np.asarray(constraint_near_proxy, dtype=np.float32)),
        "constraint_near_sample_counts": constraint_near_counts,
        "co2_segment_mae": {
            "early_control_horizon": float(early),
            "mid_horizon": float(mid),
            "late_horizon": float(late),
        },
        "co2_segment_bias": {
            "early_control_horizon": float(early_bias),
            "mid_horizon": float(mid_bias),
            "late_horizon": float(late_bias),
        },
    }


def _summarize_gradient_array(arr: np.ndarray, control_cols: list[str]) -> dict:
    abs_arr = np.abs(arr)
    by_control = abs_arr.mean(axis=(0, 1))
    by_control_signed = arr.mean(axis=(0, 1))
    flat_threshold = 1e-5
    top = sorted(
        [
            {"control": name, "mean_abs_grad": float(by_control[idx])}
            for idx, name in enumerate(control_cols)
        ],
        key=lambda item: item["mean_abs_grad"],
        reverse=True,
    )
    return {
        "mean_abs_grad": float(abs_arr.mean()),
        "max_abs_grad": float(abs_arr.max()),
        "by_control": {name: float(by_control[idx]) for idx, name in enumerate(control_cols)},
        "by_control_signed": {name: float(by_control_signed[idx]) for idx, name in enumerate(control_cols)},
        "positive_fraction": {
            name: float((arr[..., idx] > flat_threshold).mean()) for idx, name in enumerate(control_cols)
        },
        "negative_fraction": {
            name: float((arr[..., idx] < -flat_threshold).mean()) for idx, name in enumerate(control_cols)
        },
        "flat_fraction": {
            name: float((np.abs(arr[..., idx]) <= flat_threshold).mean()) for idx, name in enumerate(control_cols)
        },
        "top_controls": top[:5],
    }


def _gradient_diagnostics(adapter: PredictiveControlAdapter, raw_bundle, indices: list[int]) -> dict:
    cost_grads = []
    co2_mean_grads = []
    co2_first_grads = []
    baseline_costs = []
    co2_idx = adapter.y_index["CO2air"]

    for idx in indices:
        current_x_real = raw_bundle["X_past_test"][idx]
        w_future_real = raw_bundle["W_future_test"][idx]
        baseline_u_real = raw_bundle["U_future_test"][idx]
        ref_y_real = raw_bundle["Y_future_test"][idx]

        x_scaled = adapter.x_real_to_scaled(current_x_real).unsqueeze(0)
        w_scaled = adapter.w_real_to_scaled(w_future_real).unsqueeze(0)
        ref_scaled = adapter.build_reference_scaled(ref_y_real)
        baseline_short = adapter.u_real_to_unit(
            baseline_u_real[: adapter.cfg.control_horizon]
        ).unsqueeze(0)
        short_plan = baseline_short.clone().detach().requires_grad_(True)
        plan_unit = adapter.expand_control_plan(short_plan)
        baseline_plan = adapter.expand_control_plan(baseline_short)
        u_scaled = adapter.u_unit_to_scaled(plan_unit)
        pred_scaled = adapter.predict_scaled(x_scaled, w_scaled, u_scaled)
        cost = adapter.control_cost(
            pred_scaled,
            ref_scaled,
            plan_unit,
            baseline_plan,
            torch.zeros_like(baseline_short[:, 0]),
        ).mean()

        cost.backward(retain_graph=True)
        cost_grads.append(short_plan.grad.detach().squeeze(0).cpu().numpy())
        baseline_costs.append(float(cost.detach().item()))

        short_plan.grad.zero_()
        pred_scaled[..., co2_idx].mean().backward(retain_graph=True)
        co2_mean_grads.append(short_plan.grad.detach().squeeze(0).cpu().numpy())

        short_plan.grad.zero_()
        pred_scaled[:, 0, co2_idx].mean().backward()
        co2_first_grads.append(short_plan.grad.detach().squeeze(0).cpu().numpy())

    return {
        "baseline_control_cost_mean": float(np.mean(baseline_costs)),
        "baseline_control_cost_std": float(np.std(baseline_costs)),
        "cost_gradient": _summarize_gradient_array(np.asarray(cost_grads, dtype=np.float32), adapter.u_cols),
        "co2_mean_gradient": _summarize_gradient_array(np.asarray(co2_mean_grads, dtype=np.float32), adapter.u_cols),
        "co2_first_step_gradient": _summarize_gradient_array(np.asarray(co2_first_grads, dtype=np.float32), adapter.u_cols),
    }


def _load_controller_summary(cfg: AGCConfig, predictor: str, controller: str) -> dict:
    path = Path(cfg.control_summaries_dir) / f"{predictor}_{controller}_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _flat_record(predictor: str, payload: dict) -> dict[str, float | str]:
    logged = payload["logged_forecast_metrics"]
    gradient = payload["gradient_diagnostics"]
    mpc = payload.get("closed_loop_gradient_mpc", {})
    recorded = payload.get("closed_loop_recorded", {})
    target_mae = mpc.get("target_mae", {})
    recorded_target_mae = recorded.get("target_mae", {})
    mpc_co2_mae = target_mae.get("CO2air", np.nan)
    recorded_co2_mae = recorded_target_mae.get("CO2air", np.nan)
    mpc_action_tv = mpc.get("action_tv", np.nan)
    recorded_action_tv = recorded.get("action_tv", np.nan)
    return {
        "predictor": predictor,
        "first_step_tair_mae": logged["first_step_mae"]["Tair"],
        "first_step_rhair_mae": logged["first_step_mae"]["Rhair"],
        "co2_first_step_mae": logged["first_step_mae"]["CO2air"],
        "co2_control_horizon_mae": logged["control_horizon_mae"]["CO2air"],
        "co2_weighted_horizon_mae": logged["weighted_horizon_mae"]["CO2air"],
        "co2_full_horizon_mae": logged["full_horizon_mae"]["CO2air"],
        "co2_final_step_mae": logged["final_step_mae"]["CO2air"],
        "co2_first_step_bias": logged["first_step_bias"]["CO2air"],
        "co2_control_horizon_bias": logged["control_horizon_bias"]["CO2air"],
        "co2_control_horizon_abs_bias": abs(logged["control_horizon_bias"]["CO2air"]),
        "co2_final_step_bias": logged["final_step_bias"]["CO2air"],
        "co2_constraint_near_mae_proxy": logged["constraint_near_mae_proxy"]["CO2air"],
        "co2_early_segment_mae": logged["co2_segment_mae"]["early_control_horizon"],
        "co2_mid_segment_mae": logged["co2_segment_mae"]["mid_horizon"],
        "co2_late_segment_mae": logged["co2_segment_mae"]["late_horizon"],
        "co2_early_segment_bias": logged["co2_segment_bias"]["early_control_horizon"],
        "co2_mid_segment_bias": logged["co2_segment_bias"]["mid_horizon"],
        "co2_late_segment_bias": logged["co2_segment_bias"]["late_horizon"],
        "cost_grad_mean_abs": gradient["cost_gradient"]["mean_abs_grad"],
        "co2_first_grad_mean_abs": gradient["co2_first_step_gradient"]["mean_abs_grad"],
        "co2_sp_first_grad": gradient["co2_first_step_gradient"]["by_control"].get("co2_sp", 0.0),
        "co2_sp_first_grad_signed": gradient["co2_first_step_gradient"]["by_control_signed"].get("co2_sp", 0.0),
        "co2_sp_first_grad_positive_fraction": gradient["co2_first_step_gradient"]["positive_fraction"].get("co2_sp", 0.0),
        "co2_sp_first_grad_flat_fraction": gradient["co2_first_step_gradient"]["flat_fraction"].get("co2_sp", 0.0),
        "t_vent_sp_first_grad": gradient["co2_first_step_gradient"]["by_control"].get("t_vent_sp", 0.0),
        "assim_sp_first_grad": gradient["co2_first_step_gradient"]["by_control"].get("assim_sp", 0.0),
        "mpc_objective": mpc.get("objective_mean", np.nan),
        "mpc_control_delta": mpc.get("control_delta_mae", np.nan),
        "mpc_action_tv": mpc_action_tv,
        "recorded_action_tv": recorded_action_tv,
        "action_tv_ratio_vs_recorded": (
            float(mpc_action_tv / max(recorded_action_tv, 1e-6))
            if np.isfinite(mpc_action_tv) and np.isfinite(recorded_action_tv)
            else np.nan
        ),
        "mpc_tair_mae": target_mae.get("Tair", np.nan),
        "mpc_rhair_mae": target_mae.get("Rhair", np.nan),
        "mpc_co2_mae": mpc_co2_mae,
        "recorded_co2_mae": recorded_co2_mae,
        "recorded_policy_co2_improvement": (
            float(recorded_co2_mae - mpc_co2_mae)
            if np.isfinite(recorded_co2_mae) and np.isfinite(mpc_co2_mae)
            else np.nan
        ),
    }


def _rank_records(records: list[dict]) -> list[dict]:
    ranked = [record.copy() for record in records]
    for metric in TARGETS_FOR_RANKING:
        values = np.asarray([float(record.get(metric, np.nan)) for record in records], dtype=np.float32)
        finite = np.isfinite(values)
        order = np.argsort(values[finite])
        ranks = np.full(values.shape, np.nan, dtype=np.float32)
        finite_indices = np.where(finite)[0]
        for rank, local_idx in enumerate(order, start=1):
            ranks[finite_indices[local_idx]] = float(rank)
        for idx, record in enumerate(ranked):
            record[f"{metric}_rank"] = float(ranks[idx]) if np.isfinite(ranks[idx]) else np.nan
    for record in ranked:
        rank_values = [
            float(record[f"{metric}_rank"])
            for metric in TARGETS_FOR_RANKING
            if np.isfinite(float(record[f"{metric}_rank"]))
        ]
        record["control_relevant_mean_rank"] = float(np.mean(rank_values)) if rank_values else np.nan
    return sorted(ranked, key=lambda item: item["control_relevant_mean_rank"])


def _write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys()) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


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


def _write_markdown(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "predictor",
        "control_relevant_mean_rank",
        "co2_first_step_mae",
        "co2_control_horizon_mae",
        "co2_control_horizon_bias",
        "co2_constraint_near_mae_proxy",
        "co2_final_step_mae",
        "mpc_objective",
        "mpc_co2_mae",
        "recorded_policy_co2_improvement",
        "cost_grad_mean_abs",
        "co2_sp_first_grad_signed",
        "co2_sp_first_grad_positive_fraction",
        "co2_sp_first_grad",
        "t_vent_sp_first_grad",
    ]
    lines = [
        "# Control-Relevant Validation Summary",
        "",
        "Lower ranks and lower MAE/objective values are better. Gradient columns are diagnostic magnitudes, not direct objectives.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for record in records:
        values = []
        for col in cols:
            value = record[col]
            if isinstance(value, str):
                values.append(value)
            else:
                values.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_summary(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [PREDICTOR_LABELS.get(record["predictor"], record["predictor"]) for record in records]
    metrics = [
        ("co2_first_step_mae", "CO2 first-step MAE"),
        ("co2_control_horizon_mae", "CO2 first 6-step MAE"),
        ("co2_control_horizon_abs_bias", "CO2 first 6-step abs bias"),
        ("co2_constraint_near_mae_proxy", "CO2 constraint-near proxy MAE"),
        ("co2_final_step_mae", "CO2 final-step MAE"),
        ("mpc_co2_mae", "Closed-loop CO2 MAE"),
        ("mpc_objective", "Closed-loop objective"),
        ("control_relevant_mean_rank", "Mean validation rank"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 20), sharex=True)
    x = np.arange(len(records))
    for ax, (metric, title) in zip(axes, metrics):
        values = [record[metric] for record in records]
        ax.bar(x, values, color="#2c6fbb")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{float(value):.2f}", ha="center", va="bottom", fontsize=8)
    axes[-1].set_xticks(x, labels, rotation=20, ha="right")
    fig.suptitle("Control-Relevant Forecast Validation", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_validation(cfg: AGCConfig, predictors: list[str], samples: int, stride: int) -> dict:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    _set_global_seed(cfg.seed)
    ensure_results_layout(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AGCDataProcessor(cfg)
    raw_bundle = processor.build_compartment_raw_bundle(cfg.control_compartment)
    scaled_bundle = processor.build_compartment_bundle(cfg.control_compartment)
    available = min(
        len(raw_bundle["X_past_test"]),
        len(raw_bundle["W_future_test"]),
        len(raw_bundle["U_future_test"]),
        len(raw_bundle["Y_future_test"]),
    )
    indices = _sample_indices(cfg.control_start_idx, available, samples, stride)

    results = {
        "compartment": cfg.control_compartment,
        "control_horizon": cfg.control_horizon,
        "forecast_horizon": cfg.horizon,
        "sample_count": len(indices),
        "sample_stride": stride,
        "sample_indices": indices,
        "ranking_metrics": TARGETS_FOR_RANKING,
        "predictors": {},
    }
    flat_records = []

    for predictor in predictors:
        print(f"Validating {predictor} ...")
        adapter = _load_adapter(predictor, cfg, scaled_bundle, raw_bundle, device)
        payload = {
            "logged_forecast_metrics": _logged_forecast_metrics(adapter, cfg, scaled_bundle, indices),
            "gradient_diagnostics": _gradient_diagnostics(adapter, raw_bundle, indices),
            "closed_loop_gradient_mpc": _load_controller_summary(cfg, predictor, "gradient_mpc"),
            "closed_loop_recorded": _load_controller_summary(cfg, predictor, "recorded"),
            "closed_loop_cem_mpc": _load_controller_summary(cfg, predictor, "cem_mpc"),
        }
        results["predictors"][predictor] = payload
        flat_records.append(_flat_record(predictor, payload))

    ranked_records = _rank_records(flat_records)
    results["ranked_summary"] = ranked_records

    analysis_dir = Path(cfg.forecast_analysis_dir)
    figure_dir = Path(cfg.forecast_figures_dir) / "comparisons"
    json_path = analysis_dir / f"control_relevant_validation_{cfg.control_compartment.lower()}.json"
    csv_path = analysis_dir / f"control_relevant_validation_{cfg.control_compartment.lower()}.csv"
    md_path = analysis_dir / f"control_relevant_validation_{cfg.control_compartment.lower()}.md"
    figure_path = figure_dir / f"control_relevant_validation_{cfg.control_compartment.lower()}.png"

    json_path.write_text(
        json.dumps(_json_ready(results), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(csv_path, ranked_records)
    _write_markdown(md_path, ranked_records)
    _plot_summary(figure_path, ranked_records)
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved figure: {figure_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictors", nargs="+", default=DEFAULT_PREDICTORS)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--stride", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _apply_three_target_control_protocol(AGCConfig())
    cfg.control_compartment = args.compartment
    run_validation(cfg, args.predictors, args.samples, args.stride)


if __name__ == "__main__":
    main()
