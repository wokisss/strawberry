# -*- coding: utf-8 -*-
"""Diagnose why offline forecast gains may fail to transfer to MPC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
    "itransformer_co2_late_residual",
    "itransformer_co2_late_frozen_expert",
    "itransformer_co2_horizon_mixture",
    "itransformer_co2_frozen_backbone_horizon_mixture",
]


def _inverse_targets(y_scaler, arr: np.ndarray) -> np.ndarray:
    shape = arr.shape
    flat = arr.reshape(-1, shape[-1])
    inv = y_scaler.inverse_transform(flat)
    return inv.reshape(shape)


def _load_adapter(predictor: str, cfg: AGCConfig, scaled_bundle, raw_bundle, device: torch.device) -> PredictiveControlAdapter:
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


def _sample_indices(start_idx: int, available: int, count: int, stride: int) -> list[int]:
    indices = list(range(start_idx, available, max(stride, 1)))
    return indices[:count]


def _logged_forecast_metrics(adapter: PredictiveControlAdapter, scaled_bundle, indices: list[int]) -> dict:
    x = torch.tensor(scaled_bundle["X_past_test"][indices], dtype=torch.float32, device=adapter.device)
    w = torch.tensor(scaled_bundle["W_future_test"][indices], dtype=torch.float32, device=adapter.device)
    u = torch.tensor(scaled_bundle["U_future_test"][indices], dtype=torch.float32, device=adapter.device)
    with torch.no_grad():
        pred_scaled = adapter.predict_scaled(x, w, u).detach().cpu().numpy()
    true_scaled = scaled_bundle["Y_future_test"][indices]
    pred_real = _inverse_targets(scaled_bundle["scalers"]["y"], pred_scaled)
    true_real = _inverse_targets(scaled_bundle["scalers"]["y"], true_scaled)
    abs_err = np.abs(pred_real - true_real)
    return {
        "first_step_mae": {
            target: float(abs_err[:, 0, idx].mean())
            for idx, target in enumerate(adapter.y_cols)
        },
        "full_horizon_mae": {
            target: float(abs_err[:, :, idx].mean())
            for idx, target in enumerate(adapter.y_cols)
        },
        "final_step_mae": {
            target: float(abs_err[:, -1, idx].mean())
            for idx, target in enumerate(adapter.y_cols)
        },
    }


def _gradient_diagnostics(adapter: PredictiveControlAdapter, raw_bundle, indices: list[int]) -> dict:
    cost_grads = []
    co2_grads = []
    first_co2_grads = []
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
        baseline_short_unit = adapter.u_real_to_unit(baseline_u_real[: adapter.cfg.control_horizon]).unsqueeze(0)
        short_plan_unit = baseline_short_unit.clone().detach().requires_grad_(True)
        plan_unit = adapter.expand_control_plan(short_plan_unit)
        baseline_plan_unit = adapter.expand_control_plan(baseline_short_unit)
        u_scaled = adapter.u_unit_to_scaled(plan_unit)
        pred_scaled = adapter.predict_scaled(x_scaled, w_scaled, u_scaled)
        cost = adapter.control_cost(
            pred_scaled,
            ref_scaled,
            plan_unit,
            baseline_plan_unit,
            torch.zeros_like(baseline_short_unit[:, 0]),
        ).mean()
        cost.backward(retain_graph=True)
        cost_grads.append(short_plan_unit.grad.detach().abs().squeeze(0).cpu().numpy())
        baseline_costs.append(float(cost.detach().item()))

        short_plan_unit.grad.zero_()
        co2_mean = pred_scaled[..., co2_idx].mean()
        co2_mean.backward(retain_graph=True)
        co2_grads.append(short_plan_unit.grad.detach().abs().squeeze(0).cpu().numpy())

        short_plan_unit.grad.zero_()
        first_co2 = pred_scaled[:, 0, co2_idx].mean()
        first_co2.backward()
        first_co2_grads.append(short_plan_unit.grad.detach().abs().squeeze(0).cpu().numpy())

    cost_grads_arr = np.asarray(cost_grads, dtype=np.float32)
    co2_grads_arr = np.asarray(co2_grads, dtype=np.float32)
    first_co2_grads_arr = np.asarray(first_co2_grads, dtype=np.float32)

    def summarize(arr: np.ndarray) -> dict:
        by_control = arr.mean(axis=(0, 1))
        top = sorted(
            [
                {"control": name, "mean_abs_grad": float(by_control[idx])}
                for idx, name in enumerate(adapter.u_cols)
            ],
            key=lambda item: item["mean_abs_grad"],
            reverse=True,
        )
        return {
            "mean_abs_grad": float(arr.mean()),
            "max_abs_grad": float(arr.max()),
            "by_control": {name: float(by_control[idx]) for idx, name in enumerate(adapter.u_cols)},
            "top_controls": top[:5],
        }

    return {
        "baseline_control_cost_mean": float(np.mean(baseline_costs)),
        "baseline_control_cost_std": float(np.std(baseline_costs)),
        "cost_gradient": summarize(cost_grads_arr),
        "co2_mean_gradient": summarize(co2_grads_arr),
        "co2_first_step_gradient": summarize(first_co2_grads_arr),
    }


def run_diagnostic(cfg: AGCConfig, predictors: list[str], sample_count: int, stride: int) -> dict:
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
    indices = _sample_indices(cfg.control_start_idx, available, sample_count, stride)

    results = {
        "compartment": cfg.control_compartment,
        "control_horizon": cfg.control_horizon,
        "forecast_horizon": cfg.horizon,
        "sample_count": len(indices),
        "sample_stride": stride,
        "sample_indices": indices,
        "predictors": {},
    }

    for predictor in predictors:
        print(f"Diagnosing {predictor} ...")
        adapter = _load_adapter(predictor, cfg, scaled_bundle, raw_bundle, device)
        results["predictors"][predictor] = {
            "logged_forecast_metrics": _logged_forecast_metrics(adapter, scaled_bundle, indices),
            "gradient_diagnostics": _gradient_diagnostics(adapter, raw_bundle, indices),
        }

    output_path = Path(cfg.forecast_analysis_dir) / "control_sensitivity_diagnostic_reference.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved diagnostic: {output_path}")
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
    run_diagnostic(cfg, args.predictors, args.samples, args.stride)


if __name__ == "__main__":
    main()
