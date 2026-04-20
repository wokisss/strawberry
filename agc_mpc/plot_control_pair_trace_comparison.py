# -*- coding: utf-8 -*-
"""Trace-based comparison plots for two GradientMPC predictor rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import AGCConfig
from control.controller import GradientMPCController, PredictiveControlAdapter
from control.simulator import AGCClosedLoopSimulator
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


DEFAULT_LEFT = "itransformer_co2_horizon_mixture"
DEFAULT_RIGHT = "itransformer_co2_late_frozen_expert"

PREDICTOR_LABELS = {
    "itransformer_co2_horizon_mixture": "Horizon Mix",
    "itransformer_co2_late_frozen_expert": "Late Frozen Expert",
    "itransformer_co2_recoupled_expert": "Recoupled Expert",
    "itransformer_co2_late_residual": "Late Residual",
    "itransformer_co2_frozen_backbone_horizon_mixture": "Safe Horizon Mix",
}

COLORS = {
    "itransformer_co2_horizon_mixture": "#c0392b",
    "itransformer_co2_late_frozen_expert": "#1f7a4d",
    "itransformer_co2_recoupled_expert": "#2c6fbb",
    "itransformer_co2_late_residual": "#7f8c8d",
    "itransformer_co2_frozen_backbone_horizon_mixture": "#8e44ad",
}

KEY_ACTIONS = ["co2_sp", "t_vent_sp", "assim_sp"]


def _build_adapter(project_root: Path, predictor: str, cfg: AGCConfig, raw_bundle, scaled_bundle, device):
    specs = _build_model_specs(scaled_bundle, cfg)
    if predictor not in specs:
        raise ValueError(f"Unsupported predictor: {predictor}")
    model = specs[predictor]["builder"]()
    _load_frozen_expert_if_needed(model, predictor, cfg, device)
    _load_main_if_needed(model, predictor, cfg, device)
    _load_checkpoint(
        model,
        project_root / "results" / "forecasting" / "checkpoints" / specs[predictor]["checkpoint"],
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


def _run_gradient_trace(predictor: str, adapter: PredictiveControlAdapter, raw_bundle, cfg: AGCConfig) -> dict:
    simulator = AGCClosedLoopSimulator(adapter, raw_bundle, cfg)
    controller = GradientMPCController(adapter, cfg)

    x_test = raw_bundle["X_past_test"]
    w_test = raw_bundle["W_future_test"]
    u_test = raw_bundle["U_future_test"]
    y_test = raw_bundle["Y_future_test"]
    t_test = raw_bundle["t0_test"]

    available_steps = min(len(x_test), len(w_test), len(u_test), len(y_test)) - cfg.control_start_idx - 1
    sim_steps = min(cfg.control_eval_steps, available_steps)
    if sim_steps <= 0:
        raise ValueError("Not enough AGC test samples for the requested control rollout.")

    current_x = x_test[cfg.control_start_idx].copy()
    controller.reset(initial_action_real=u_test[cfg.control_start_idx, 0])

    predicted_targets = []
    reference_targets = []
    executed_actions = []
    baseline_actions = []
    objectives = []
    control_delta = []
    action_tv = []
    timestamps = []

    for step in range(sim_steps):
        idx = cfg.control_start_idx + step
        base_u_window = u_test[idx]
        ref_y_window = y_test[idx]
        plan = controller.optimize(current_x, w_test[idx], base_u_window, ref_y_window)

        executed_action = plan.plan_real[0]
        x_scaled = adapter.x_real_to_scaled(current_x).unsqueeze(0)
        w_scaled = adapter.w_real_to_scaled(w_test[idx]).unsqueeze(0)
        u_scaled = adapter.u_real_to_scaled(adapter._to_tensor(plan.plan_real)).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = adapter.predict_scaled(x_scaled, w_scaled, u_scaled)
        next_targets = adapter.y_scaled_to_real(pred_scaled[:, 0]).detach().cpu().numpy()[0]

        current_row = current_x[-1].copy()
        if cfg.control_rollout_mode == "semi_grounded":
            current_row = x_test[idx + 1, -1].copy()
        current_x = np.concatenate(
            [
                current_x[1:],
                simulator._build_next_row(current_row, executed_action, base_u_window[0], next_targets)[None, :],
            ],
            axis=0,
        )

        stage_ref = simulator._build_stage_reference(ref_y_window)
        if executed_actions:
            step_tv = float(np.mean(np.abs(executed_action - np.asarray(executed_actions[-1], dtype=np.float32))))
        else:
            step_tv = 0.0

        predicted_targets.append(next_targets.tolist())
        reference_targets.append(stage_ref.tolist())
        executed_actions.append(executed_action.tolist())
        baseline_actions.append(base_u_window[0].tolist())
        objectives.append(float(plan.objective))
        control_delta.append(float(np.mean(np.abs(executed_action - base_u_window[0]))))
        action_tv.append(step_tv)
        timestamps.append(str(t_test[idx]))

    pred = np.asarray(predicted_targets, dtype=np.float32)
    ref = np.asarray(reference_targets, dtype=np.float32)
    target_mae = np.mean(np.abs(pred - ref), axis=0)

    return {
        "predictor": predictor,
        "controller": "gradient_mpc",
        "timestamps": timestamps,
        "predicted_targets": predicted_targets,
        "reference_targets": reference_targets,
        "executed_actions": executed_actions,
        "baseline_actions": baseline_actions,
        "objectives": objectives,
        "control_delta": control_delta,
        "action_tv": action_tv,
        "target_mae": {name: float(target_mae[idx]) for idx, name in enumerate(adapter.y_cols)},
        "objective_mean": float(np.mean(objectives)),
        "control_delta_mae": float(np.mean(control_delta)),
        "action_tv_mean": float(np.mean(action_tv)),
        "target_cols": adapter.y_cols,
        "control_cols": adapter.u_cols,
    }


def _plot_horizontal_pair(ax, names, left_values, right_values, left_name, right_name, left_color, right_color, title):
    y = np.arange(len(names))
    height = 0.34
    ax.barh(y + height / 2, left_values, height, color=left_color, label=left_name)
    ax.barh(y - height / 2, right_values, height, color=right_color, label=right_name)
    for idx, value in enumerate(left_values):
        ax.text(value, idx + height / 2, f" {value:.3f}", va="center", fontsize=9)
    for idx, value in enumerate(right_values):
        ax.text(value, idx - height / 2, f" {value:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, names)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=9)


def _plot_action_axis(ax, trace: dict, color: str, label: str, action_name: str) -> None:
    actions = np.asarray(trace["executed_actions"], dtype=np.float32)
    baseline = np.asarray(trace["baseline_actions"], dtype=np.float32)
    cols = trace["control_cols"]
    if action_name not in cols:
        return
    idx = cols.index(action_name)
    lo = min(float(np.min(actions[:, idx])), float(np.min(baseline[:, idx])))
    hi = max(float(np.max(actions[:, idx])), float(np.max(baseline[:, idx])))
    denom = hi - lo if hi > lo else 1.0
    action_unit = (actions[:, idx] - lo) / denom
    baseline_unit = (baseline[:, idx] - lo) / denom
    steps = np.arange(1, len(action_unit) + 1)
    ax.plot(steps, action_unit, color=color, linewidth=2.0, label=label)
    ax.plot(steps, baseline_unit, color=color, linewidth=1.3, linestyle=":", alpha=0.65)


def _write_trace_json(project_root: Path, left: str, right: str, left_trace: dict, right_trace: dict) -> Path:
    path = project_root / "results" / "control" / "summaries" / f"trace_comparison_{left}_vs_{right}_gradient_mpc.json"
    path.write_text(
        json.dumps({"left": left_trace, "right": right_trace}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def build_trace_comparison(left: str, right: str, output: Path | None = None) -> Path:
    project_root = Path(__file__).resolve().parent
    cfg = _apply_three_target_control_protocol(AGCConfig())
    _set_global_seed(cfg.seed)
    ensure_results_layout(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AGCDataProcessor(cfg)
    raw_bundle = processor.build_compartment_raw_bundle(cfg.control_compartment)
    scaled_bundle = processor.build_compartment_bundle(cfg.control_compartment)

    left_adapter = _build_adapter(project_root, left, cfg, raw_bundle, scaled_bundle, device)
    right_adapter = _build_adapter(project_root, right, cfg, raw_bundle, scaled_bundle, device)
    left_trace = _run_gradient_trace(left, left_adapter, raw_bundle, cfg)
    right_trace = _run_gradient_trace(right, right_adapter, raw_bundle, cfg)
    trace_path = _write_trace_json(project_root, left, right, left_trace, right_trace)

    left_label = PREDICTOR_LABELS.get(left, left)
    right_label = PREDICTOR_LABELS.get(right, right)
    left_color = COLORS.get(left, "#c0392b")
    right_color = COLORS.get(right, "#1f7a4d")

    target_cols = left_trace["target_cols"]
    steps = np.arange(1, len(left_trace["predicted_targets"]) + 1)
    left_pred = np.asarray(left_trace["predicted_targets"], dtype=np.float32)
    right_pred = np.asarray(right_trace["predicted_targets"], dtype=np.float32)
    ref = np.asarray(left_trace["reference_targets"], dtype=np.float32)

    fig = plt.figure(figsize=(16, 24))
    grid = fig.add_gridspec(8, 2, height_ratios=[1.35, 1.1, 2.1, 2.1, 2.4, 1.7, 1.7, 1.7])

    ax_mae = fig.add_subplot(grid[0, 0])
    _plot_horizontal_pair(
        ax_mae,
        target_cols,
        [left_trace["target_mae"][name] for name in target_cols],
        [right_trace["target_mae"][name] for name in target_cols],
        left_label,
        right_label,
        left_color,
        right_color,
        "Closed-loop target MAE",
    )

    ax_control = fig.add_subplot(grid[0, 1])
    control_names = ["Objective", "|u - logged|", "Action TV"]
    left_control = [left_trace["objective_mean"], left_trace["control_delta_mae"], left_trace["action_tv_mean"]]
    right_control = [right_trace["objective_mean"], right_trace["control_delta_mae"], right_trace["action_tv_mean"]]
    _plot_horizontal_pair(
        ax_control,
        control_names,
        left_control,
        right_control,
        left_label,
        right_label,
        left_color,
        right_color,
        "MPC behavior metrics",
    )

    ax_delta = fig.add_subplot(grid[1, :])
    deltas = np.asarray([left_trace["target_mae"][name] - right_trace["target_mae"][name] for name in target_cols])
    delta_colors = ["#b03a2e" if value > 0 else "#2874a6" for value in deltas]
    bars = ax_delta.barh(target_cols, deltas, color=delta_colors)
    for bar, value in zip(bars, deltas):
        ax_delta.text(value, bar.get_y() + bar.get_height() / 2, f" {value:+.3f}", va="center", fontsize=10)
    ax_delta.axvline(0.0, color="#333333", linewidth=1.0)
    ax_delta.set_title("MAE gap: positive means the left model is worse")
    ax_delta.grid(axis="x", alpha=0.25)

    for row, target in enumerate(target_cols, start=2):
        idx = target_cols.index(target)
        ax = fig.add_subplot(grid[row, :])
        ax.plot(steps, ref[:, idx], color="#222222", linewidth=2.2, label="Reference")
        ax.plot(steps, left_pred[:, idx], color=left_color, linewidth=2.0, linestyle="--", label=left_label)
        ax.plot(steps, right_pred[:, idx], color=right_color, linewidth=2.0, linestyle="-.", label=right_label)
        ax.set_ylabel(target)
        ax.set_title(
            f"{target} closed-loop trajectory | "
            f"{left_label} MAE={left_trace['target_mae'][target]:.3f}, "
            f"{right_label} MAE={right_trace['target_mae'][target]:.3f}"
        )
        ax.grid(True, alpha=0.25)
        if row == 2:
            ax.legend(ncol=3, fontsize=9)

    ax_err = fig.add_subplot(grid[5, :])
    co2_idx = target_cols.index("CO2air")
    left_err = np.abs(left_pred[:, co2_idx] - ref[:, co2_idx])
    right_err = np.abs(right_pred[:, co2_idx] - ref[:, co2_idx])
    ax_err.plot(steps, left_err, color=left_color, linewidth=2.0, label=left_label)
    ax_err.plot(steps, right_err, color=right_color, linewidth=2.0, label=right_label)
    ax_err.fill_between(steps, left_err, right_err, where=left_err > right_err, color=left_color, alpha=0.12)
    ax_err.fill_between(steps, left_err, right_err, where=right_err > left_err, color=right_color, alpha=0.12)
    ax_err.set_ylabel("CO2 abs error")
    ax_err.set_title("CO2air per-step absolute tracking error")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(ncol=2, fontsize=9)

    for offset, action in enumerate(KEY_ACTIONS):
        ax = fig.add_subplot(grid[6 + offset // 2, offset % 2])
        _plot_action_axis(ax, left_trace, left_color, left_label, action)
        _plot_action_axis(ax, right_trace, right_color, right_label, action)
        ax.set_title(f"Normalized action: {action}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        if offset == 0:
            ax.legend(fontsize=8)
        ax.set_xlabel("Closed-loop step")

    # Hide the unused lower-right panel if there are three key actions.
    if len(KEY_ACTIONS) % 2 == 1:
        ax_unused = fig.add_subplot(grid[7, 1])
        ax_unused.axis("off")
        ax_unused.text(
            0.0,
            0.7,
            f"Trace JSON:\n{trace_path.relative_to(project_root)}",
            fontsize=10,
            va="top",
        )

    fig.suptitle(
        "Trace-Based GradientMPC Comparison\n"
        f"left={left} | right={right} | steps={cfg.control_eval_steps} | compartment={cfg.control_compartment}",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    if output is None:
        output = project_root / "results" / "control" / "figures" / f"comparison_{left}_vs_{right}_gradient_mpc.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trace-based pairwise GradientMPC comparison.")
    parser.add_argument("--left", default=DEFAULT_LEFT, help="Left predictor name.")
    parser.add_argument("--right", default=DEFAULT_RIGHT, help="Right predictor name.")
    parser.add_argument("--output", default=None, help="Optional output figure path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output is not None else None
    saved = build_trace_comparison(args.left, args.right, output)
    print(f"Saved trace comparison figure: {saved}")


if __name__ == "__main__":
    main()
