# -*- coding: utf-8 -*-
"""Build a three-model PHF comparison dashboard for forecasting and control."""

from __future__ import annotations

import argparse
import json
import os
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


MODELS = [
    "itransformer_co2_control_aware_fusion",
    "itransformer_co2_late_frozen_expert",
    "itransformer_co2_horizon_mixture",
]

LABELS = {
    "itransformer_co2_control_aware_fusion": "Control-aware fusion",
    "itransformer_co2_late_frozen_expert": "Late frozen expert",
    "itransformer_co2_horizon_mixture": "Horizon mixture",
}

COLORS = {
    "itransformer_co2_control_aware_fusion": "#2c6fbb",
    "itransformer_co2_late_frozen_expert": "#1f7a4d",
    "itransformer_co2_horizon_mixture": "#c0392b",
}

ACTION_PANELS = ["co2_sp", "t_vent_sp", "assim_sp"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_adapter(
    project_root: Path,
    predictor: str,
    cfg: AGCConfig,
    raw_bundle,
    scaled_bundle,
    device: torch.device,
) -> PredictiveControlAdapter:
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


def _run_gradient_trace(
    predictor: str,
    adapter: PredictiveControlAdapter,
    raw_bundle,
    cfg: AGCConfig,
) -> dict:
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


def _load_rank_map(project_root: Path, compartment: str) -> dict[str, float]:
    path = project_root / "results" / "forecasting" / "analysis" / f"control_relevant_validation_{compartment.lower()}.json"
    payload = _load_json(path)
    return {
        record["predictor"]: float(record["control_relevant_mean_rank"])
        for record in payload.get("ranked_summary", [])
    }


def _normalize_pair(trace: dict, action_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    actions = np.asarray(trace["executed_actions"], dtype=np.float32)
    baseline = np.asarray(trace["baseline_actions"], dtype=np.float32)
    cols = trace["control_cols"]
    if action_name not in cols:
        return None
    idx = cols.index(action_name)
    lo = min(float(np.min(actions[:, idx])), float(np.min(baseline[:, idx])))
    hi = max(float(np.max(actions[:, idx])), float(np.max(baseline[:, idx])))
    denom = hi - lo if hi > lo else 1.0
    return (actions[:, idx] - lo) / denom, (baseline[:, idx] - lo) / denom


def _plot_dashboard(project_root: Path, cfg: AGCConfig, output: Path, json_out: Path) -> None:
    analysis_dir = project_root / "results" / "forecasting" / "analysis"
    summaries = {
        model: _load_json(analysis_dir / f"{model}_joint_all_{cfg.control_compartment.lower()}_summary.json")
        for model in MODELS
    }
    control_summaries = {
        model: _load_json(project_root / "results" / "control" / "summaries" / f"{model}_gradient_mpc_summary.json")
        for model in MODELS
    }
    rank_map = _load_rank_map(project_root, cfg.control_compartment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AGCDataProcessor(cfg)
    raw_bundle = processor.build_compartment_raw_bundle(cfg.control_compartment)
    scaled_bundle = processor.build_compartment_bundle(cfg.control_compartment)
    traces = {}
    for model in MODELS:
        adapter = _build_adapter(project_root, model, cfg, raw_bundle, scaled_bundle, device)
        traces[model] = _run_gradient_trace(model, adapter, raw_bundle, cfg)

    target_cols = traces[MODELS[0]]["target_cols"]
    co2_idx = target_cols.index("CO2air")
    steps_forecast = np.arange(1, len(summaries[MODELS[0]]["representative_window"]["true"]) + 1)
    steps_control = np.arange(1, len(traces[MODELS[0]]["predicted_targets"]) + 1)

    fig = plt.figure(figsize=(18, 22))
    grid = fig.add_gridspec(6, 2, height_ratios=[1.0, 1.2, 1.0, 1.2, 1.2, 1.2], hspace=0.35, wspace=0.22)

    ax_offline = fig.add_subplot(grid[0, 0])
    x = np.arange(len(MODELS))
    width = 0.35
    full_vals = [summaries[m]["metrics_by_target"]["CO2air"]["full_mae"] for m in MODELS]
    final_vals = [summaries[m]["metrics_by_target"]["CO2air"]["final_mae"] for m in MODELS]
    bars1 = ax_offline.bar(x - width / 2, full_vals, width, label="CO2 full MAE", color="#7aa6d1")
    bars2 = ax_offline.bar(x + width / 2, final_vals, width, label="CO2 final MAE", color="#355c7d")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax_offline.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax_offline.set_xticks(x, [LABELS[m] for m in MODELS], rotation=10, ha="right")
    ax_offline.set_title("Offline CO2 forecasting")
    ax_offline.set_ylabel("MAE")
    ax_offline.grid(axis="y", alpha=0.25)
    ax_offline.legend(fontsize=9)

    ax_rank = fig.add_subplot(grid[0, 1])
    rank_vals = [rank_map.get(m, np.nan) for m in MODELS]
    co2_control_vals = [control_summaries[m]["target_mae"]["CO2air"] for m in MODELS]
    obj_vals = [control_summaries[m]["objective_mean"] for m in MODELS]
    x2 = np.arange(len(MODELS))
    width2 = 0.25
    bars_rank = ax_rank.bar(x2 - width2, rank_vals, width2, label="Validation rank", color="#2c6fbb")
    bars_co2 = ax_rank.bar(x2, co2_control_vals, width2, label="Closed-loop CO2 MAE", color="#1f7a4d")
    bars_obj = ax_rank.bar(x2 + width2, obj_vals, width2, label="Objective", color="#c0392b")
    for bars in (bars_rank, bars_co2, bars_obj):
        for bar in bars:
            h = bar.get_height()
            ax_rank.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    ax_rank.set_xticks(x2, [LABELS[m] for m in MODELS], rotation=10, ha="right")
    ax_rank.set_title("Control transfer summary")
    ax_rank.grid(axis="y", alpha=0.25)
    ax_rank.legend(fontsize=9)

    ax_forecast = fig.add_subplot(grid[1, :])
    true_co2 = np.asarray(summaries[MODELS[0]]["representative_window"]["true"], dtype=np.float32)[:, 2]
    ax_forecast.plot(steps_forecast, true_co2, color="#111111", linewidth=2.4, label="Ground truth")
    for model in MODELS:
        pred = np.asarray(summaries[model]["representative_window"]["pred"], dtype=np.float32)[:, 2]
        ax_forecast.plot(steps_forecast, pred, linewidth=2.0, color=COLORS[model], label=LABELS[model])
    ax_forecast.set_title("Representative 24-step CO2 forecast window")
    ax_forecast.set_ylabel("CO2air")
    ax_forecast.grid(True, alpha=0.25)
    ax_forecast.legend(ncol=4, fontsize=9)

    ax_first6 = fig.add_subplot(grid[2, 0])
    true_first6 = true_co2[:6]
    ax_first6.plot(np.arange(1, 7), true_first6, color="#111111", linewidth=2.2, label="Ground truth")
    for model in MODELS:
        pred = np.asarray(summaries[model]["representative_window"]["pred"], dtype=np.float32)[:6, 2]
        ax_first6.plot(np.arange(1, 7), pred, linewidth=2.0, color=COLORS[model], label=LABELS[model])
    ax_first6.set_title("First 6 forecast steps")
    ax_first6.set_ylabel("CO2air")
    ax_first6.grid(True, alpha=0.25)

    ax_first6_err = fig.add_subplot(grid[2, 1])
    for model in MODELS:
        pred = np.asarray(summaries[model]["representative_window"]["pred"], dtype=np.float32)[:6, 2]
        err = np.abs(pred - true_first6)
        ax_first6_err.plot(np.arange(1, 7), err, linewidth=2.0, color=COLORS[model], label=LABELS[model])
    ax_first6_err.set_title("First 6 forecast abs error")
    ax_first6_err.set_ylabel("|pred - true|")
    ax_first6_err.grid(True, alpha=0.25)
    ax_first6_err.legend(fontsize=9)

    ax_closed_loop = fig.add_subplot(grid[3, :])
    ref = np.asarray(traces[MODELS[0]]["reference_targets"], dtype=np.float32)[:, co2_idx]
    ax_closed_loop.plot(steps_control, ref, color="#111111", linewidth=2.4, label="Reference")
    for model in MODELS:
        pred = np.asarray(traces[model]["predicted_targets"], dtype=np.float32)[:, co2_idx]
        ax_closed_loop.plot(steps_control, pred, linewidth=1.9, color=COLORS[model], label=LABELS[model])
    ax_closed_loop.set_title("GradientMPC 96-step closed-loop CO2 trajectory")
    ax_closed_loop.set_ylabel("CO2air")
    ax_closed_loop.grid(True, alpha=0.25)
    ax_closed_loop.legend(ncol=4, fontsize=9)

    ax_control_err = fig.add_subplot(grid[4, :])
    for model in MODELS:
        pred = np.asarray(traces[model]["predicted_targets"], dtype=np.float32)[:, co2_idx]
        err = np.abs(pred - ref)
        ax_control_err.plot(steps_control, err, linewidth=1.8, color=COLORS[model], label=LABELS[model])
    ax_control_err.set_title("Closed-loop CO2 absolute tracking error")
    ax_control_err.set_ylabel("|pred - ref|")
    ax_control_err.grid(True, alpha=0.25)
    ax_control_err.legend(ncol=3, fontsize=9)

    action_grid = grid[5, :].subgridspec(1, len(ACTION_PANELS), wspace=0.25)
    for idx, action_name in enumerate(ACTION_PANELS):
        ax = fig.add_subplot(action_grid[0, idx])
        for model in MODELS:
            normalized = _normalize_pair(traces[model], action_name)
            if normalized is None:
                continue
            executed, baseline = normalized
            ax.plot(steps_control, executed, linewidth=1.8, color=COLORS[model], label=LABELS[model])
            ax.plot(steps_control, baseline, linewidth=1.0, linestyle=":", color=COLORS[model], alpha=0.55)
        ax.set_title(f"Normalized action: {action_name}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Closed-loop step")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "PHF Triplet Comparison: Control-aware fusion vs Late frozen expert vs Horizon mixture\n"
        f"compartment={cfg.control_compartment} | GradientMPC steps={cfg.control_eval_steps}",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "models": MODELS,
        "labels": {m: LABELS[m] for m in MODELS},
        "forecast_sample_idx": summaries[MODELS[0]]["representative_window"]["sample_idx"],
        "control_steps": cfg.control_eval_steps,
        "compartment": cfg.control_compartment,
        "offline_co2": {
            m: {
                "full_mae": summaries[m]["metrics_by_target"]["CO2air"]["full_mae"],
                "final_mae": summaries[m]["metrics_by_target"]["CO2air"]["final_mae"],
            }
            for m in MODELS
        },
        "control_transfer": {
            m: {
                "validation_rank": rank_map.get(m),
                "objective_mean": control_summaries[m]["objective_mean"],
                "co2_mae": control_summaries[m]["target_mae"]["CO2air"],
            }
            for m in MODELS
        },
        "figure_path": str(output.relative_to(project_root)),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument(
        "--output",
        default="results/forecasting/figures/comparisons/phf_triplet_forecast_control_dashboard.png",
    )
    parser.add_argument(
        "--summary-json",
        default="results/forecasting/analysis/phf_triplet_forecast_control_dashboard.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    cfg = _apply_three_target_control_protocol(AGCConfig())
    cfg.control_compartment = args.compartment
    _set_global_seed(cfg.seed)
    ensure_results_layout(cfg)
    _plot_dashboard(project_root, cfg, project_root / args.output, project_root / args.summary_json)
    print(f"Saved figure: {project_root / args.output}")
    print(f"Saved summary: {project_root / args.summary_json}")


if __name__ == "__main__":
    main()
