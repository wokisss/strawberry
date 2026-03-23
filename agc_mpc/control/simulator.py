# -*- coding: utf-8 -*-
"""Semi-grounded AGC closed-loop surrogate rollout."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from control.controller import ControlPlan, PredictiveControlAdapter


@dataclass
class RolloutSummary:
    """Compact closed-loop benchmark summary."""

    predictor: str
    controller: str
    compartment: str
    reference_mode: str
    steps: int
    objective_mean: float
    control_delta_mae: float
    action_tv: float
    target_mae: Dict[str, float]
    figure_path: str


@dataclass
class RolloutTrace:
    """Per-step rollout traces."""

    timestamps: List[str] = field(default_factory=list)
    predicted_targets: List[List[float]] = field(default_factory=list)
    reference_targets: List[List[float]] = field(default_factory=list)
    executed_actions: List[List[float]] = field(default_factory=list)
    baseline_actions: List[List[float]] = field(default_factory=list)
    objectives: List[float] = field(default_factory=list)


class AGCClosedLoopSimulator:
    """Run a receding-horizon surrogate rollout on one AGC compartment."""

    CONTROL_TO_PAST_MAP = {
        "AssimLight": "assim_sp",
        "EnScr": "scr_enrg_sp",
        "BlackScr": "scr_blck_sp",
        "PipeLow": "t_heat_sp",
        "PipeGrow": "t_heat_sp",
        "co2_dos": "co2_sp",
        "VentLee": "window_pos_lee_sp",
        "Ventwind": "window_pos_lee_sp",
        "Cum_irr": "water_sup_intervals_sp_min",
        "Tot_PAR_Lamps": "assim_sp",
    }

    def __init__(self, adapter: PredictiveControlAdapter, raw_bundle, cfg):
        self.adapter = adapter
        self.raw_bundle = raw_bundle
        self.cfg = cfg
        self.x_cols = adapter.x_cols
        self.u_cols = adapter.u_cols
        self.y_cols = adapter.y_cols
        self.x_index = adapter.x_index
        self.u_index = adapter.u_index
        self.y_index = adapter.y_index

    def _build_stage_reference(self, ref_y_window: np.ndarray) -> np.ndarray:
        if self.cfg.control_reference_mode == "constant":
            return np.asarray(self.cfg.constant_target_values, dtype=np.float32)
        return ref_y_window[0].astype(np.float32)

    def _control_ratio(self, executed: float, baseline: float, u_name: str, x_name: str) -> float:
        u_idx = self.u_index[u_name]
        x_idx = self.x_index[x_name]
        lo = float(self.adapter.u_lower_real[u_idx].item())
        hi = float(self.adapter.u_upper_real[u_idx].item())
        x_lo = float(self.adapter.x_min_real[x_idx].item())
        x_hi = float(self.adapter.x_max_real[x_idx].item())

        if abs(baseline) > 1e-6:
            candidate = executed / baseline
            if np.isfinite(candidate):
                return max(candidate, 0.0)

        unit = 0.0 if hi <= lo else (executed - lo) / (hi - lo)
        unit = float(np.clip(unit, 0.0, 1.0))
        if abs(x_hi - x_lo) <= 1e-6:
            return 1.0
        proxy = x_lo + unit * (x_hi - x_lo)
        return max(proxy / max(x_hi, 1e-6), 0.0)

    def _build_next_row(
        self,
        base_next_row: np.ndarray,
        executed_action: np.ndarray,
        baseline_action: np.ndarray,
        next_targets: np.ndarray,
    ) -> np.ndarray:
        next_row = base_next_row.copy()

        for name, value in zip(self.y_cols, next_targets):
            if name in self.x_index:
                next_row[self.x_index[name]] = float(value)

        for x_name, u_name in self.CONTROL_TO_PAST_MAP.items():
            if x_name not in self.x_index or u_name not in self.u_index:
                continue
            x_idx = self.x_index[x_name]
            u_idx = self.u_index[u_name]
            ratio = self._control_ratio(
                float(executed_action[u_idx]),
                float(baseline_action[u_idx]),
                u_name,
                x_name,
            )
            base_val = float(base_next_row[x_idx])
            if abs(base_val) > 1e-6:
                next_row[x_idx] = max(base_val * ratio, 0.0)
            else:
                x_hi = float(self.adapter.x_max_real[x_idx].item())
                next_row[x_idx] = max(ratio * x_hi, 0.0)

        return next_row

    def _save_figure(self, trace: RolloutTrace, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred = np.asarray(trace.predicted_targets, dtype=np.float32)
        ref = np.asarray(trace.reference_targets, dtype=np.float32)
        act = np.asarray(trace.executed_actions, dtype=np.float32)
        base = np.asarray(trace.baseline_actions, dtype=np.float32)
        steps = np.arange(1, len(pred) + 1)

        fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
        for idx, target in enumerate(self.y_cols):
            axes[idx].plot(steps, ref[:, idx], label="Reference", linewidth=2.0)
            axes[idx].plot(steps, pred[:, idx], label="Closed-loop", linewidth=2.0, linestyle="--")
            axes[idx].set_ylabel(target)
            axes[idx].grid(True, alpha=0.25)
            if idx == 0:
                axes[idx].legend()

        action_delta = np.mean(np.abs(act - base), axis=1)
        axes[4].plot(steps, action_delta, color="tab:red", linewidth=2.0)
        axes[4].set_ylabel("|u - u_log| mean")
        axes[4].set_xlabel("Closed-loop step")
        axes[4].grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def run(self, controller, predictor_name: str) -> RolloutSummary:
        x_test = self.raw_bundle["X_past_test"]
        w_test = self.raw_bundle["W_future_test"]
        u_test = self.raw_bundle["U_future_test"]
        y_test = self.raw_bundle["Y_future_test"]
        t_test = self.raw_bundle["t0_test"]

        available_steps = min(len(x_test), len(w_test), len(u_test), len(y_test)) - self.cfg.control_start_idx - 1
        sim_steps = min(self.cfg.control_eval_steps, available_steps)
        if sim_steps <= 0:
            raise ValueError("Not enough AGC test samples for the requested control rollout.")

        current_x = x_test[self.cfg.control_start_idx].copy()
        controller.reset(initial_action_real=u_test[self.cfg.control_start_idx, 0])

        trace = RolloutTrace()
        stage_errors = []
        action_tv = []
        control_delta = []

        for step in range(sim_steps):
            idx = self.cfg.control_start_idx + step
            base_u_window = u_test[idx]
            ref_y_window = y_test[idx]
            plan: ControlPlan = controller.optimize(current_x, w_test[idx], base_u_window, ref_y_window)

            executed_action = plan.plan_real[0]
            x_scaled = self.adapter.x_real_to_scaled(current_x).unsqueeze(0)
            w_scaled = self.adapter.w_real_to_scaled(w_test[idx]).unsqueeze(0)
            u_scaled = self.adapter.u_real_to_scaled(
                self.adapter._to_tensor(plan.plan_real)
            ).unsqueeze(0)

            with torch.no_grad():
                pred_scaled = self.adapter.predict_scaled(x_scaled, w_scaled, u_scaled)
            next_targets = self.adapter.y_scaled_to_real(pred_scaled[:, 0]).detach().cpu().numpy()[0]

            base_next_row = x_test[idx + 1, -1].copy()
            current_x = np.concatenate(
                [current_x[1:], self._build_next_row(base_next_row, executed_action, base_u_window[0], next_targets)[None, :]],
                axis=0,
            )

            stage_ref = self._build_stage_reference(ref_y_window)
            stage_errors.append(np.abs(next_targets - stage_ref))
            control_delta.append(np.mean(np.abs(executed_action - base_u_window[0])))
            if len(trace.executed_actions) > 0:
                action_tv.append(np.mean(np.abs(executed_action - np.asarray(trace.executed_actions[-1], dtype=np.float32))))

            trace.timestamps.append(str(t_test[idx]))
            trace.predicted_targets.append(next_targets.tolist())
            trace.reference_targets.append(stage_ref.tolist())
            trace.executed_actions.append(executed_action.tolist())
            trace.baseline_actions.append(base_u_window[0].tolist())
            trace.objectives.append(float(plan.objective))

        target_mae = np.mean(np.asarray(stage_errors, dtype=np.float32), axis=0)
        figure_path = self._save_figure(
            trace,
            Path(self.cfg.control_figures_dir) / f"{predictor_name}_{controller.name}_closed_loop.png",
        )

        summary = RolloutSummary(
            predictor=predictor_name,
            controller=controller.name,
            compartment=self.cfg.control_compartment,
            reference_mode=self.cfg.control_reference_mode,
            steps=sim_steps,
            objective_mean=float(np.mean(trace.objectives)),
            control_delta_mae=float(np.mean(control_delta)) if control_delta else 0.0,
            action_tv=float(np.mean(action_tv)) if action_tv else 0.0,
            target_mae={name: float(target_mae[idx]) for idx, name in enumerate(self.y_cols)},
            figure_path=str(figure_path),
        )

        summary_path = Path(self.cfg.control_results_dir) / f"{predictor_name}_{controller.name}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
