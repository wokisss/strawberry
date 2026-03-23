# -*- coding: utf-8 -*-
"""Controllers and model adapters for AGC closed-loop predictive control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class ControlPlan:
    """One receding-horizon control plan."""

    plan_real: np.ndarray
    objective: float


class PredictiveControlAdapter:
    """Wrap a forecasting model with scaling and control-bound helpers."""

    def __init__(self, model, scalers, feature_groups: Dict[str, List[str]], cfg, raw_bundle, device):
        self.model = model.to(device)
        self.model.eval()
        self.scalers = scalers
        self.feature_groups = feature_groups
        self.cfg = cfg
        self.device = device

        self.x_cols = feature_groups["x_past"]
        self.w_cols = feature_groups["w_future"]
        self.u_cols = feature_groups["u_future"]
        self.y_cols = feature_groups["y_future"]

        self.x_index = {name: idx for idx, name in enumerate(self.x_cols)}
        self.u_index = {name: idx for idx, name in enumerate(self.u_cols)}
        self.y_index = {name: idx for idx, name in enumerate(self.y_cols)}

        x_train = raw_bundle["X_past_train"].reshape(-1, len(self.x_cols))
        u_train = raw_bundle["U_future_train"].reshape(-1, len(self.u_cols))

        self.x_min_real = torch.tensor(np.nanmin(x_train, axis=0), dtype=torch.float32, device=device)
        self.x_max_real = torch.tensor(np.nanmax(x_train, axis=0), dtype=torch.float32, device=device)
        self.u_lower_real = torch.tensor(
            np.nanquantile(u_train, cfg.control_min_quantile, axis=0),
            dtype=torch.float32,
            device=device,
        )
        self.u_upper_real = torch.tensor(
            np.nanquantile(u_train, cfg.control_max_quantile, axis=0),
            dtype=torch.float32,
            device=device,
        )
        self.u_upper_real = torch.maximum(self.u_upper_real, self.u_lower_real + 1e-3)

        self.x_mean = torch.tensor(self.scalers["x"].mean_, dtype=torch.float32, device=device)
        self.x_scale = torch.tensor(self.scalers["x"].scale_, dtype=torch.float32, device=device)
        self.w_mean = torch.tensor(self.scalers["w"].mean_, dtype=torch.float32, device=device)
        self.w_scale = torch.tensor(self.scalers["w"].scale_, dtype=torch.float32, device=device)
        self.u_mean = torch.tensor(self.scalers["u"].mean_, dtype=torch.float32, device=device)
        self.u_scale = torch.tensor(self.scalers["u"].scale_, dtype=torch.float32, device=device)
        self.y_mean = torch.tensor(self.scalers["y"].mean_, dtype=torch.float32, device=device)
        self.y_scale = torch.tensor(self.scalers["y"].scale_, dtype=torch.float32, device=device)

        self.track_weights = torch.tensor(cfg.track_weights, dtype=torch.float32, device=device).view(1, 1, -1)
        horizon_weights = np.power(cfg.horizon_decay, np.arange(cfg.horizon, dtype=np.float32))
        self.horizon_weights = torch.tensor(horizon_weights, dtype=torch.float32, device=device).view(1, -1, 1)
        constant_target = np.asarray(cfg.constant_target_values, dtype=np.float32)
        self.constant_target_real = torch.tensor(constant_target, dtype=torch.float32, device=device).view(1, 1, -1)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def x_real_to_scaled(self, x_real: np.ndarray) -> torch.Tensor:
        x = self._to_tensor(x_real)
        return (x - self.x_mean) / self.x_scale

    def w_real_to_scaled(self, w_real: np.ndarray) -> torch.Tensor:
        w = self._to_tensor(w_real)
        return (w - self.w_mean) / self.w_scale

    def y_real_to_scaled(self, y_real: np.ndarray) -> torch.Tensor:
        y = self._to_tensor(y_real)
        return (y - self.y_mean) / self.y_scale

    def y_scaled_to_real(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * self.y_scale + self.y_mean

    def u_real_to_scaled(self, u_real: torch.Tensor) -> torch.Tensor:
        return (u_real - self.u_mean) / self.u_scale

    def u_real_to_unit(self, u_real: np.ndarray) -> torch.Tensor:
        real = self._to_tensor(u_real)
        unit = (real - self.u_lower_real) / (self.u_upper_real - self.u_lower_real)
        return torch.clamp(unit, 0.0, 1.0)

    def u_unit_to_real(self, u_unit: torch.Tensor) -> torch.Tensor:
        return self.u_lower_real + u_unit * (self.u_upper_real - self.u_lower_real)

    def u_unit_to_scaled(self, u_unit: torch.Tensor) -> torch.Tensor:
        u_real = self.u_unit_to_real(u_unit)
        return self.u_real_to_scaled(u_real)

    def expand_control_plan(self, short_plan_unit: torch.Tensor) -> torch.Tensor:
        if short_plan_unit.size(1) == self.cfg.horizon:
            return short_plan_unit
        pad_steps = self.cfg.horizon - short_plan_unit.size(1)
        tail = short_plan_unit[:, -1:, :].expand(-1, pad_steps, -1)
        return torch.cat([short_plan_unit, tail], dim=1)

    def predict_scaled(self, x_scaled: torch.Tensor, w_scaled: torch.Tensor, u_scaled: torch.Tensor) -> torch.Tensor:
        return self.model(x_scaled, w_scaled, u_scaled)

    def build_reference_scaled(self, ref_y_real: np.ndarray) -> torch.Tensor:
        if self.cfg.control_reference_mode == "constant":
            constant = self.constant_target_real.expand(1, self.cfg.horizon, -1)
            return (constant - self.y_mean.view(1, 1, -1)) / self.y_scale.view(1, 1, -1)
        return self.y_real_to_scaled(ref_y_real).unsqueeze(0)

    def control_cost(
        self,
        pred_scaled: torch.Tensor,
        ref_scaled: torch.Tensor,
        plan_unit: torch.Tensor,
        baseline_unit: torch.Tensor,
        last_action_unit: torch.Tensor,
    ) -> torch.Tensor:
        track_error = (pred_scaled - ref_scaled) ** 2
        track_cost = (track_error * self.track_weights * self.horizon_weights).mean(dim=(1, 2))
        effort_cost = (plan_unit ** 2).mean(dim=(1, 2))
        deviation_cost = ((plan_unit - baseline_unit) ** 2).mean(dim=(1, 2))

        if plan_unit.size(1) > 1:
            smooth_delta = (plan_unit[:, 1:] - plan_unit[:, :-1]) ** 2
            smooth_cost = smooth_delta.mean(dim=(1, 2))
        else:
            smooth_cost = torch.zeros(pred_scaled.size(0), dtype=torch.float32, device=self.device)
        smooth_cost = smooth_cost + ((plan_unit[:, 0] - last_action_unit) ** 2).mean(dim=1)

        return (
            track_cost
            + self.cfg.control_effort_weight * effort_cost
            + self.cfg.control_deviation_weight * deviation_cost
            + self.cfg.control_smoothness_weight * smooth_cost
        )


class BaseController:
    """Shared receding-horizon controller state."""

    def __init__(self, name: str, adapter: PredictiveControlAdapter, cfg):
        self.name = name
        self.adapter = adapter
        self.cfg = cfg
        self.last_action_unit = torch.zeros(len(adapter.u_cols), dtype=torch.float32, device=adapter.device)
        self.last_short_plan_unit: Optional[torch.Tensor] = None

    def reset(self, initial_action_real: Optional[np.ndarray] = None) -> None:
        self.last_short_plan_unit = None
        if initial_action_real is None:
            self.last_action_unit = torch.zeros(len(self.adapter.u_cols), dtype=torch.float32, device=self.adapter.device)
            return
        self.last_action_unit = self.adapter.u_real_to_unit(initial_action_real)

    def _baseline_short_plan_unit(self, baseline_u_real: np.ndarray) -> torch.Tensor:
        return self.adapter.u_real_to_unit(baseline_u_real[: self.cfg.control_horizon]).unsqueeze(0)

    def _warm_start_short_plan_unit(self, baseline_u_real: np.ndarray) -> torch.Tensor:
        baseline_short = self._baseline_short_plan_unit(baseline_u_real)
        if self.last_short_plan_unit is None:
            return baseline_short

        shifted = torch.cat([self.last_short_plan_unit[:, 1:], self.last_short_plan_unit[:, -1:]], dim=1)
        mix = float(np.clip(self.cfg.control_warm_start_mix, 0.0, 1.0))
        return torch.clamp(mix * shifted + (1.0 - mix) * baseline_short, 0.0, 1.0)

    def optimize(
        self,
        current_x_real: np.ndarray,
        w_future_real: np.ndarray,
        baseline_u_real: np.ndarray,
        ref_y_real: np.ndarray,
    ) -> ControlPlan:
        raise NotImplementedError


class RecordedBaselineController(BaseController):
    """Execute the logged AGC setpoint plan without optimization."""

    def __init__(self, adapter: PredictiveControlAdapter, cfg):
        super().__init__("recorded", adapter, cfg)

    def optimize(self, current_x_real, w_future_real, baseline_u_real, ref_y_real) -> ControlPlan:
        plan_real = baseline_u_real.copy()
        self.last_action_unit = self.adapter.u_real_to_unit(plan_real[0])
        return ControlPlan(plan_real=plan_real, objective=0.0)


class GradientMPCController(BaseController):
    """Gradient-based MPC solver on top of a differentiable surrogate."""

    def __init__(self, adapter: PredictiveControlAdapter, cfg):
        super().__init__("gradient_mpc", adapter, cfg)

    def optimize(self, current_x_real, w_future_real, baseline_u_real, ref_y_real) -> ControlPlan:
        x_scaled = self.adapter.x_real_to_scaled(current_x_real).unsqueeze(0)
        w_scaled = self.adapter.w_real_to_scaled(w_future_real).unsqueeze(0)
        ref_scaled = self.adapter.build_reference_scaled(ref_y_real)

        baseline_short_unit = self._baseline_short_plan_unit(baseline_u_real)
        init_short_unit = self._warm_start_short_plan_unit(baseline_u_real)
        logits = torch.logit(torch.clamp(init_short_unit, 1e-4, 1.0 - 1e-4)).detach().clone()
        logits.requires_grad_(True)
        optimizer = torch.optim.Adam([logits], lr=self.cfg.dpc_lr)

        best_cost = float("inf")
        best_plan_unit = None
        best_short_plan_unit = None

        for _ in range(self.cfg.dpc_iterations):
            optimizer.zero_grad()
            short_plan_unit = torch.sigmoid(logits)
            plan_unit = self.adapter.expand_control_plan(short_plan_unit)
            u_scaled = self.adapter.u_unit_to_scaled(plan_unit)
            pred_scaled = self.adapter.predict_scaled(x_scaled, w_scaled, u_scaled)
            cost = self.adapter.control_cost(
                pred_scaled,
                ref_scaled,
                plan_unit,
                self.adapter.expand_control_plan(baseline_short_unit),
                self.last_action_unit.unsqueeze(0),
            ).mean()
            cost.backward()
            optimizer.step()

            cost_value = float(cost.detach().item())
            if cost_value < best_cost:
                best_cost = cost_value
                best_plan_unit = plan_unit.detach()
                best_short_plan_unit = short_plan_unit.detach()

        if best_plan_unit is None:
            best_short_plan_unit = torch.sigmoid(logits).detach()
            best_plan_unit = self.adapter.expand_control_plan(best_short_plan_unit)

        best_plan_real = self.adapter.u_unit_to_real(best_plan_unit).squeeze(0).detach().cpu().numpy()
        self.last_action_unit = best_plan_unit[0, 0].detach()
        self.last_short_plan_unit = best_short_plan_unit
        return ControlPlan(plan_real=best_plan_real, objective=best_cost)


class CEMMPCController(BaseController):
    """Sampling-based MPC using the Cross-Entropy Method over future setpoint plans."""

    def __init__(self, adapter: PredictiveControlAdapter, cfg):
        super().__init__("cem_mpc", adapter, cfg)
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(cfg.seed))

    def reset(self, initial_action_real: Optional[np.ndarray] = None) -> None:
        super().reset(initial_action_real=initial_action_real)
        self.rng.manual_seed(int(self.cfg.seed))

    def optimize(self, current_x_real, w_future_real, baseline_u_real, ref_y_real) -> ControlPlan:
        x_scaled = self.adapter.x_real_to_scaled(current_x_real).unsqueeze(0)
        w_scaled = self.adapter.w_real_to_scaled(w_future_real).unsqueeze(0)
        ref_scaled = self.adapter.build_reference_scaled(ref_y_real)

        baseline_short_unit = self._baseline_short_plan_unit(baseline_u_real)
        init_short_unit = self._warm_start_short_plan_unit(baseline_u_real)
        mean = init_short_unit.clone()
        std = torch.full_like(mean, self.cfg.mpc_init_std)

        best_cost = float("inf")
        best_short_plan = init_short_unit.clone()
        baseline_plan = self.adapter.expand_control_plan(baseline_short_unit)

        for _ in range(self.cfg.mpc_iterations):
            samples = mean.cpu() + std.cpu() * torch.randn(
                self.cfg.mpc_population,
                self.cfg.control_horizon,
                len(self.adapter.u_cols),
                generator=self.rng,
            )
            samples = torch.clamp(samples, 0.0, 1.0).to(self.adapter.device)
            if self.cfg.mpc_population >= 1:
                samples[0] = baseline_short_unit[0]
            if self.cfg.mpc_population >= 2:
                samples[1] = init_short_unit[0]
            if self.cfg.mpc_population >= 3:
                samples[2] = best_short_plan[0]
            plans = self.adapter.expand_control_plan(samples)

            with torch.no_grad():
                pred_scaled = self.adapter.predict_scaled(
                    x_scaled.expand(self.cfg.mpc_population, -1, -1),
                    w_scaled.expand(self.cfg.mpc_population, -1, -1),
                    self.adapter.u_unit_to_scaled(plans),
                )
                costs = self.adapter.control_cost(
                    pred_scaled,
                    ref_scaled.expand(self.cfg.mpc_population, -1, -1),
                    plans,
                    baseline_plan.expand(self.cfg.mpc_population, -1, -1),
                    self.last_action_unit.unsqueeze(0).expand(self.cfg.mpc_population, -1),
                )

            elite_idx = torch.topk(-costs, self.cfg.mpc_elites).indices
            elites = samples[elite_idx]
            elite_costs = costs[elite_idx]
            elite_mean = elites.mean(dim=0, keepdim=True)
            elite_std = elites.std(dim=0, unbiased=False, keepdim=True)
            momentum = float(np.clip(self.cfg.mpc_momentum, 0.0, 1.0))
            mean = torch.clamp(momentum * mean + (1.0 - momentum) * elite_mean, 0.0, 1.0)
            std = torch.clamp(
                momentum * std + (1.0 - momentum) * elite_std,
                min=self.cfg.mpc_min_std,
                max=self.cfg.mpc_max_std,
            )

            elite_best = int(torch.argmin(elite_costs).item())
            elite_best_cost = float(elite_costs[elite_best].item())
            if elite_best_cost < best_cost:
                best_cost = elite_best_cost
                best_short_plan = elites[elite_best : elite_best + 1].detach()

        candidate_shorts = torch.cat([best_short_plan, mean, baseline_short_unit], dim=0)
        candidate_plans = self.adapter.expand_control_plan(candidate_shorts)
        with torch.no_grad():
            candidate_costs = self.adapter.control_cost(
                self.adapter.predict_scaled(
                    x_scaled.expand(candidate_shorts.size(0), -1, -1),
                    w_scaled.expand(candidate_shorts.size(0), -1, -1),
                    self.adapter.u_unit_to_scaled(candidate_plans),
                ),
                ref_scaled.expand(candidate_shorts.size(0), -1, -1),
                candidate_plans,
                baseline_plan.expand(candidate_shorts.size(0), -1, -1),
                self.last_action_unit.unsqueeze(0).expand(candidate_shorts.size(0), -1),
            )
        best_idx = int(torch.argmin(candidate_costs).item())
        best_cost = float(candidate_costs[best_idx].item())
        best_short_plan = candidate_shorts[best_idx : best_idx + 1].detach()
        best_plan_unit = self.adapter.expand_control_plan(best_short_plan)
        best_plan_real = self.adapter.u_unit_to_real(best_plan_unit).squeeze(0).detach().cpu().numpy()
        self.last_action_unit = best_plan_unit[0, 0].detach()
        self.last_short_plan_unit = best_short_plan
        return ControlPlan(plan_real=best_plan_real, objective=best_cost)


# Backward-compatible aliases. These refer to MPC solvers, not separate control paradigms.
GradientDPCController = GradientMPCController
SamplingMPCController = CEMMPCController
