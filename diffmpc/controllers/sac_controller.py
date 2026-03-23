# -*- coding: utf-8 -*-
"""SAC policy wrapper used in simulator rollouts."""

import time

import numpy as np
import torch


class SACController:
    """Wrap a trained SAC agent with the simulator controller interface."""

    def __init__(
        self,
        agent_model,
        scaler,
        target_indices,
        future_indices,
        feature_order=None,
        config=None,
        horizon=10,
        target_temp=25.0,
    ):
        self.agent = agent_model
        self.scaler = scaler
        self.target_indices = target_indices
        self.future_indices = future_indices
        self._device = self.agent.device

        self._feature_order = feature_order or []
        self._out_temp_idx = self._feature_order.index("Outdoor_Temp") if "Outdoor_Temp" in self._feature_order else -1
        self._out_hum_idx = self._feature_order.index("Outdoor_Hum") if "Outdoor_Hum" in self._feature_order else -1
        self._out_solar_idx = self._feature_order.index("Outdoor_Solar") if "Outdoor_Solar" in self._feature_order else -1

        if config is not None:
            self.target_temp = config.target_temp
            self._integral_decay = config.integral_decay
            self._integral_clip = config.integral_clip
            self._safety = {
                "tier1_error": config.safety_tier1_error,
                "tier1_min": config.safety_tier1_min,
                "tier2_error": config.safety_tier2_error,
                "tier2_min": config.safety_tier2_min,
                "tier3_error": config.safety_tier3_error,
                "tier3_min": config.safety_tier3_min,
                "tier4_min": config.safety_tier4_min,
            }
        else:
            self.target_temp = target_temp
            self._integral_decay = 0.95
            self._integral_clip = 20.0
            self._safety = {
                "tier1_error": 3.0,
                "tier1_min": 0.98,
                "tier2_error": 1.0,
                "tier2_min": 0.90,
                "tier3_error": 0.3,
                "tier3_min": 0.80,
                "tier4_min": 0.60,
            }

        self.integral_error = 0.0
        self.last_step_time = 0.0
        self.last_action = [0.0, 0.0, 0.0, 0.0]
        self._prev_state = np.zeros(3, dtype=np.float32)
        self._first_step = True

    def _inverse_transform(self, val, col_idx):
        dummy = np.zeros((1, len(self.scaler.scale_)))
        dummy[0, col_idx] = val
        return self.scaler.inverse_transform(dummy)[0, col_idx]

    def get_optimal_action(self, current_past_tensor, current_future_base, current_temp=None):
        """Build the SAC observation and query the actor deterministically."""
        t_start = time.time()

        last_state_norm = current_past_tensor[0, -1, :].detach().cpu().numpy()
        t_idx, h_idx, c_idx = self.target_indices

        current_state = np.array(
            [
                self._inverse_transform(last_state_norm[t_idx], t_idx),
                self._inverse_transform(last_state_norm[h_idx], h_idx),
                self._inverse_transform(last_state_norm[c_idx], c_idx),
            ],
            dtype=np.float32,
        )

        out_temp = (
            self._inverse_transform(last_state_norm[self._out_temp_idx], self._out_temp_idx)
            if self._out_temp_idx >= 0
            else 15.0
        )
        out_hum = (
            self._inverse_transform(last_state_norm[self._out_hum_idx], self._out_hum_idx)
            if self._out_hum_idx >= 0
            else 50.0
        )
        out_solar = (
            self._inverse_transform(last_state_norm[self._out_solar_idx], self._out_solar_idx)
            if self._out_solar_idx >= 0
            else 0.0
        )

        mean_future_norm = torch.mean(current_future_base[0], dim=0).detach().cpu().numpy()
        idx_future_out_temp = self.future_indices.index(self._out_temp_idx) if self._out_temp_idx in self.future_indices else -1
        idx_future_out_solar = self.future_indices.index(self._out_solar_idx) if self._out_solar_idx in self.future_indices else -1

        future_temp_mean = (
            self._inverse_transform(mean_future_norm[idx_future_out_temp], self._out_temp_idx)
            if idx_future_out_temp >= 0
            else out_temp
        )
        future_solar_mean = (
            self._inverse_transform(mean_future_norm[idx_future_out_solar], self._out_solar_idx)
            if idx_future_out_solar >= 0
            else out_solar
        )

        if self._first_step:
            delta = np.zeros(3, dtype=np.float32)
            self._first_step = False
        else:
            delta = current_state - self._prev_state
        self._prev_state = current_state.copy()

        obs = np.concatenate(
            [
                current_state,
                delta,
                np.array(self.last_action, dtype=np.float32),
                [out_temp, out_hum, out_solar],
                [future_temp_mean, future_solar_mean],
            ]
        ).astype(np.float32)

        action = self.agent.select_action(obs, evaluate=True)

        final_heater = float(action[0])
        final_vent = float(action[1])
        final_fog = float(action[2])
        final_lighting = float(action[3])

        if current_temp is not None:
            error = self.target_temp - current_temp
            s = self._safety
            if error >= s["tier1_error"]:
                final_heater = max(final_heater, s["tier1_min"])
            elif error >= s["tier2_error"]:
                final_heater = max(final_heater, s["tier2_min"])
            elif error >= s["tier3_error"]:
                final_heater = max(final_heater, s["tier3_min"])
            elif error > 0:
                final_heater = max(final_heater, s["tier4_min"])

        if final_heater > 0.1 and final_vent > 0.1:
            if final_heater > final_vent:
                final_vent = 0.0
            else:
                final_heater = 0.0

        best_action = [final_heater, final_vent, final_fog, final_lighting]
        self.last_action = best_action.copy()
        self.last_step_time = time.time() - t_start
        return best_action, 0.0

    def update_integral(self, current_temp):
        error = self.target_temp - current_temp
        self.integral_error *= self._integral_decay
        self.integral_error += error
        self.integral_error = np.clip(
            self.integral_error,
            -self._integral_clip,
            self._integral_clip,
        )

    def reset(self):
        """Reset controller state before a fresh rollout."""
        self.integral_error = 0.0
        self.last_step_time = 0.0
        self.last_action = [0.0, 0.0, 0.0, 0.0]
        self._prev_state = np.zeros(3, dtype=np.float32)
        self._first_step = True
