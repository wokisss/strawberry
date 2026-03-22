# -*- coding: utf-8 -*-
"""Closed-loop simulation for DPC vs SAC."""

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class SimResult:
    history_dpc: List[np.ndarray] = field(default_factory=list)
    history_sac: List[np.ndarray] = field(default_factory=list)
    actions_dpc: List[list] = field(default_factory=list)
    actions_sac: List[list] = field(default_factory=list)
    pwm_actions_dpc: List[list] = field(default_factory=list)
    pwm_actions_sac: List[list] = field(default_factory=list)
    time_dpc: List[float] = field(default_factory=list)
    time_sac: List[float] = field(default_factory=list)
    targets: List[np.ndarray] = field(default_factory=list)
    target_temp: float = 25.0
    target_hum: float = 70.0
    target_co2: float = 800.0
    sim_steps: int = 0
    pwm_enabled: bool = False


class Simulator:
    def __init__(
        self,
        dpc,
        sac,
        env_class,
        feature_order,
        scaler,
        target_indices,
        config=None,
        device=None,
        pwm_sim_dpc=None,
        pwm_sim_sac=None,
    ):
        self.dpc = dpc
        self.sac = sac
        self.env_class = env_class
        self.feature_order = feature_order
        self.scaler = scaler
        self.target_indices = target_indices
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pwm_sim_dpc = pwm_sim_dpc
        self.pwm_sim_sac = pwm_sim_sac
        self.config = config

        if config is not None:
            self._sim_steps = config.sim_steps
            self._start_idx = config.sim_start_idx
            self._target_temp = config.target_temp
            self._target_hum = config.target_hum
            self._target_co2 = config.target_co2
            self._seq_len = config.seq_len
            self._control_cols = list(config.control_cols)
        else:
            self._sim_steps = 300
            self._start_idx = 0
            self._target_temp = 25.0
            self._target_hum = 70.0
            self._target_co2 = 800.0
            self._seq_len = 60
            self._control_cols = ["Heater", "Ventilation", "Fog", "Lighting"]

        self._feature_to_idx = {name: idx for idx, name in enumerate(feature_order)}
        self._control_indices = [
            self._feature_to_idx[col] for col in self._control_cols if col in self._feature_to_idx
        ]
        self._future_feature_indices = getattr(self.dpc, "future_indices", [])
        self._future_idx_to_pos = {
            feature_idx: pos for pos, feature_idx in enumerate(self._future_feature_indices)
        }

    def _inverse_transform(self, val, col_idx):
        return (val - self.scaler.min_[col_idx]) / self.scaler.scale_[col_idx]

    def _forward_transform(self, val, col_idx):
        return val * self.scaler.scale_[col_idx] + self.scaler.min_[col_idx]

    def _apply_pwm(self, pwm_sim, action):
        if pwm_sim is None:
            return action, None

        pwm_sim.set_duty("heater", action[0])
        pwm_sim.set_duty("vent", action[1])
        pwm_sim.set_duty("humidifier", action[2])
        pwm_sim.set_duty("co2_gen", action[3])
        relay_states = pwm_sim.step()
        actual = [
            1.0 if relay_states.get("heater", False) else 0.0,
            1.0 if relay_states.get("vent", False) else 0.0,
            1.0 if relay_states.get("humidifier", False) else 0.0,
            1.0 if relay_states.get("co2_gen", False) else 0.0,
        ]
        return actual, actual

    def _get_future_raw_value(self, future_step, feature_name, default):
        feature_idx = self._feature_to_idx.get(feature_name)
        if feature_idx is None:
            return default
        pos = self._future_idx_to_pos.get(feature_idx)
        if pos is None:
            return default
        return self._inverse_transform(float(future_step[pos]), feature_idx)

    def _seed_action_queue(self, past_sequence):
        if not self._control_indices:
            return []
        queue = past_sequence[-60:, self._control_indices]
        return np.asarray(queue, dtype=np.float32).tolist()

    def _reset_controller_state(self, controller):
        if hasattr(controller, "reset"):
            controller.reset()

    def _update_energy_features(self, next_row, action_queue):
        queue_arr = np.asarray(action_queue, dtype=np.float32)
        if queue_arr.size == 0:
            energy_sum = np.zeros(len(self._control_cols), dtype=np.float32)
        else:
            energy_sum = np.sum(queue_arr, axis=0)

        for idx, col in enumerate(self._control_cols):
            energy_col = f"{col}_Energy_60m"
            feature_idx = self._feature_to_idx.get(energy_col)
            if feature_idx is not None:
                next_row[feature_idx] = self._forward_transform(float(energy_sum[idx]), feature_idx)

    def _update_weather_derivatives(self, prev_row, next_row):
        deriv_specs = [
            ("Outdoor_Temp", "Outdoor_Temp_Deriv"),
            ("Outdoor_Solar", "Outdoor_Solar_Deriv"),
        ]

        for base_name, deriv_name in deriv_specs:
            base_idx = self._feature_to_idx.get(base_name)
            deriv_idx = self._feature_to_idx.get(deriv_name)
            if base_idx is None or deriv_idx is None:
                continue

            prev_raw = self._inverse_transform(float(prev_row[base_idx]), base_idx)
            next_raw = self._inverse_transform(float(next_row[base_idx]), base_idx)
            deriv_per_hour = (next_raw - prev_raw) * 60.0
            next_row[deriv_idx] = self._forward_transform(deriv_per_hour, deriv_idx)

    def _step_env(self, env, actual_action, current_state, current_future_base, action_queue):
        prev_row = current_state[0, -1, :].detach().cpu().numpy()
        future_step = current_future_base[0, 0, :].detach().cpu().numpy()

        outside_temp = self._get_future_raw_value(future_step, "Outdoor_Temp", 15.0)
        outside_hum = self._get_future_raw_value(future_step, "Outdoor_Hum", 50.0)
        outside_solar = self._get_future_raw_value(future_step, "Outdoor_Solar", 0.0)

        actual_action_tensor = torch.tensor(actual_action, dtype=torch.float32, device=self.device)
        next_state_real = env.step(
            actual_action_tensor,
            outside_temp,
            outside_hum,
            400.0,
            outside_solar,
        )

        if isinstance(next_state_real, torch.Tensor):
            next_state_real_np = next_state_real.detach().cpu().numpy()
        else:
            next_state_real_np = np.asarray(next_state_real, dtype=np.float32)

        next_row = prev_row.copy()

        for feature_idx, pos in self._future_idx_to_pos.items():
            next_row[feature_idx] = float(future_step[pos])

        for idx, target_idx in enumerate(self.target_indices):
            next_row[target_idx] = self._forward_transform(float(next_state_real_np[idx]), target_idx)

        for idx, feature_idx in enumerate(self._control_indices[: len(actual_action)]):
            next_row[feature_idx] = float(actual_action[idx])

        action_queue.append(list(np.asarray(actual_action[: len(self._control_indices)], dtype=np.float32)))
        if len(action_queue) > 60:
            action_queue.pop(0)

        self._update_energy_features(next_row, action_queue)
        self._update_weather_derivatives(prev_row, next_row)

        next_row_tensor = torch.tensor(next_row, dtype=torch.float32, device=self.device)
        new_state = torch.cat([current_state[:, 1:, :], next_row_tensor.view(1, 1, -1)], dim=1)
        return next_state_real_np, new_state

    def run(self, X_test_p, X_test_f, config=None):
        if len(X_test_p) == 0 or len(X_test_f) == 0:
            raise ValueError("Simulation requires non-empty test sequences.")
        if self._start_idx >= len(X_test_p) or self._start_idx >= len(X_test_f):
            raise IndexError(
                f"Simulation start index {self._start_idx} is outside the test set."
            )

        available_steps = min(len(X_test_p) - self._start_idx, len(X_test_f) - self._start_idx)
        sim_steps = min(self._sim_steps, available_steps)
        if sim_steps <= 0:
            raise ValueError("No simulation steps available from the selected start index.")
        if sim_steps < self._sim_steps:
            print(
                f"---> [Warning] Clipping sim_steps from {self._sim_steps} to {sim_steps} "
                "to fit test data."
            )

        pwm_on = self.pwm_sim_dpc is not None
        result = SimResult(
            target_temp=self._target_temp,
            target_hum=self._target_hum,
            target_co2=self._target_co2,
            sim_steps=sim_steps,
            pwm_enabled=pwm_on,
        )

        current_state_dpc = torch.tensor(
            X_test_p[self._start_idx : self._start_idx + 1], dtype=torch.float32, device=self.device
        )
        current_state_sac = current_state_dpc.clone()
        future_base_seq_tensor = torch.tensor(
            X_test_f[self._start_idx : self._start_idx + sim_steps],
            dtype=torch.float32,
            device=self.device,
        )

        action_queue_dpc = self._seed_action_queue(X_test_p[self._start_idx])
        action_queue_sac = self._seed_action_queue(X_test_p[self._start_idx])

        result.targets = [
            np.array([self._target_temp, self._target_hum, self._target_co2], dtype=np.float32)
            for _ in range(sim_steps)
        ]

        init_state_norm = current_state_dpc[0, -1, self.target_indices].detach().cpu().numpy()
        init_state_np = np.array(
            [
                self._inverse_transform(float(init_state_norm[0]), self.target_indices[0]),
                self._inverse_transform(float(init_state_norm[1]), self.target_indices[1]),
                self._inverse_transform(float(init_state_norm[2]), self.target_indices[2]),
            ],
            dtype=np.float32,
        )

        init_state_tensor = torch.tensor(init_state_np, dtype=torch.float32, device=self.device)
        env_dpc = self.env_class(init_state_tensor, config)
        env_sac = self.env_class(init_state_tensor.clone(), config)

        if self.pwm_sim_dpc is not None:
            self.pwm_sim_dpc.reset()
        if self.pwm_sim_sac is not None:
            self.pwm_sim_sac.reset()

        self._reset_controller_state(self.dpc)
        self._reset_controller_state(self.sac)

        pwm_label = " + PWM" if pwm_on else ""
        print(f"---> Running {sim_steps} closed-loop simulation steps (DPC vs SAC{pwm_label})...")

        for t in range(sim_steps):
            current_future_base = future_base_seq_tensor[t].unsqueeze(0)

            curr_dpc_temp = (
                env_dpc.current_temp.item()
                if isinstance(env_dpc.current_temp, torch.Tensor)
                else env_dpc.current_temp
            )
            t0 = time.time()
            opt_action_dpc, _ = self.dpc.get_optimal_action(
                current_state_dpc, current_future_base, current_temp=curr_dpc_temp
            )
            result.time_dpc.append((time.time() - t0) * 1000)
            result.actions_dpc.append(opt_action_dpc)
            actual_dpc, pwm_dpc = self._apply_pwm(self.pwm_sim_dpc, opt_action_dpc)
            if pwm_dpc is not None:
                result.pwm_actions_dpc.append(pwm_dpc)

            curr_sac_temp = (
                env_sac.current_temp.item()
                if isinstance(env_sac.current_temp, torch.Tensor)
                else env_sac.current_temp
            )
            t0 = time.time()
            opt_action_sac, _ = self.sac.get_optimal_action(
                current_state_sac, current_future_base, current_temp=curr_sac_temp
            )
            result.time_sac.append((time.time() - t0) * 1000)
            result.actions_sac.append(opt_action_sac)
            actual_sac, pwm_sac = self._apply_pwm(self.pwm_sim_sac, opt_action_sac)
            if pwm_sac is not None:
                result.pwm_actions_sac.append(pwm_sac)

            with torch.no_grad():
                next_state_dpc, current_state_dpc = self._step_env(
                    env_dpc, actual_dpc, current_state_dpc, current_future_base, action_queue_dpc
                )
                result.history_dpc.append(next_state_dpc)
                self.dpc.update_integral(next_state_dpc[0])

                next_state_sac, current_state_sac = self._step_env(
                    env_sac, actual_sac, current_state_sac, current_future_base, action_queue_sac
                )
                result.history_sac.append(next_state_sac)
                self.sac.update_integral(next_state_sac[0])

            if (t + 1) % 50 == 0:
                print(
                    f"    Step {t + 1}/{sim_steps} | "
                    f"DPC_T={next_state_dpc[0]:.1f}C (avg {np.mean(result.time_dpc[-50:]):.0f}ms) | "
                    f"SAC_T={next_state_sac[0]:.1f}C (avg {np.mean(result.time_sac[-50:]):.0f}ms)"
                )

        print("---> Simulation complete.")
        return result
