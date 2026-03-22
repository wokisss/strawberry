# -*- coding: utf-8 -*-
"""Gymnasium wrapper around the greenhouse physics environment."""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class GreenhouseGymEnv(gym.Env):
    """Offline training environment used by the SAC controller."""

    OBS_DIM = 15

    def __init__(self, physics_env, config, scaler, target_indices):
        super().__init__()
        self.physics_env = physics_env
        self.config = config
        self.scaler = scaler
        self.target_indices = target_indices

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = 300
        self._prev_state = np.zeros(3, dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._reward_scale = (
            abs(10.0 / 25.0) + abs(30.0 / 70.0) + abs(500.0 / 800.0)
        )

        self.np_random = np.random.default_rng(getattr(config, "seed", None))
        self._generate_random_weather()

    def _generate_random_weather(self):
        self.weather_profile = np.zeros((self.max_steps, 4), dtype=np.float32)

        base_t = self.np_random.uniform(5, 30)
        base_h = self.np_random.uniform(30, 90)
        is_day = self.np_random.random() > 0.3
        peak_solar = self.np_random.uniform(100, 700) if is_day else 0.0

        for t in range(self.max_steps):
            base_t += self.np_random.normal(0, 0.15)
            base_t += 0.01 * np.sin(2 * np.pi * t / self.max_steps)
            base_t = np.clip(base_t, 0, 40)

            base_h += self.np_random.normal(0, 0.8)
            base_h = np.clip(base_h, 15, 98)

            if is_day:
                solar = peak_solar * max(0, np.sin(np.pi * t / self.max_steps))
                solar += self.np_random.normal(0, 20)
                solar = np.clip(solar, 0, 900)
            else:
                solar = 0.0

            self.weather_profile[t] = [base_t, base_h, 400.0, solar]

    def _build_obs(self, state, weather, t_idx):
        delta = state - self._prev_state
        out_temp = weather[0]
        out_hum = weather[1]
        out_solar = weather[3]

        horizon = self.config.horizon
        end_idx = min(t_idx + horizon, self.max_steps)
        future_weather = self.weather_profile[t_idx:end_idx]
        if len(future_weather) > 0:
            future_mean = np.mean(future_weather, axis=0)
        else:
            future_mean = weather

        obs = np.concatenate(
            [
                state,
                delta,
                self._prev_action,
                [out_temp, out_hum, out_solar],
                [future_mean[0], future_mean[3]],
            ]
        )
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        self._generate_random_weather()

        init_state = np.array(
            [
                self.np_random.uniform(10.0, 35.0),
                self.np_random.uniform(30.0, 90.0),
                self.np_random.uniform(400.0, 1500.0),
            ],
            dtype=np.float32,
        )

        self.physics_env.reset(init_state)
        self._prev_state = init_state.copy()
        self._prev_action = np.zeros(4, dtype=np.float32)

        weather = self.weather_profile[0]
        obs = self._build_obs(init_state, weather, t_idx=0)
        return obs, {}

    def step(self, action):
        weather = self.weather_profile[self.current_step]
        out_temp, out_hum, out_co2, solar = weather

        next_state_real = self.physics_env.step(action, out_temp, out_hum, out_co2, solar)
        if isinstance(next_state_real, torch.Tensor):
            next_state_real = next_state_real.cpu().numpy()

        next_weather = self.weather_profile[min(self.current_step + 1, self.max_steps - 1)]
        obs = self._build_obs(next_state_real, next_weather, t_idx=self.current_step + 1)

        rel_error_temp = abs(next_state_real[0] - self.config.target_temp) / max(
            self.config.target_temp, 1.0
        )
        rel_error_hum = abs(next_state_real[1] - self.config.target_hum) / max(
            self.config.target_hum, 1.0
        )
        rel_error_co2 = abs(next_state_real[2] - self.config.target_co2) / max(
            self.config.target_co2, 1.0
        )
        track_penalty = (rel_error_temp + rel_error_hum + rel_error_co2) / self._reward_scale
        reward = -track_penalty

        self._prev_state = next_state_real.copy()
        self._prev_action = np.array(action, dtype=np.float32)
        self.current_step += 1

        terminated = False
        truncated = bool(self.current_step >= self.max_steps)
        return obs, reward, terminated, truncated, {}
