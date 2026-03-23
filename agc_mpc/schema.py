# -*- coding: utf-8 -*-
"""Schema helpers for AGC 2019."""

from typing import Dict, List

from config import AGCConfig


def build_feature_groups(cfg: AGCConfig) -> Dict[str, List[str]]:
    """Return the feature groups used by the AGC predictive control pipeline."""
    return {
        "x_past": cfg.past_state_cols,
        "w_future": cfg.future_weather_cols + cfg.future_time_cols,
        "u_future": cfg.future_control_cols,
        "y_future": cfg.target_cols,
    }

