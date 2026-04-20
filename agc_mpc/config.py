# -*- coding: utf-8 -*-
"""AGC 2019 predictive control project configuration."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AGCConfig:
    """Global configuration for the AGC-based predictive control project."""

    seed: int = 42

    # ------------------------ Paths ------------------------
    data_root: str = "../AutonomousGreenhouseChallenge_edition2"
    results_dir: str = "results"
    forecast_results_dir: str = "results/forecasting"
    forecast_checkpoints_dir: str = "results/forecasting/checkpoints"
    forecast_figures_dir: str = "results/forecasting/figures"
    forecast_analysis_dir: str = "results/forecasting/analysis"

    # ------------------------ Compartments ------------------------
    selected_compartments: List[str] = field(default_factory=lambda: [
        "AICU",
        "Automatoes",
        "Digilog",
        "IUACAAS",
        "Reference",
        "TheAutomators",
    ])

    # ------------------------ Temporal setup ------------------------
    freq: str = "5min"
    seq_len: int = 288          # 24 hours of history at 5-minute resolution
    horizon: int = 24           # 2 hours of future prediction
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    # ------------------------ Optimization ------------------------
    batch_size: int = 256
    learning_rate: float = 1e-3
    num_epochs: int = 12
    early_stop_patience: int = 4
    lambda_trend: float = 0.1
    lambda_auxiliary: float = 0.0
    model_save_path: str = "results/forecasting/checkpoints/best_gru_baseline.pt"

    # ------------------------ Model ------------------------
    hidden_dim: int = 96
    num_layers: int = 2
    dropout: float = 0.1
    seg_len: int = 12
    transformer_heads: int = 4
    transformer_ff_dim: int = 192

    # ------------------------ Evaluation / plotting ------------------------
    plot_examples: int = 3
    plot_history_steps: int = 96
    forecast_rollout_examples: int = 2
    forecast_rollout_steps: int = 96
    forecast_rollout_stride: int = 6
    control_results_dir: str = "results/control"
    control_summaries_dir: str = "results/control/summaries"
    control_figures_dir: str = "results/control/figures"

    # ------------------------ Control benchmark ------------------------
    control_compartment: str = "Reference"
    control_reference_mode: str = "trajectory"   # "trajectory" or "constant"
    control_start_idx: int = 0
    control_eval_steps: int = 96
    control_horizon: int = 6
    control_rollout_mode: str = "surrogate"      # "surrogate" or "semi_grounded"
    control_warm_start_mix: float = 0.70
    control_state_blend: float = 0.70
    control_min_quantile: float = 0.01
    control_max_quantile: float = 0.99

    # Gradient-based MPC solver
    dpc_iterations: int = 30
    dpc_lr: float = 0.08

    # Sampling-based MPC solver (CEM)
    mpc_population: int = 96
    mpc_elites: int = 12
    mpc_iterations: int = 6
    mpc_init_std: float = 0.12
    mpc_min_std: float = 0.02
    mpc_max_std: float = 0.25
    mpc_momentum: float = 0.60

    # Closed-loop objective
    track_weights: List[float] = field(default_factory=lambda: [1.5, 1.0, 1.0, 0.75])
    horizon_decay: float = 0.97
    control_effort_weight: float = 0.05
    control_deviation_weight: float = 0.25
    control_smoothness_weight: float = 0.10
    constant_target_values: List[float] = field(default_factory=lambda: [21.0, 70.0, 800.0, 150.0])

    # ------------------------ Modeling targets ------------------------
    target_cols: List[str] = field(default_factory=lambda: [
        "Tair",
        "Rhair",
        "CO2air",
        "Tot_PAR",
    ])

    # Historical observed states fed into the encoder
    past_state_cols: List[str] = field(default_factory=lambda: [
        "Tair",
        "Rhair",
        "CO2air",
        "HumDef",
        "Tot_PAR",
        "Tot_PAR_Lamps",
        "VentLee",
        "Ventwind",
        "PipeLow",
        "PipeGrow",
        "AssimLight",
        "EnScr",
        "BlackScr",
        "co2_dos",
        "Cum_irr",
    ])

    # Known future exogenous variables
    future_weather_cols: List[str] = field(default_factory=lambda: [
        "Tout",
        "Rhout",
        "AbsHumOut",
        "Iglob",
        "PARout",
        "Rain",
        "Windsp",
        "Winddir",
    ])

    # Known future time features
    future_time_cols: List[str] = field(default_factory=lambda: [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ])

    # Future requested control inputs
    future_control_cols: List[str] = field(default_factory=lambda: [
        "t_heat_sp",
        "t_vent_sp",
        "co2_sp",
        "dx_sp",
        "assim_sp",
        "scr_enrg_sp",
        "scr_blck_sp",
        "window_pos_lee_sp",
        "water_sup_intervals_sp_min",
    ])

    # Fallback map: use VIP values when requested setpoints are missing
    sp_vip_fallbacks: Dict[str, str] = field(default_factory=lambda: {
        "co2_sp": "co2_vip",
        "dx_sp": "dx_vip",
        "t_rail_min_sp": "t_rail_min_vip",
        "t_grow_min_sp": "t_grow_min_vip",
        "assim_sp": "assim_vip",
        "scr_enrg_sp": "scr_enrg_vip",
        "scr_blck_sp": "scr_blck_vip",
        "t_heat_sp": "t_heat_vip",
        "t_vent_sp": "t_ventlee_vip",
        "window_pos_lee_sp": "window_pos_lee_vip",
        "water_sup_intervals_sp_min": "water_sup_intervals_vip_min",
        "int_blue_sp": "int_blue_vip",
        "int_red_sp": "int_red_vip",
        "int_farred_sp": "int_farred_vip",
        "int_white_sp": "int_white_vip",
    })

    # Column aliases present in different dataset versions / docs
    column_aliases: Dict[str, str] = field(default_factory=lambda: {
        "time": "%time",
        "%Time": "%time",
        "%Time ": "%time",
        "Time": "%time",
        "Time ": "%time",
        "Water_sup": "water_sup",
        "Assim_sp": "assim_sp",
        "Assim_vip": "assim_vip",
        "water_sup_int_sp_min": "water_sup_intervals_sp_min",
        "water_sup_int_vip_min": "water_sup_intervals_vip_min",
    })
