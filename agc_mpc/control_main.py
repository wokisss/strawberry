# -*- coding: utf-8 -*-
"""Entry point for AGC closed-loop MPC solver benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from config import AGCConfig
from control.controller import (
    CEMMPCController,
    GradientMPCController,
    PredictiveControlAdapter,
    RecordedBaselineController,
)
from control.simulator import AGCClosedLoopSimulator
from data_processing.processor import AGCDataProcessor
from models.dlinear_forecaster import ConditionalDLinearForecaster
from models.frequency_forecaster import ConditionalFrequencyMLPForecaster
from models.gru_forecaster import ConditionalGRUForecaster
from models.hybrid_residual_forecaster import ConditionalHybridResidualForecaster
from models.itransformer_residual_forecaster import (
    ConditionalITransformerCO2ControlAwareFusionForecaster,
    ConditionalITransformerCO2LateResidualForecaster,
    ConditionalITransformerCO2LateFrozenExpertForecaster,
    ConditionalITransformerCO2FrozenBackboneHorizonMixtureForecaster,
    ConditionalITransformerCO2HorizonMixtureForecaster,
    ConditionalITransformerCO2ProtectedExpertForecaster,
    ConditionalITransformerCO2ProtectedTerminalForecaster,
    ConditionalITransformerCO2RecoupledExpertForecaster,
    ConditionalITransformerCO2FrozenExpertForecaster,
    ConditionalITransformerCO2ResidualForecaster,
    ConditionalITransformerCO2TeacherDistillForecaster,
    ConditionalITransformerCO2WaveletBlendForecaster,
    ConditionalITransformerCO2WaveletResidualForecaster,
    ConditionalITransformerResidualForecaster,
)
from models.lstm_forecaster import ConditionalLSTMForecaster
from models.nlinear_forecaster import ConditionalNLinearForecaster
from models.patchtst_residual_forecaster import ConditionalPatchTSTResidualForecaster
from models.seg_rnn_forecaster import ConditionalSegRNNForecaster
from models.transformer_forecaster import ConditionalTransformerForecaster
from models.transformer_hybrid_forecaster import ConditionalTransformerHybridForecaster
from results_utils import ensure_results_layout


def _set_global_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


LATEST_PREDICTORS = [
    "current_hybrid_transformer",
    "transformer_hybrid_residual",
    "itransformer_residual",
    "patchtst_residual",
]


def _apply_three_target_control_protocol(cfg: AGCConfig) -> AGCConfig:
    cfg.target_cols = ["Tair", "Rhair", "CO2air"]
    cfg.track_weights = cfg.track_weights[:3]
    cfg.constant_target_values = cfg.constant_target_values[:3]
    return cfg


def _build_model_specs(bundle, cfg):
    return {
        "dlinear_baseline": {
            "builder": lambda: ConditionalDLinearForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
            ),
            "checkpoint": "dlinear_baseline.pt",
        },
        "dlinear_forecaster": {
            "builder": lambda: ConditionalDLinearForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
            ),
            "checkpoint": f"dlinear_forecaster_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "frequency_forecaster": {
            "builder": lambda: ConditionalFrequencyMLPForecaster(
                seq_len=cfg.seq_len,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
            ),
            "checkpoint": f"frequency_baseline_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "gru_forecaster": {
            "builder": lambda: ConditionalGRUForecaster(
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            ),
            "checkpoint": f"gru_baseline_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "lstm_forecaster": {
            "builder": lambda: ConditionalLSTMForecaster(
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            ),
            "checkpoint": f"lstm_baseline_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "nlinear_forecaster": {
            "builder": lambda: ConditionalNLinearForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
            ),
            "checkpoint": f"nlinear_baseline_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "segrnn_forecaster": {
            "builder": lambda: ConditionalSegRNNForecaster(
                seq_len=cfg.seq_len,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                seg_len=cfg.seg_len,
            ),
            "checkpoint": f"segrnn_baseline_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "transformer_forecaster": {
            "builder": lambda: ConditionalTransformerForecaster(
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
                max_past_len=cfg.seq_len,
                max_future_len=cfg.horizon,
            ),
            "checkpoint": f"transformer_baseline_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "transformer_hybrid_baseline": {
            "builder": lambda: ConditionalTransformerHybridForecaster(
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
                max_past_len=cfg.seq_len,
                max_future_len=cfg.horizon,
            ),
            "checkpoint": "transformer_hybrid_baseline.pt",
        },
        "transformer_baseline": {
            "builder": lambda: ConditionalTransformerForecaster(
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
                max_past_len=cfg.seq_len,
                max_future_len=cfg.horizon,
            ),
            "checkpoint": "transformer_baseline.pt",
        },
        "current_hybrid_transformer": {
            "builder": lambda: ConditionalTransformerHybridForecaster(
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
                max_past_len=cfg.seq_len,
                max_future_len=cfg.horizon,
            ),
            "checkpoint": f"current_hybrid_transformer_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "transformer_hybrid_residual": {
            "builder": lambda: ConditionalHybridResidualForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
                max_past_len=cfg.seq_len,
                max_future_len=cfg.horizon,
            ),
            "checkpoint": f"transformer_hybrid_residual_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_residual": {
            "builder": lambda: ConditionalITransformerResidualForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_residual_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_residual": {
            "builder": lambda: ConditionalITransformerCO2ResidualForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_residual_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_late_residual": {
            "builder": lambda: ConditionalITransformerCO2LateResidualForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_late_residual_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_frozen_expert": {
            "builder": lambda: ConditionalITransformerCO2FrozenExpertForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_frozen_expert_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_late_frozen_expert": {
            "builder": lambda: ConditionalITransformerCO2LateFrozenExpertForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_late_frozen_expert_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_teacher_distill": {
            "builder": lambda: ConditionalITransformerCO2TeacherDistillForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_teacher_distill_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_recoupled_expert": {
            "builder": lambda: ConditionalITransformerCO2RecoupledExpertForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_recoupled_expert_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_protected_expert": {
            "builder": lambda: ConditionalITransformerCO2ProtectedExpertForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_protected_expert_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_protected_terminal": {
            "builder": lambda: ConditionalITransformerCO2ProtectedTerminalForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_protected_terminal_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_horizon_mixture": {
            "builder": lambda: ConditionalITransformerCO2HorizonMixtureForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_horizon_mixture_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_frozen_backbone_horizon_mixture": {
            "builder": lambda: ConditionalITransformerCO2FrozenBackboneHorizonMixtureForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_frozen_backbone_horizon_mixture_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_control_aware_fusion": {
            "builder": lambda: ConditionalITransformerCO2ControlAwareFusionForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_control_aware_fusion_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_wavelet_residual": {
            "builder": lambda: ConditionalITransformerCO2WaveletResidualForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_wavelet_residual_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "itransformer_co2_wavelet_blend": {
            "builder": lambda: ConditionalITransformerCO2WaveletBlendForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"itransformer_co2_wavelet_blend_joint_all_{cfg.control_compartment.lower()}.pt",
        },
        "patchtst_residual": {
            "builder": lambda: ConditionalPatchTSTResidualForecaster(
                seq_len=cfg.seq_len,
                horizon=cfg.horizon,
                past_dim=bundle["X_past_test"].shape[-1],
                weather_dim=bundle["W_future_test"].shape[-1],
                control_dim=bundle["U_future_test"].shape[-1],
                target_dim=bundle["Y_future_test"].shape[-1],
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                nhead=cfg.transformer_heads,
                ff_dim=cfg.transformer_ff_dim,
            ),
            "checkpoint": f"patchtst_residual_joint_all_{cfg.control_compartment.lower()}.pt",
        },
    }


def _load_checkpoint(model, ckpt_path: Path, device) -> None:
    if not ckpt_path.exists():
        legacy_path = ckpt_path.parents[2] / f"{ckpt_path.stem}.pt"
        if legacy_path.exists():
            ckpt_path = legacy_path
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}. Run agc_mpc/main.py first to produce forecasting baselines."
        )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()


def _load_frozen_expert_if_needed(model, predictor_name: str, cfg: AGCConfig, device) -> None:
    if predictor_name not in {
        "itransformer_co2_frozen_expert",
        "itransformer_co2_late_frozen_expert",
        "itransformer_co2_teacher_distill",
        "itransformer_co2_recoupled_expert",
        "itransformer_co2_protected_expert",
        "itransformer_co2_protected_terminal",
        "itransformer_co2_horizon_mixture",
        "itransformer_co2_frozen_backbone_horizon_mixture",
    }:
        return
    checkpoint_name = f"co2_wavelet_gru_attn_joint_all_{cfg.control_compartment.lower()}.pt"
    checkpoint_path = Path(cfg.forecast_checkpoints_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing standalone CO2 expert checkpoint: {checkpoint_path}. "
            "Run benchmark_co2_specialist_forecasters.py for co2_wavelet_gru_attn first."
        )
    model.load_frozen_expert_checkpoint(str(checkpoint_path), map_location=device)


def _load_main_if_needed(model, predictor_name: str, cfg: AGCConfig, device) -> None:
    if predictor_name == "itransformer_co2_frozen_backbone_horizon_mixture":
        checkpoint_name = f"itransformer_co2_late_residual_joint_all_{cfg.control_compartment.lower()}.pt"
        checkpoint_path = Path(cfg.forecast_checkpoints_dir) / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing main late-residual checkpoint: {checkpoint_path}")
        model.load_main_checkpoint(str(checkpoint_path), map_location=device)
        return
    if predictor_name != "itransformer_co2_control_aware_fusion":
        return
    base_path = Path(cfg.forecast_checkpoints_dir) / (
        f"itransformer_co2_late_frozen_expert_joint_all_{cfg.control_compartment.lower()}.pt"
    )
    terminal_path = Path(cfg.forecast_checkpoints_dir) / (
        f"itransformer_co2_horizon_mixture_joint_all_{cfg.control_compartment.lower()}.pt"
    )
    if not base_path.exists():
        raise FileNotFoundError(f"Missing late-frozen expert checkpoint: {base_path}")
    if not terminal_path.exists():
        raise FileNotFoundError(f"Missing horizon-mixture checkpoint: {terminal_path}")
    model.load_base_checkpoint(str(base_path), map_location=device)
    model.load_terminal_checkpoint(str(terminal_path), map_location=device)


def _print_summary(summary) -> None:
    print(
        f"    {summary.controller:<8} | "
        f"objective={summary.objective_mean:.4f} | "
        f"control_delta={summary.control_delta_mae:.4f} | "
        f"action_tv={summary.action_tv:.4f}"
    )
    for target, mae in summary.target_mae.items():
        print(f"        {target:<10} MAE={mae:.3f}")
    print(f"        figure={summary.figure_path}")


def _suite_name(cfg: AGCConfig, predictors: list[str]) -> str:
    tag = str(getattr(cfg, "control_output_tag", "") or "").strip()
    suffix = f"_{tag}" if tag else ""
    if predictors == LATEST_PREDICTORS:
        return f"latest_predictor_suite_{cfg.control_compartment.lower()}_{cfg.control_eval_steps}steps{suffix}"
    if len(predictors) == 1:
        return f"{predictors[0]}_{cfg.control_compartment.lower()}_{cfg.control_eval_steps}steps{suffix}_control_suite"
    joined = "_".join(predictors)
    if len(joined) > 120:
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
        joined = f"{len(predictors)}predictors_{digest}"
    return f"predictor_suite_{joined}_{cfg.control_compartment.lower()}_{cfg.control_eval_steps}steps{suffix}"


def run_control_benchmarks(cfg: AGCConfig, predictors: list[str]) -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    _set_global_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("AGC Closed-Loop MPC Solver Benchmark")
    print("=" * 72)
    print(f"project_root: {project_root}")
    print(f"device: {device}")
    print(f"compartment: {cfg.control_compartment}")
    print(f"reference_mode: {cfg.control_reference_mode}")
    print(f"rollout_mode: {cfg.control_rollout_mode}")
    print(f"control_eval_steps: {cfg.control_eval_steps}")

    ensure_results_layout(cfg)
    processor = AGCDataProcessor(cfg)
    raw_bundle = processor.build_compartment_raw_bundle(cfg.control_compartment)
    scaled_bundle = processor.build_compartment_bundle(cfg.control_compartment)

    model_specs = _build_model_specs(scaled_bundle, cfg)
    unknown = [name for name in predictors if name not in model_specs]
    if unknown:
        raise ValueError(f"Unsupported predictors: {unknown}")

    suite_records = []
    for name in predictors:
        model = model_specs[name]["builder"]()
        _load_frozen_expert_if_needed(model, name, cfg, device)
        _load_main_if_needed(model, name, cfg, device)
        _load_checkpoint(
            model,
            project_root / "results" / "forecasting" / "checkpoints" / model_specs[name]["checkpoint"],
            device,
        )
        adapter = PredictiveControlAdapter(
            model=model,
            scalers=scaled_bundle["scalers"],
            feature_groups=scaled_bundle["feature_groups"],
            cfg=cfg,
            raw_bundle=raw_bundle,
            device=device,
        )
        simulator = AGCClosedLoopSimulator(adapter, raw_bundle, cfg)

        print("\n" + "-" * 72)
        print(f"Predictor: {name}")
        print("-" * 72)
        for controller in [
            RecordedBaselineController(adapter, cfg),
            GradientMPCController(adapter, cfg),
            CEMMPCController(adapter, cfg),
        ]:
            summary = simulator.run(controller, predictor_name=name)
            _print_summary(summary)
            suite_records.append(asdict(summary))

    suite_name = _suite_name(cfg, predictors)
    suite_path = Path(cfg.control_summaries_dir) / f"{suite_name}.json"
    suite_path.write_text(
        json.dumps(
            {
                "predictors": predictors,
                "controllers": ["recorded", "gradient_mpc", "cem_mpc"],
                "compartment": cfg.control_compartment,
                "reference_mode": cfg.control_reference_mode,
                "rollout_mode": cfg.control_rollout_mode,
                "start_idx": cfg.control_start_idx,
                "output_tag": getattr(cfg, "control_output_tag", ""),
                "steps": cfg.control_eval_steps,
                "target_cols": cfg.target_cols,
                "records": suite_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved suite summary: {suite_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AGC closed-loop control benchmarks.")
    parser.add_argument("--compartment", type=str, default=None, help="AGC compartment to benchmark.")
    parser.add_argument("--steps", type=int, default=None, help="Closed-loop rollout length.")
    parser.add_argument("--start-idx", type=int, default=None, help="Test-set rollout start index.")
    parser.add_argument(
        "--reference-mode",
        type=str,
        choices=["trajectory", "constant"],
        default=None,
        help="State reference source for the control objective.",
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=LATEST_PREDICTORS,
        choices=[
            "dlinear_baseline",
            "dlinear_forecaster",
            "frequency_forecaster",
            "gru_forecaster",
            "lstm_forecaster",
            "nlinear_forecaster",
            "segrnn_forecaster",
            "transformer_forecaster",
            "transformer_hybrid_baseline",
            "transformer_baseline",
            "itransformer_co2_residual",
            "itransformer_co2_late_residual",
            "itransformer_co2_frozen_expert",
            "itransformer_co2_late_frozen_expert",
            "itransformer_co2_teacher_distill",
            "itransformer_co2_recoupled_expert",
            "itransformer_co2_protected_expert",
            "itransformer_co2_protected_terminal",
            "itransformer_co2_horizon_mixture",
            "itransformer_co2_frozen_backbone_horizon_mixture",
            "itransformer_co2_control_aware_fusion",
            "itransformer_co2_wavelet_residual",
            "itransformer_co2_wavelet_blend",
            *LATEST_PREDICTORS,
        ],
        help="Forecast predictors to benchmark in closed loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _apply_three_target_control_protocol(AGCConfig())
    if args.compartment is not None:
        cfg.control_compartment = args.compartment
    if args.steps is not None:
        cfg.control_eval_steps = args.steps
    if args.start_idx is not None:
        cfg.control_start_idx = args.start_idx
    if args.reference_mode is not None:
        cfg.control_reference_mode = args.reference_mode

    run_control_benchmarks(cfg, predictors=args.predictors)


if __name__ == "__main__":
    main()
