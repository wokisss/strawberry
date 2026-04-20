# -*- coding: utf-8 -*-
"""Generate standard forecast figures for trained residual forecaster variants."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from config import AGCConfig
from data_processing.processor import AGCDataProcessor
from evaluation.evaluator import ForecasterEvaluator
from figure_layout import residual_figures_dir
from models.hybrid_residual_forecaster import ConditionalHybridResidualForecaster
from models.itransformer_residual_forecaster import (
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
from models.patchtst_residual_forecaster import ConditionalPatchTSTResidualForecaster
from results_utils import ensure_results_layout


MODEL_REGISTRY = {
    "transformer_hybrid_residual": ConditionalHybridResidualForecaster,
    "itransformer_residual": ConditionalITransformerResidualForecaster,
    "itransformer_co2_residual": ConditionalITransformerCO2ResidualForecaster,
    "itransformer_co2_late_residual": ConditionalITransformerCO2LateResidualForecaster,
    "itransformer_co2_frozen_expert": ConditionalITransformerCO2FrozenExpertForecaster,
    "itransformer_co2_late_frozen_expert": ConditionalITransformerCO2LateFrozenExpertForecaster,
    "itransformer_co2_teacher_distill": ConditionalITransformerCO2TeacherDistillForecaster,
    "itransformer_co2_recoupled_expert": ConditionalITransformerCO2RecoupledExpertForecaster,
    "itransformer_co2_protected_expert": ConditionalITransformerCO2ProtectedExpertForecaster,
    "itransformer_co2_protected_terminal": ConditionalITransformerCO2ProtectedTerminalForecaster,
    "itransformer_co2_horizon_mixture": ConditionalITransformerCO2HorizonMixtureForecaster,
    "itransformer_co2_frozen_backbone_horizon_mixture": ConditionalITransformerCO2FrozenBackboneHorizonMixtureForecaster,
    "itransformer_co2_wavelet_residual": ConditionalITransformerCO2WaveletResidualForecaster,
    "itransformer_co2_wavelet_blend": ConditionalITransformerCO2WaveletBlendForecaster,
    "patchtst_residual": ConditionalPatchTSTResidualForecaster,
}


def _apply_plot_protocol(cfg: AGCConfig) -> AGCConfig:
    run_cfg = copy.deepcopy(cfg)
    run_cfg.target_cols = ["Tair", "Rhair", "CO2air"]
    return run_cfg


def _build_bundle(processor: AGCDataProcessor, cfg: AGCConfig, regime: str, target_compartment: str):
    compartments = list(cfg.selected_compartments)
    if regime == "single":
        return processor.build_custom_bundle([target_compartment], [target_compartment])
    if regime == "joint_all":
        return processor.build_custom_bundle(compartments, [target_compartment])
    if regime == "leave_one_out":
        train_compartments = [comp for comp in compartments if comp != target_compartment]
        return processor.build_custom_bundle(train_compartments, [target_compartment])
    raise ValueError(f"Unsupported regime: {regime}")


def _build_model(model_name: str, cfg: AGCConfig, bundle):
    builder = MODEL_REGISTRY[model_name]
    return builder(
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        past_dim=bundle["X_past_train"].shape[-1],
        weather_dim=bundle["W_future_train"].shape[-1],
        control_dim=bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        nhead=cfg.transformer_heads,
        ff_dim=cfg.transformer_ff_dim,
    )


def _run_name(model_name: str, cfg: AGCConfig, regime: str, target_compartment: str) -> str:
    horizon_suffix = f"_h{cfg.horizon}" if cfg.horizon != 24 else ""
    return f"{model_name}{horizon_suffix}_{regime}_{target_compartment.lower()}"


def _load_frozen_expert_if_needed(model, model_name: str, cfg: AGCConfig, regime: str, target_compartment: str, device) -> None:
    if model_name not in {
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
    horizon_suffix = f"_h{cfg.horizon}" if cfg.horizon != 24 else ""
    checkpoint_name = f"co2_wavelet_gru_attn{horizon_suffix}_{regime}_{target_compartment.lower()}.pt"
    checkpoint_path = Path(cfg.forecast_checkpoints_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing standalone CO2 expert checkpoint: {checkpoint_path}. "
            "Run benchmark_co2_specialist_forecasters.py for co2_wavelet_gru_attn first."
        )
    model.load_frozen_expert_checkpoint(str(checkpoint_path), map_location=device)


def _load_main_if_needed(model, model_name: str, cfg: AGCConfig, regime: str, target_compartment: str, device) -> None:
    if model_name not in {"itransformer_co2_frozen_backbone_horizon_mixture"}:
        return
    horizon_suffix = f"_h{cfg.horizon}" if cfg.horizon != 24 else ""
    checkpoint_name = f"itransformer_co2_late_residual{horizon_suffix}_{regime}_{target_compartment.lower()}.pt"
    checkpoint_path = Path(cfg.forecast_checkpoints_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing main late-residual checkpoint: {checkpoint_path}")
    model.load_main_checkpoint(str(checkpoint_path), map_location=device)


def _load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _plot_one(model_name: str, cfg: AGCConfig, bundle, device: torch.device, regime: str, target_compartment: str) -> None:
    run_name = _run_name(model_name, cfg, regime, target_compartment)
    checkpoint_path = Path(cfg.forecast_checkpoints_dir) / f"{run_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint_path}")

    model = _build_model(model_name, cfg, bundle)
    _load_frozen_expert_if_needed(model, model_name, cfg, regime, target_compartment, device)
    _load_main_if_needed(model, model_name, cfg, regime, target_compartment, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    evaluator = ForecasterEvaluator(
        model,
        x_scaler=bundle["scalers"]["x"],
        y_scaler=bundle["scalers"]["y"],
        target_cols=bundle["feature_groups"]["y_future"],
        past_feature_cols=bundle["feature_groups"]["x_past"],
        device=device,
    )
    result = evaluator.evaluate(
        bundle["X_past_test"],
        bundle["W_future_test"],
        bundle["U_future_test"],
        bundle["Y_future_test"],
        model_name=run_name,
        output_dir=residual_figures_dir(cfg.forecast_figures_dir),
        num_plot_examples=cfg.plot_examples,
        plot_history_steps=cfg.plot_history_steps,
        forecast_rollout_examples=cfg.forecast_rollout_examples,
        forecast_rollout_steps=cfg.forecast_rollout_steps,
        forecast_rollout_stride=cfg.forecast_rollout_stride,
    )

    summary_path = Path(cfg.forecast_analysis_dir) / f"{run_name}_summary.json"
    summary = _load_summary(summary_path) or {
        "run_name": run_name,
        "dataset": "AGC",
        "target_compartment": target_compartment,
        "regime": regime,
    }
    summary["figure_paths"] = result["figure_paths"]
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODEL_REGISTRY.keys()],
        help="Residual variant to plot.",
    )
    parser.add_argument(
        "--regime",
        default="joint_all",
        choices=["single", "joint_all", "leave_one_out"],
    )
    parser.add_argument("--target-compartment", default="Reference")
    parser.add_argument("--horizon", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    cfg = _apply_plot_protocol(AGCConfig())
    if args.horizon is not None:
        cfg.horizon = args.horizon

    ensure_results_layout(cfg)
    processor = AGCDataProcessor(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = _build_bundle(processor, cfg, args.regime, args.target_compartment)

    model_names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for model_name in model_names:
        print("\n" + "=" * 72)
        print(f"Plot residual variant: {model_name}")
        print("=" * 72)
        _plot_one(model_name, cfg, bundle, device, args.regime, args.target_compartment)


if __name__ == "__main__":
    main()
