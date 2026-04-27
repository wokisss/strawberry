# -*- coding: utf-8 -*-
"""Benchmark DLinear-main residual forecaster variants under a fair AGC budget."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from config import AGCConfig
from data_processing.processor import AGCDataProcessor
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
from models.patchtst_residual_forecaster import ConditionalPatchTSTResidualForecaster
from results_utils import ensure_results_layout
from training.trainer import Trainer


MODEL_REGISTRY = {
    "transformer_hybrid_residual": {
        "builder": ConditionalHybridResidualForecaster,
        "label": "dlinear main path + transformer-hybrid residual",
    },
    "itransformer_residual": {
        "builder": ConditionalITransformerResidualForecaster,
        "label": "dlinear main path + itransformer residual",
    },
    "itransformer_co2_residual": {
        "builder": ConditionalITransformerCO2ResidualForecaster,
        "label": "dlinear main path + itransformer residual + co2 specialist",
    },
    "itransformer_co2_late_residual": {
        "builder": ConditionalITransformerCO2LateResidualForecaster,
        "label": "dlinear main path + itransformer residual + late-horizon co2 adapter",
    },
    "itransformer_co2_frozen_expert": {
        "builder": ConditionalITransformerCO2FrozenExpertForecaster,
        "label": "dlinear main path + itransformer residual + frozen wavelet co2 expert",
    },
    "itransformer_co2_late_frozen_expert": {
        "builder": ConditionalITransformerCO2LateFrozenExpertForecaster,
        "label": "dlinear main path + itransformer residual + late-horizon frozen wavelet co2 expert",
    },
    "itransformer_co2_teacher_distill": {
        "builder": ConditionalITransformerCO2TeacherDistillForecaster,
        "label": "dlinear main path + itransformer residual + frozen wavelet teacher distillation",
    },
    "itransformer_co2_recoupled_expert": {
        "builder": ConditionalITransformerCO2RecoupledExpertForecaster,
        "label": "dlinear main path + itransformer residual + recoupled late frozen co2 expert",
    },
    "itransformer_co2_protected_expert": {
        "builder": ConditionalITransformerCO2ProtectedExpertForecaster,
        "label": "late-residual main path + protected late frozen co2 expert",
    },
    "itransformer_co2_protected_terminal": {
        "builder": ConditionalITransformerCO2ProtectedTerminalForecaster,
        "label": "late-residual main path + protected late frozen co2 expert + terminal-aware loss",
    },
    "itransformer_co2_horizon_mixture": {
        "builder": ConditionalITransformerCO2HorizonMixtureForecaster,
        "label": "late-residual main path + protected frozen co2 expert + terminal pullback",
    },
    "itransformer_co2_frozen_backbone_horizon_mixture": {
        "builder": ConditionalITransformerCO2FrozenBackboneHorizonMixtureForecaster,
        "label": "frozen late-residual backbone + protected frozen co2 expert + terminal pullback",
    },
    "itransformer_co2_control_aware_fusion": {
        "builder": ConditionalITransformerCO2ControlAwareFusionForecaster,
        "label": "late-frozen control anchor + horizon-mixture terminal fusion",
    },
    "itransformer_co2_wavelet_residual": {
        "builder": ConditionalITransformerCO2WaveletResidualForecaster,
        "label": "dlinear main path + itransformer residual + wavelet-style co2 adapter",
    },
    "itransformer_co2_wavelet_blend": {
        "builder": ConditionalITransformerCO2WaveletBlendForecaster,
        "label": "dlinear main path + itransformer residual + wavelet-style co2 blend expert",
    },
    "patchtst_residual": {
        "builder": ConditionalPatchTSTResidualForecaster,
        "label": "dlinear main path + patchtst residual",
    },
}

AUXILIARY_WEIGHTS = {
    "itransformer_co2_teacher_distill": 0.2,
    "itransformer_co2_recoupled_expert": 0.15,
    "itransformer_co2_protected_terminal": 0.08,
    "itransformer_co2_horizon_mixture": 0.05,
    "itransformer_co2_frozen_backbone_horizon_mixture": 0.05,
    "itransformer_co2_control_aware_fusion": 0.08,
}


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


def _apply_fair_budget_overrides(cfg: AGCConfig) -> AGCConfig:
    run_cfg = copy.deepcopy(cfg)
    run_cfg.batch_size = 256
    run_cfg.num_epochs = 200
    run_cfg.learning_rate = 1e-4
    run_cfg.lambda_trend = 0.3
    run_cfg.lambda_auxiliary = 0.0
    run_cfg.early_stop_patience = 15
    run_cfg.target_cols = ["Tair", "Rhair", "CO2air"]
    return run_cfg


def _build_bundle(processor: AGCDataProcessor, cfg: AGCConfig, regime: str, target_compartment: str):
    compartments = list(cfg.selected_compartments)
    if target_compartment not in compartments:
        raise ValueError(f"Unknown target compartment: {target_compartment}")

    if regime == "single":
        return processor.build_custom_bundle([target_compartment], [target_compartment])
    if regime == "joint_all":
        return processor.build_custom_bundle(compartments, [target_compartment])
    if regime == "leave_one_out":
        train_compartments = [comp for comp in compartments if comp != target_compartment]
        return processor.build_custom_bundle(train_compartments, [target_compartment])
    raise ValueError(f"Unsupported regime: {regime}")


def _inverse_targets(y_scaler, arr: np.ndarray) -> np.ndarray:
    shape = arr.shape
    flat = arr.reshape(-1, shape[-1])
    inv = y_scaler.inverse_transform(flat)
    return inv.reshape(shape)


def _evaluate_metrics(model, bundle, device: torch.device) -> dict:
    model.eval()
    preds = []
    truths = []
    batch_size = 128

    with torch.no_grad():
        for start in range(0, len(bundle["X_past_test"]), batch_size):
            end = min(start + batch_size, len(bundle["X_past_test"]))
            xb = torch.tensor(bundle["X_past_test"][start:end], dtype=torch.float32, device=device)
            wb = torch.tensor(bundle["W_future_test"][start:end], dtype=torch.float32, device=device)
            ub = torch.tensor(bundle["U_future_test"][start:end], dtype=torch.float32, device=device)
            preds.append(model(xb, wb, ub).cpu().numpy())
            truths.append(bundle["Y_future_test"][start:end])

    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(truths, axis=0)
    pred_real = _inverse_targets(bundle["scalers"]["y"], pred_norm)
    true_real = _inverse_targets(bundle["scalers"]["y"], true_norm)

    pred_flat = pred_real.reshape(-1, pred_real.shape[-1])
    true_flat = true_real.reshape(-1, true_real.shape[-1])
    pred_final = pred_real[:, -1, :]
    true_final = true_real[:, -1, :]

    return {
        "full_r2": [float(r2_score(true_flat[:, i], pred_flat[:, i])) for i in range(pred_flat.shape[-1])],
        "full_mae": [float(mean_absolute_error(true_flat[:, i], pred_flat[:, i])) for i in range(pred_flat.shape[-1])],
        "final_r2": [float(r2_score(true_final[:, i], pred_final[:, i])) for i in range(pred_flat.shape[-1])],
        "final_mae": [float(mean_absolute_error(true_final[:, i], pred_final[:, i])) for i in range(pred_flat.shape[-1])],
        "representative_window": {
            "sample_idx": int(len(pred_real) // 2),
            "true": true_real[len(true_real) // 2].tolist(),
            "pred": pred_real[len(pred_real) // 2].tolist(),
        },
    }


def _summarize_metrics(target_cols: list[str], metrics: dict) -> dict:
    return {
        target: {
            "full_r2": metrics["full_r2"][idx],
            "full_mae": metrics["full_mae"][idx],
            "final_r2": metrics["final_r2"][idx],
            "final_mae": metrics["final_mae"][idx],
        }
        for idx, target in enumerate(target_cols)
    }


def _build_model(model_name: str, cfg: AGCConfig, bundle):
    builder = MODEL_REGISTRY[model_name]["builder"]
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


def _maybe_load_frozen_expert(model, model_name: str, cfg: AGCConfig, bundle, regime: str, device: torch.device) -> None:
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
    checkpoint_name = f"co2_wavelet_gru_attn{horizon_suffix}_{regime}_{bundle['eval_compartments'][0].lower()}.pt"
    checkpoint_path = Path(cfg.forecast_checkpoints_dir) / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing standalone CO2 expert checkpoint: {checkpoint_path}. "
            "Run benchmark_co2_specialist_forecasters.py for co2_wavelet_gru_attn first."
        )
    model.load_frozen_expert_checkpoint(str(checkpoint_path), map_location=device)


def _maybe_load_main_checkpoint(model, model_name: str, cfg: AGCConfig, bundle, regime: str, device: torch.device) -> None:
    horizon_suffix = f"_h{cfg.horizon}" if cfg.horizon != 24 else ""
    eval_name = bundle["eval_compartments"][0].lower()
    if model_name == "itransformer_co2_frozen_backbone_horizon_mixture":
        checkpoint_name = f"itransformer_co2_late_residual{horizon_suffix}_{regime}_{eval_name}.pt"
        checkpoint_path = Path(cfg.forecast_checkpoints_dir) / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing main late-residual checkpoint: {checkpoint_path}. "
                "Run benchmark_residual_forecaster_variants.py for itransformer_co2_late_residual first."
            )
        model.load_main_checkpoint(str(checkpoint_path), map_location=device)
        return
    if model_name != "itransformer_co2_control_aware_fusion":
        return
    base_path = Path(cfg.forecast_checkpoints_dir) / (
        f"itransformer_co2_late_frozen_expert{horizon_suffix}_{regime}_{eval_name}.pt"
    )
    terminal_path = Path(cfg.forecast_checkpoints_dir) / (
        f"itransformer_co2_horizon_mixture{horizon_suffix}_{regime}_{eval_name}.pt"
    )
    if not base_path.exists():
        raise FileNotFoundError(
            f"Missing late-frozen expert checkpoint: {base_path}. "
            "Run benchmark_residual_forecaster_variants.py for itransformer_co2_late_frozen_expert first."
        )
    if not terminal_path.exists():
        raise FileNotFoundError(
            f"Missing horizon-mixture checkpoint: {terminal_path}. "
            "Run benchmark_residual_forecaster_variants.py for itransformer_co2_horizon_mixture first."
        )
    model.load_base_checkpoint(str(base_path), map_location=device)
    model.load_terminal_checkpoint(str(terminal_path), map_location=device)


def _run_one_model(model_name: str, cfg: AGCConfig, bundle, device: torch.device, regime: str, describe_only: bool) -> None:
    cfg = copy.deepcopy(cfg)
    cfg.lambda_auxiliary = AUXILIARY_WEIGHTS.get(model_name, cfg.lambda_auxiliary)

    print("\n" + "=" * 72)
    print(f"Residual Variant: {model_name}")
    print("=" * 72)
    print(f"device: {device}")
    print(f"model_family: {MODEL_REGISTRY[model_name]['label']}")
    print(f"seq_len = {cfg.seq_len} ({cfg.seq_len * 5} min)")
    print(f"horizon = {cfg.horizon} ({cfg.horizon * 5} min)")
    print(f"batch_size = {cfg.batch_size}, epochs = {cfg.num_epochs}, lr = {cfg.learning_rate}")
    print(
        f"lambda_trend = {cfg.lambda_trend}, lambda_auxiliary = {cfg.lambda_auxiliary}, "
        f"patience = {cfg.early_stop_patience}"
    )
    if describe_only:
        print("---> Describe only mode, no training executed.")
        return

    model = _build_model(model_name, cfg, bundle)
    _maybe_load_frozen_expert(model, model_name, cfg, bundle, regime, device)
    _maybe_load_main_checkpoint(model, model_name, cfg, bundle, regime, device)
    horizon_suffix = f"_h{cfg.horizon}" if cfg.horizon != 24 else ""
    run_name = f"{model_name}{horizon_suffix}_{regime}_{bundle['eval_compartments'][0].lower()}"
    cfg.model_save_path = f"{cfg.forecast_checkpoints_dir}/{run_name}.pt"
    trainer = Trainer(model, cfg, device=device)
    trainer.train(
        bundle["X_past_train"],
        bundle["W_future_train"],
        bundle["U_future_train"],
        bundle["Y_future_train"],
        bundle["X_past_val"],
        bundle["W_future_val"],
        bundle["U_future_val"],
        bundle["Y_future_val"],
    )

    metrics = _evaluate_metrics(model, bundle, device)
    summary = {
        "run_name": run_name,
        "dataset": "AGC",
        "target_compartment": bundle["eval_compartments"][0],
        "regime": regime,
        "protocol": {
            "intent": "Train DLinear-main residual forecaster variants under a fair AGC budget.",
            "variant": model_name,
            "targets": cfg.target_cols,
            "seq_len_steps": cfg.seq_len,
            "seq_len_minutes": cfg.seq_len * 5,
            "horizon_steps": cfg.horizon,
            "horizon_minutes": cfg.horizon * 5,
            "batch_size": cfg.batch_size,
            "num_epochs": cfg.num_epochs,
            "learning_rate": cfg.learning_rate,
            "lambda_trend": cfg.lambda_trend,
            "lambda_auxiliary": cfg.lambda_auxiliary,
            "early_stop_patience": cfg.early_stop_patience,
            "model_family": MODEL_REGISTRY[model_name]["label"],
        },
        "train_compartments": bundle.get("train_compartments", bundle.get("compartments", [])),
        "eval_compartments": bundle.get("eval_compartments", bundle.get("compartments", [])),
        "metrics_by_target": _summarize_metrics(cfg.target_cols, metrics),
        "representative_window": metrics["representative_window"],
        "checkpoint_path": cfg.model_save_path,
    }

    summary_path = Path(cfg.forecast_analysis_dir) / f"{run_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("summary:")
    for target, values in summary["metrics_by_target"].items():
        print(
            f"    {target:<10} | Full R2={values['full_r2']:.4f} MAE={values['full_mae']:.3f} | "
            f"Final R2={values['final_r2']:.4f} MAE={values['final_mae']:.3f}"
        )
    print(f"summary_json: {summary_path}")
    print(f"checkpoint: {cfg.model_save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODEL_REGISTRY.keys()],
        help="Residual variant to run.",
    )
    parser.add_argument(
        "--regime",
        default="joint_all",
        choices=["single", "joint_all", "leave_one_out"],
    )
    parser.add_argument("--target-compartment", default="Reference")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--describe-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    cfg = _apply_fair_budget_overrides(AGCConfig())
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.horizon is not None:
        cfg.horizon = args.horizon
    if args.model == "itransformer_co2_teacher_distill":
        cfg.lambda_auxiliary = 0.2
    if args.model == "itransformer_co2_recoupled_expert":
        cfg.lambda_auxiliary = 0.15
    if args.model == "itransformer_co2_protected_terminal":
        cfg.lambda_auxiliary = 0.08

    ensure_results_layout(cfg)
    _set_global_seed(cfg.seed)
    processor = AGCDataProcessor(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = _build_bundle(processor, cfg, args.regime, args.target_compartment)

    print("=" * 72)
    print("Residual Forecaster Variant Benchmark")
    print("=" * 72)
    print(f"regime: {args.regime}")
    print(f"target_compartment: {args.target_compartment}")
    for line in processor.summarize_bundle(bundle):
        print(f"    {line}")

    model_names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for model_name in model_names:
        _run_one_model(model_name, cfg, bundle, device, args.regime, args.describe_only)


if __name__ == "__main__":
    main()
