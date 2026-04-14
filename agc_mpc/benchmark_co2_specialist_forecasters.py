# -*- coding: utf-8 -*-
"""Benchmark standalone CO2 specialist forecasters inspired by greenhouse CO2 papers."""

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
from models.co2_specialist_forecasters import (
    ConditionalCO2LSTMForecaster,
    ConditionalCO2VMDLSTMFusionForecaster,
    ConditionalCO2WaveletGRUAttnForecaster,
)
from results_utils import ensure_results_layout
from training.trainer import Trainer


MODEL_REGISTRY = {
    "co2_env_lstm": {
        "builder": ConditionalCO2LSTMForecaster,
        "label": "CO2-only environmental-factor LSTM",
        "paper_family": [
            "Prediction of CO2 Concentration via Long Short-Term Memory Using Environmental Factors in Greenhouses",
            "Time-serial analysis of deep neural network models for prediction of climatic conditions inside a greenhouse",
        ],
        "architecture": "past-state LSTM encoder + future-known decoder LSTM + scalar CO2 head",
        "co2_idea": "Treat CO2 as a dedicated target and rely on recurrent memory to absorb greenhouse regime dynamics.",
        "integration_hint": "Works as a clean single-target baseline or as a teacher/expert branch for CO2.",
    },
    "co2_vmd_lstm_fusion": {
        "builder": ConditionalCO2VMDLSTMFusionForecaster,
        "label": "CO2-only VMD/WT-inspired LSTM fusion",
        "paper_family": [
            "Multi-model fusion method for predicting CO2 concentration in greenhouse tomatoes",
            "Prediction of CO2 concentration in mushroom greenhouse via optimized long and short term memory algorithm",
        ],
        "architecture": "trend/detail decomposition + dual LSTM encoders + future-query attention + dynamic fusion",
        "co2_idea": "Separate low-frequency trend from fast fluctuation, then fuse them adaptively per horizon.",
        "integration_hint": "Best fit for a CO2 specialist residual branch inside the existing multi-target forecaster.",
    },
    "co2_wavelet_gru_attn": {
        "builder": ConditionalCO2WaveletGRUAttnForecaster,
        "label": "CO2-only wavelet-inspired GRU with adaptive attention",
        "paper_family": [
            "Wavelet-decoupled GRU with adaptive attention for multi-step carbon dioxide concentration prediction in intelligent glass greenhouse",
        ],
        "architecture": "multi-scale decoupling + 3 GRU encoders + horizon-aware adaptive attention fusion",
        "co2_idea": "Model CO2 at multiple temporal bands and vary the branch weight as the forecast horizon grows.",
        "integration_hint": "Most directly transferable to the current dynamic-gating residual line for CO2air.",
    },
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
    run_cfg.early_stop_patience = 15
    run_cfg.target_cols = ["CO2air"]
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
        past_dim=bundle["X_past_train"].shape[-1],
        weather_dim=bundle["W_future_train"].shape[-1],
        control_dim=bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )


def _run_one_model(model_name: str, cfg: AGCConfig, bundle, device: torch.device, regime: str, describe_only: bool) -> None:
    print("\n" + "=" * 72)
    print(f"CO2 Specialist Variant: {model_name}")
    print("=" * 72)
    print(f"device: {device}")
    print(f"model_family: {MODEL_REGISTRY[model_name]['label']}")
    print(f"seq_len = {cfg.seq_len} ({cfg.seq_len * 5} min)")
    print(f"horizon = {cfg.horizon} ({cfg.horizon * 5} min)")
    print(f"batch_size = {cfg.batch_size}, epochs = {cfg.num_epochs}, lr = {cfg.learning_rate}")
    print(f"lambda_trend = {cfg.lambda_trend}, patience = {cfg.early_stop_patience}")
    if describe_only:
        print("---> Describe only mode, no training executed.")
        return

    model = _build_model(model_name, cfg, bundle)
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
            "intent": "Train standalone CO2 specialist forecasters under a fair AGC budget.",
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
            "early_stop_patience": cfg.early_stop_patience,
            "model_family": MODEL_REGISTRY[model_name]["label"],
        },
        "paper_family": MODEL_REGISTRY[model_name]["paper_family"],
        "architecture": MODEL_REGISTRY[model_name]["architecture"],
        "co2_idea": MODEL_REGISTRY[model_name]["co2_idea"],
        "integration_hint": MODEL_REGISTRY[model_name]["integration_hint"],
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
        help="CO2 specialist variant to run.",
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

    ensure_results_layout(cfg)
    _set_global_seed(cfg.seed)
    processor = AGCDataProcessor(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = _build_bundle(processor, cfg, args.regime, args.target_compartment)

    print("=" * 72)
    print("CO2 Specialist Forecaster Benchmark")
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
