# -*- coding: utf-8 -*-
"""Benchmark DiffMPC-style Transformer-hybrid on AGC with old-project-like settings."""

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
from models.diffmpc_style_transformer import DiffMPCStyleTransformerHybridForecaster
from results_utils import ensure_results_layout
from training.trainer import Trainer


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


def _apply_diffmpc_style_overrides(cfg: AGCConfig) -> AGCConfig:
    run_cfg = copy.deepcopy(cfg)
    run_cfg.seq_len = 48
    run_cfg.horizon = 24
    run_cfg.batch_size = 256
    run_cfg.num_epochs = 200
    run_cfg.learning_rate = 1e-4
    run_cfg.lambda_trend = 0.3
    run_cfg.early_stop_patience = 15
    run_cfg.target_cols = ["Tair", "Rhair", "CO2air"]
    return run_cfg


def _build_bundle(
    processor: AGCDataProcessor,
    cfg: AGCConfig,
    regime: str,
    target_compartment: str,
):
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
        "sample_count": int(pred_real.shape[0]),
        "horizon": int(pred_real.shape[1]),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime",
        default="single",
        choices=["single", "joint_all", "leave_one_out"],
        help="Training regime for the AGC side.",
    )
    parser.add_argument(
        "--target-compartment",
        default="Reference",
        help="Held-out evaluation compartment on AGC.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default DiffMPC-style 200 epochs.",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Only build the dataset bundle and print the protocol.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    cfg = _apply_diffmpc_style_overrides(AGCConfig())
    if args.epochs is not None:
        cfg.num_epochs = args.epochs

    ensure_results_layout(cfg)
    _set_global_seed(cfg.seed)
    processor = AGCDataProcessor(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = _build_bundle(processor, cfg, args.regime, args.target_compartment)

    print("=" * 72)
    print("DiffMPC-style Transformer on AGC")
    print("=" * 72)
    print(f"device: {device}")
    print(f"regime: {args.regime}")
    print(f"target_compartment: {args.target_compartment}")
    print("protocol:")
    print("    model_family = diffmpc transformer-hybrid")
    print("    targets = ['Tair', 'Rhair', 'CO2air']")
    print("    seq_len = 48 (4h @ 5min, AGC equivalent of old 240min history)")
    print("    horizon = 24 (2h @ 5min, AGC equivalent of old 120min horizon)")
    print("    d_model = 64, nhead = 4, num_layers = 4, ff_dim = 128, dropout = 0.1")
    print(f"    batch_size = {cfg.batch_size}, epochs = {cfg.num_epochs}, lr = {cfg.learning_rate}")
    print(f"    lambda_trend = {cfg.lambda_trend}, patience = {cfg.early_stop_patience}")
    for line in processor.summarize_bundle(bundle):
        print(f"    {line}")

    if args.describe_only:
        print("---> Describe only mode, no training executed.")
        return

    model = DiffMPCStyleTransformerHybridForecaster(
        past_dim=bundle["X_past_train"].shape[-1],
        future_dim=bundle["W_future_train"].shape[-1] + bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        target_indices=[bundle["feature_groups"]["x_past"].index(col) for col in cfg.target_cols],
    )

    run_name = f"diffmpc_style_transformer_{args.regime}_{args.target_compartment.lower()}"
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
        "target_compartment": args.target_compartment,
        "regime": args.regime,
        "protocol": {
            "intent": "Keep the old DiffMPC Transformer-hybrid structure and training budget as close as possible while moving to AGC.",
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
            "model": {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 4,
                "dim_feedforward": 128,
                "dropout": 0.1,
            },
        },
        "train_compartments": bundle.get("train_compartments", bundle.get("compartments", [])),
        "eval_compartments": bundle.get("eval_compartments", bundle.get("compartments", [])),
        "metrics_by_target": _summarize_metrics(cfg.target_cols, metrics),
        "checkpoint_path": cfg.model_save_path,
    }

    summary_path = Path(cfg.forecast_analysis_dir) / f"{run_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("DiffMPC-style Transformer summary")
    print("=" * 72)
    for target, values in summary["metrics_by_target"].items():
        print(
            f"{target:<10} | "
            f"Full R2={values['full_r2']:.4f} MAE={values['full_mae']:.3f} | "
            f"Final R2={values['final_r2']:.4f} MAE={values['final_mae']:.3f}"
        )
    print(f"summary_json: {summary_path}")
    print(f"checkpoint: {cfg.model_save_path}")


if __name__ == "__main__":
    main()
