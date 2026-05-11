# -*- coding: utf-8 -*-
"""Compare single-compartment, joint, and leave-one-out training regimes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import AGCConfig
from data_processing.processor import AGCDataProcessor
from evaluation.evaluator import ForecasterEvaluator
from models.dlinear_forecaster import ConditionalDLinearForecaster
from models.frequency_forecaster import ConditionalFrequencyMLPForecaster
from models.gru_forecaster import ConditionalGRUForecaster
from models.lstm_forecaster import ConditionalLSTMForecaster
from models.nlinear_forecaster import ConditionalNLinearForecaster
from models.seg_rnn_forecaster import ConditionalSegRNNForecaster
from models.transformer_forecaster import ConditionalTransformerForecaster
from models.transformer_hybrid_forecaster import ConditionalTransformerHybridForecaster
from results_utils import ensure_results_layout
from training.trainer import Trainer


MODEL_ALIASES = {
    "gru": "gru_baseline",
    "gru_baseline": "gru_baseline",
    "lstm": "lstm_baseline",
    "lstm_baseline": "lstm_baseline",
    "dlinear": "dlinear_baseline",
    "dlinear_baseline": "dlinear_baseline",
    "frequency": "frequency_baseline",
    "frequency_baseline": "frequency_baseline",
    "freq": "frequency_baseline",
    "nlinear": "nlinear_baseline",
    "nlinear_baseline": "nlinear_baseline",
    "segrnn": "segrnn_baseline",
    "segrnn_baseline": "segrnn_baseline",
    "transformer": "transformer_baseline",
    "transformer_baseline": "transformer_baseline",
    "transformer_hybrid": "transformer_hybrid_baseline",
    "transformer_hybrid_baseline": "transformer_hybrid_baseline",
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


def _normalize_model_names(model_names: List[str]) -> List[str]:
    normalized = []
    for name in model_names:
        key = name.strip().lower()
        if key == "all":
            return [
                "gru_baseline",
                "lstm_baseline",
                "dlinear_baseline",
                "frequency_baseline",
                "nlinear_baseline",
                "segrnn_baseline",
                "transformer_baseline",
                "transformer_hybrid_baseline",
            ]
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unsupported model name: {name}")
        normalized.append(MODEL_ALIASES[key])
    return normalized


def _create_model(model_name: str, bundle: Dict[str, np.ndarray], cfg: AGCConfig):
    common = {
        "past_dim": bundle["X_past_train"].shape[-1],
        "weather_dim": bundle["W_future_train"].shape[-1],
        "control_dim": bundle["U_future_train"].shape[-1],
        "target_dim": bundle["Y_future_train"].shape[-1],
    }
    if model_name == "gru_baseline":
        return ConditionalGRUForecaster(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            **common,
        )
    if model_name == "lstm_baseline":
        return ConditionalLSTMForecaster(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            **common,
        )
    if model_name == "dlinear_baseline":
        return ConditionalDLinearForecaster(
            seq_len=cfg.seq_len,
            horizon=cfg.horizon,
            hidden_dim=cfg.hidden_dim,
            **common,
        )
    if model_name == "frequency_baseline":
        return ConditionalFrequencyMLPForecaster(
            seq_len=cfg.seq_len,
            hidden_dim=cfg.hidden_dim,
            **common,
        )
    if model_name == "nlinear_baseline":
        return ConditionalNLinearForecaster(
            seq_len=cfg.seq_len,
            horizon=cfg.horizon,
            hidden_dim=cfg.hidden_dim,
            **common,
        )
    if model_name == "segrnn_baseline":
        return ConditionalSegRNNForecaster(
            seq_len=cfg.seq_len,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            seg_len=cfg.seg_len,
            **common,
        )
    if model_name == "transformer_baseline":
        return ConditionalTransformerForecaster(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            nhead=cfg.transformer_heads,
            ff_dim=cfg.transformer_ff_dim,
            max_past_len=cfg.seq_len,
            max_future_len=cfg.horizon,
            **common,
        )
    if model_name == "transformer_hybrid_baseline":
        return ConditionalTransformerHybridForecaster(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            nhead=cfg.transformer_heads,
            ff_dim=cfg.transformer_ff_dim,
            max_past_len=cfg.seq_len,
            max_future_len=cfg.horizon,
            **common,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def _describe_bundle(processor: AGCDataProcessor, bundle: Dict[str, np.ndarray]) -> List[str]:
    lines = processor.summarize_bundle(bundle)
    for key in ["train_compartments", "eval_compartments", "compartments"]:
        if key in bundle:
            lines.append(f"{key}: {bundle[key]}")
    return lines


def _build_regime_bundle(
    processor: AGCDataProcessor,
    cfg: AGCConfig,
    regime: str,
    target_compartment: str,
) -> Dict[str, np.ndarray]:
    all_compartments = list(cfg.selected_compartments)
    if target_compartment not in all_compartments:
        raise ValueError(f"Unknown target compartment: {target_compartment}")

    if regime == "single":
        return processor.build_custom_bundle([target_compartment], [target_compartment])
    if regime == "joint_all":
        return processor.build_custom_bundle(all_compartments, [target_compartment])
    if regime == "leave_one_out":
        train_compartments = [comp for comp in all_compartments if comp != target_compartment]
        return processor.build_custom_bundle(train_compartments, [target_compartment])
    raise ValueError(f"Unsupported regime: {regime}")


def _metrics_to_summary(metrics: Dict[str, List[float]], target_cols: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for idx, target in enumerate(target_cols):
        summary[target] = {
            "full_r2": float(metrics["full_r2"][idx]),
            "full_mae": float(metrics["full_mae"][idx]),
            "final_r2": float(metrics["final_r2"][idx]),
            "final_mae": float(metrics["final_mae"][idx]),
        }
    return summary


def _save_regime_figure(
    records: List[Dict[str, object]],
    target_cols: List[str],
    target_compartment: str,
    output_path: Path,
) -> None:
    regimes = [record["regime"] for record in records]
    model_names = [record["model_name"] for record in records]
    labels = [f"{model}\n{regime}" for model, regime in zip(model_names, regimes)]

    fig, axes = plt.subplots(len(target_cols), 2, figsize=(14, 3.8 * len(target_cols)), squeeze=False)
    x = np.arange(len(records))

    for row_idx, target in enumerate(target_cols):
        final_mae = [record["metrics_by_target"][target]["final_mae"] for record in records]
        final_r2 = [record["metrics_by_target"][target]["final_r2"] for record in records]

        axes[row_idx, 0].bar(x, final_mae, color="#5b8ff9")
        axes[row_idx, 0].set_title(f"{target} final MAE")
        axes[row_idx, 0].set_xticks(x, labels, rotation=35, ha="right")
        axes[row_idx, 0].grid(axis="y", alpha=0.25)

        axes[row_idx, 1].bar(x, final_r2, color="#5ad8a6")
        axes[row_idx, 1].set_title(f"{target} final R2")
        axes[row_idx, 1].set_xticks(x, labels, rotation=35, ha="right")
        axes[row_idx, 1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Training regime comparison on held-out test of {target_compartment}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run_one(
    model_name: str,
    regime: str,
    target_compartment: str,
    bundle: Dict[str, np.ndarray],
    cfg: AGCConfig,
    device: torch.device,
) -> Dict[str, object]:
    run_cfg = copy.deepcopy(cfg)
    tag = f"{model_name}_{regime}_{target_compartment.lower()}"
    run_cfg.model_save_path = f"{run_cfg.forecast_checkpoints_dir}/{tag}.pt"

    print("\n" + "=" * 72)
    print(f"Training regime run: model={model_name} | regime={regime} | target={target_compartment}")
    print("=" * 72)

    model = _create_model(model_name, bundle, run_cfg)
    trainer = Trainer(model, run_cfg, device=device)
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

    evaluator = ForecasterEvaluator(
        model,
        x_scaler=bundle["scalers"]["x"],
        y_scaler=bundle["scalers"]["y"],
        target_cols=bundle["feature_groups"]["y_future"],
        past_feature_cols=bundle["feature_groups"]["x_past"],
        device=device,
    )
    eval_result = evaluator.evaluate(
        bundle["X_past_test"],
        bundle["W_future_test"],
        bundle["U_future_test"],
        bundle["Y_future_test"],
        model_name=tag,
        output_dir=run_cfg.forecast_analysis_dir,
        num_plot_examples=run_cfg.plot_examples,
        plot_history_steps=run_cfg.plot_history_steps,
        forecast_rollout_examples=run_cfg.forecast_rollout_examples,
        forecast_rollout_steps=run_cfg.forecast_rollout_steps,
        forecast_rollout_stride=run_cfg.forecast_rollout_stride,
    )

    return {
        "model_name": model_name,
        "regime": regime,
        "target_compartment": target_compartment,
        "train_compartments": bundle.get("train_compartments", bundle.get("compartments", [])),
        "eval_compartments": bundle.get("eval_compartments", bundle.get("compartments", [])),
        "shapes": {
            split: {
                "X_past": list(bundle[f"X_past_{split}"].shape),
                "W_future": list(bundle[f"W_future_{split}"].shape),
                "U_future": list(bundle[f"U_future_{split}"].shape),
                "Y_future": list(bundle[f"Y_future_{split}"].shape),
            }
            for split in ["train", "val", "test"]
        },
        "metrics_by_target": _metrics_to_summary(
            eval_result["metrics"],
            bundle["feature_groups"]["y_future"],
        ),
        "figure_paths": eval_result["figure_paths"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dlinear"],
        help="Models to run: dlinear, frequency, nlinear, gru, lstm, segrnn, transformer, transformer_hybrid, or all",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["single", "joint_all", "leave_one_out"],
        choices=["single", "joint_all", "leave_one_out"],
        help="Training regimes to compare.",
    )
    parser.add_argument(
        "--target-compartment",
        default="Reference",
        help="Compartment used for single-compartment training and held-out evaluation.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size.")
    parser.add_argument(
        "--control-protocol",
        action="store_true",
        help="Use the strict three-target AGC control protocol: Tair, Rhair, CO2air.",
    )
    parser.add_argument(
        "--fair-budget",
        action="store_true",
        help="Use the formal fair-budget training protocol.",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Build the bundles and print shapes without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    cfg = AGCConfig()
    if args.control_protocol:
        cfg.target_cols = ["Tair", "Rhair", "CO2air"]
        cfg.track_weights = cfg.track_weights[:3]
        cfg.constant_target_values = cfg.constant_target_values[:3]
    if args.fair_budget:
        cfg.batch_size = 256
        cfg.num_epochs = 200
        cfg.learning_rate = 1e-4
        cfg.lambda_trend = 0.3
        cfg.early_stop_patience = 15
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    _set_global_seed(cfg.seed)
    ensure_results_layout(cfg)
    processor = AGCDataProcessor(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = _normalize_model_names(args.models)

    print("=" * 72)
    print("AGC Training Regime Comparison")
    print("=" * 72)
    print(f"target_compartment: {args.target_compartment}")
    print(f"regimes: {args.regimes}")
    print(f"models: {model_names}")
    print(f"device: {device}")
    print(f"targets: {cfg.target_cols}")
    print(
        "training_budget: "
        f"batch_size={cfg.batch_size}, epochs={cfg.num_epochs}, "
        f"lr={cfg.learning_rate}, lambda_trend={cfg.lambda_trend}, "
        f"patience={cfg.early_stop_patience}"
    )

    bundles_by_regime: Dict[str, Dict[str, np.ndarray]] = {}
    for regime in args.regimes:
        bundle = _build_regime_bundle(processor, cfg, regime, args.target_compartment)
        bundles_by_regime[regime] = bundle
        print("\n" + "-" * 72)
        print(f"Bundle summary | regime={regime}")
        print("-" * 72)
        for line in _describe_bundle(processor, bundle):
            print(f"    {line}")

    if args.describe_only:
        print("---> Describe only mode, no training executed.")
        return

    records: List[Dict[str, object]] = []
    for model_name in model_names:
        for regime in args.regimes:
            records.append(
                _run_one(
                    model_name=model_name,
                    regime=regime,
                    target_compartment=args.target_compartment,
                    bundle=bundles_by_regime[regime],
                    cfg=cfg,
                    device=device,
                )
            )

    summary = {
        "target_compartment": args.target_compartment,
        "regimes": args.regimes,
        "models": model_names,
        "records": records,
    }

    summary_path = Path(cfg.forecast_analysis_dir) / (
        f"training_regimes_{args.target_compartment.lower()}_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    figure_path = Path(cfg.forecast_analysis_dir) / (
        f"training_regimes_{args.target_compartment.lower()}_summary.png"
    )
    _save_regime_figure(
        records=records,
        target_cols=bundles_by_regime[args.regimes[0]]["feature_groups"]["y_future"],
        target_compartment=args.target_compartment,
        output_path=figure_path,
    )

    print("\n" + "=" * 72)
    print("Training regime summary")
    print("=" * 72)
    for record in records:
        print(f"{record['model_name']} | {record['regime']}")
        for target, metrics in record["metrics_by_target"].items():
            print(
                f"    {target:<10} | "
                f"Final R2={metrics['final_r2']:.4f} | "
                f"Final MAE={metrics['final_mae']:.3f}"
            )
    print(f"summary_json: {summary_path}")
    print(f"summary_figure: {figure_path}")


if __name__ == "__main__":
    main()
