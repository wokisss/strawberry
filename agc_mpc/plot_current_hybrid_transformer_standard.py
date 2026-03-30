# -*- coding: utf-8 -*-
"""Generate standard forecast figures for the trained current AGC hybrid-transformer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from benchmark_current_hybrid_transformer import (
    _apply_fair_budget_overrides,
    _build_bundle,
    _set_global_seed,
)
from config import AGCConfig
from data_processing.processor import AGCDataProcessor
from evaluation.evaluator import ForecasterEvaluator
from figure_layout import current_hybrid_figures_dir
from models.transformer_hybrid_forecaster import ConditionalTransformerHybridForecaster
from results_utils import ensure_results_layout


def _run_name(cfg: AGCConfig, regime: str, target_compartment: str) -> str:
    horizon_suffix = f"_h{cfg.horizon}" if cfg.horizon != 24 else ""
    return f"current_hybrid_transformer{horizon_suffix}_{regime}_{target_compartment.lower()}"


def _load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _build_model(cfg: AGCConfig, bundle) -> ConditionalTransformerHybridForecaster:
    return ConditionalTransformerHybridForecaster(
        past_dim=bundle["X_past_train"].shape[-1],
        weather_dim=bundle["W_future_train"].shape[-1],
        control_dim=bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        nhead=cfg.transformer_heads,
        ff_dim=cfg.transformer_ff_dim,
        max_past_len=cfg.seq_len,
        max_future_len=cfg.horizon,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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

    cfg = _apply_fair_budget_overrides(AGCConfig())
    if args.horizon is not None:
        cfg.horizon = args.horizon

    ensure_results_layout(cfg)
    _set_global_seed(cfg.seed)
    processor = AGCDataProcessor(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = _build_bundle(processor, cfg, args.regime, args.target_compartment)
    run_name = _run_name(cfg, args.regime, args.target_compartment)

    checkpoint_path = Path(cfg.forecast_checkpoints_dir) / f"{run_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = _build_model(cfg, bundle)
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
        output_dir=current_hybrid_figures_dir(cfg.forecast_figures_dir),
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
        "target_compartment": args.target_compartment,
        "regime": args.regime,
    }
    summary["figure_paths"] = result["figure_paths"]
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Generated figure paths:")
    for key, value in result["figure_paths"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
