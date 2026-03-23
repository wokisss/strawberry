# -*- coding: utf-8 -*-
"""Entry point for AGC closed-loop MPC solver benchmarks."""

from __future__ import annotations

import argparse
import os
import random
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
from models.transformer_forecaster import ConditionalTransformerForecaster
from models.transformer_hybrid_forecaster import ConditionalTransformerHybridForecaster
from results_utils import ensure_results_layout


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_models(bundle, cfg):
    models = {
        "dlinear_baseline": ConditionalDLinearForecaster(
            seq_len=cfg.seq_len,
            horizon=cfg.horizon,
            past_dim=bundle["X_past_test"].shape[-1],
            weather_dim=bundle["W_future_test"].shape[-1],
            control_dim=bundle["U_future_test"].shape[-1],
            target_dim=bundle["Y_future_test"].shape[-1],
            hidden_dim=cfg.hidden_dim,
        ),
        "transformer_hybrid_baseline": ConditionalTransformerHybridForecaster(
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
        "transformer_baseline": ConditionalTransformerForecaster(
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
    }
    return models


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


def run_control_benchmarks(cfg: AGCConfig) -> None:
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

    models = _build_models(scaled_bundle, cfg)
    for name, model in models.items():
        _load_checkpoint(
            model,
            project_root / "results" / "forecasting" / "checkpoints" / f"{name}.pt",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AGCConfig()
    if args.compartment is not None:
        cfg.control_compartment = args.compartment
    if args.steps is not None:
        cfg.control_eval_steps = args.steps
    if args.start_idx is not None:
        cfg.control_start_idx = args.start_idx
    if args.reference_mode is not None:
        cfg.control_reference_mode = args.reference_mode

    run_control_benchmarks(cfg)


if __name__ == "__main__":
    main()
