# -*- coding: utf-8 -*-
"""Entry point for the new AGC-based predictive control project."""

import random
import os
from pathlib import Path

import numpy as np
import torch

from config import AGCConfig
from data_processing.processor import AGCDataProcessor
from evaluation.evaluator import ForecasterEvaluator
from models.dlinear_forecaster import ConditionalDLinearForecaster
from models.gru_forecaster import ConditionalGRUForecaster
from models.seg_rnn_forecaster import ConditionalSegRNNForecaster
from models.transformer_forecaster import ConditionalTransformerForecaster
from models.transformer_hybrid_forecaster import ConditionalTransformerHybridForecaster
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


def run_baseline(name, model, bundle, cfg, device):
    print("\n" + "=" * 72)
    print(f"Running baseline: {name}")
    print("=" * 72)
    cfg.model_save_path = f"{cfg.forecast_checkpoints_dir}/{name}.pt"

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

    evaluator = ForecasterEvaluator(
        model,
        x_scaler=bundle["scalers"]["x"],
        y_scaler=bundle["scalers"]["y"],
        target_cols=bundle["feature_groups"]["y_future"],
        past_feature_cols=bundle["feature_groups"]["x_past"],
        device=device,
    )
    return evaluator.evaluate(
        bundle["X_past_test"],
        bundle["W_future_test"],
        bundle["U_future_test"],
        bundle["Y_future_test"],
        model_name=name,
        output_dir=cfg.forecast_figures_dir,
        num_plot_examples=cfg.plot_examples,
        plot_history_steps=cfg.plot_history_steps,
        forecast_rollout_examples=cfg.forecast_rollout_examples,
        forecast_rollout_steps=cfg.forecast_rollout_steps,
        forecast_rollout_stride=cfg.forecast_rollout_stride,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    print("=" * 72)
    print("AGC Predictive Control Project")
    print("=" * 72)
    print(f"project_root: {project_root}")

    cfg = AGCConfig()
    _set_global_seed(cfg.seed)
    ensure_results_layout(cfg)
    processor = AGCDataProcessor(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if len(cfg.selected_compartments) == 1:
        compartment = cfg.selected_compartments[0]
        print(f"---> Building dataset bundle for compartment: {compartment}")
        bundle = processor.build_compartment_bundle(compartment)
    else:
        print(f"---> Building merged dataset bundle for: {cfg.selected_compartments}")
        bundle = processor.build_multi_compartment_bundle()

    print("---> Feature groups")
    for name, cols in bundle["feature_groups"].items():
        print(f"{name}: {len(cols)} cols")
        print(f"    {cols}")

    print("---> Tensor shapes")
    for line in processor.summarize_bundle(bundle):
        print(f"    {line}")

    gru_model = ConditionalGRUForecaster(
        past_dim=bundle["X_past_train"].shape[-1],
        weather_dim=bundle["W_future_train"].shape[-1],
        control_dim=bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )
    dlinear_model = ConditionalDLinearForecaster(
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        past_dim=bundle["X_past_train"].shape[-1],
        weather_dim=bundle["W_future_train"].shape[-1],
        control_dim=bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        hidden_dim=cfg.hidden_dim,
    )
    segrnn_model = ConditionalSegRNNForecaster(
        seq_len=cfg.seq_len,
        past_dim=bundle["X_past_train"].shape[-1],
        weather_dim=bundle["W_future_train"].shape[-1],
        control_dim=bundle["U_future_train"].shape[-1],
        target_dim=bundle["Y_future_train"].shape[-1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        seg_len=cfg.seg_len,
    )
    transformer_model = ConditionalTransformerHybridForecaster(
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
    pure_transformer_model = ConditionalTransformerForecaster(
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

    results = {}
    results["gru_baseline"] = run_baseline("gru_baseline", gru_model, bundle, cfg, device)
    results["dlinear_baseline"] = run_baseline("dlinear_baseline", dlinear_model, bundle, cfg, device)
    results["segrnn_baseline"] = run_baseline("segrnn_baseline", segrnn_model, bundle, cfg, device)
    results["transformer_baseline"] = run_baseline(
        "transformer_baseline",
        pure_transformer_model,
        bundle,
        cfg,
        device,
    )
    results["transformer_hybrid_baseline"] = run_baseline(
        "transformer_hybrid_baseline",
        transformer_model,
        bundle,
        cfg,
        device,
    )

    print("\n" + "=" * 72)
    print("Baseline summary")
    print("=" * 72)
    for model_name, result in results.items():
        print(f"{model_name}:")
        for idx, target in enumerate(bundle["feature_groups"]["y_future"]):
            print(
                f"    {target:<10} | "
                f"Full R2={result['metrics']['full_r2'][idx]:.4f} "
                f"MAE={result['metrics']['full_mae'][idx]:.3f} | "
                f"Final R2={result['metrics']['final_r2'][idx]:.4f} "
                f"MAE={result['metrics']['final_mae'][idx]:.3f}"
            )
        print(f"    forecast_examples: {result['figure_paths']['forecast_examples']}")
        print(f"    horizon_mae:      {result['figure_paths']['horizon_mae']}")
        print(f"    forecast_rollout: {result['figure_paths']['forecast_rollout']}")
        print(f"    first_step_rollout: {result['figure_paths']['forecast_first_step_rollout']}")
        print(f"    forecast_heatmap: {result['figure_paths']['forecast_error_heatmap']}")

    print("---> Done.")


if __name__ == "__main__":
    main()
