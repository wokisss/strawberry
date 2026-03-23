# -*- coding: utf-8 -*-
"""Entry point for the new AGC-based predictive control project."""

import os
from pathlib import Path

import torch

from config import AGCConfig
from data_processing.processor import AGCDataProcessor
from evaluation.evaluator import ForecasterEvaluator
from models.dlinear_forecaster import ConditionalDLinearForecaster
from models.gru_forecaster import ConditionalGRUForecaster
from models.seg_rnn_forecaster import ConditionalSegRNNForecaster
from models.transformer_hybrid_forecaster import ConditionalTransformerHybridForecaster
from training.trainer import Trainer


def run_baseline(name, model, bundle, cfg, device):
    print("\n" + "=" * 72)
    print(f"Running baseline: {name}")
    print("=" * 72)
    cfg.model_save_path = f"results/{name}.pt"

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
        y_scaler=bundle["scalers"]["y"],
        target_cols=bundle["feature_groups"]["y_future"],
        device=device,
    )
    return evaluator.evaluate(
        bundle["X_past_test"],
        bundle["W_future_test"],
        bundle["U_future_test"],
        bundle["Y_future_test"],
        model_name=name,
        output_dir=cfg.figures_dir,
        num_plot_examples=cfg.plot_examples,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    print("=" * 72)
    print("AGC Predictive Control Project")
    print("=" * 72)
    print(f"project_root: {project_root}")

    cfg = AGCConfig()
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

    results = {}
    results["gru_baseline"] = run_baseline("gru_baseline", gru_model, bundle, cfg, device)
    results["dlinear_baseline"] = run_baseline("dlinear_baseline", dlinear_model, bundle, cfg, device)
    results["segrnn_baseline"] = run_baseline("segrnn_baseline", segrnn_model, bundle, cfg, device)
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

    print("---> Done.")


if __name__ == "__main__":
    main()
