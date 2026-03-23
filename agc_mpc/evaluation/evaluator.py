# -*- coding: utf-8 -*-
"""Offline evaluation utilities for AGC forecasting models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score


class ForecasterEvaluator:
    """Evaluate multi-step forecasting models on standardized AGC bundles."""

    def __init__(self, model, y_scaler, target_cols, device=None):
        self.model = model
        self.y_scaler = y_scaler
        self.target_cols = target_cols
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _inverse_targets(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        inv = self.y_scaler.inverse_transform(flat)
        return inv.reshape(shape)

    def _plot_prediction_examples(
        self,
        pred_real: np.ndarray,
        true_real: np.ndarray,
        output_dir: Path,
        model_name: str,
        num_examples: int = 3,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        n_examples = min(num_examples, len(pred_real))
        path = output_dir / f"{model_name}_forecast_examples.png"
        if n_examples <= 0:
            return path

        indices = np.linspace(0, len(pred_real) - 1, num=n_examples, dtype=int)
        fig, axes = plt.subplots(
            len(self.target_cols),
            n_examples,
            figsize=(5 * n_examples, 3.5 * len(self.target_cols)),
            squeeze=False,
        )
        horizon = np.arange(1, pred_real.shape[1] + 1)

        for col_idx, target in enumerate(self.target_cols):
            for ex_idx, sample_idx in enumerate(indices):
                ax = axes[col_idx][ex_idx]
                ax.plot(horizon, true_real[sample_idx, :, col_idx], label="True", linewidth=2.0)
                ax.plot(horizon, pred_real[sample_idx, :, col_idx], label="Pred", linewidth=2.0, linestyle="--")
                ax.set_title(f"{target} | sample {sample_idx}")
                ax.set_xlabel("Forecast step")
                ax.set_ylabel(target)
                ax.grid(True, alpha=0.25)
                if col_idx == 0 and ex_idx == 0:
                    ax.legend()

        fig.suptitle(f"{model_name} forecast examples", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_horizon_mae(
        self,
        pred_real: np.ndarray,
        true_real: np.ndarray,
        output_dir: Path,
        model_name: str,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_horizon_mae.png"
        horizon = np.arange(1, pred_real.shape[1] + 1)
        mae_curve = np.mean(np.abs(pred_real - true_real), axis=0)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for col_idx, target in enumerate(self.target_cols):
            ax.plot(horizon, mae_curve[:, col_idx], label=target, linewidth=2.0)
        ax.set_title(f"{model_name} horizon-wise MAE")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def evaluate(self, X_past, W_future, U_future, Y_future, model_name="model", output_dir=None, num_plot_examples=3):
        self.model.eval()
        preds = []
        batch_size = 512
        with torch.no_grad():
            for start in range(0, len(X_past), batch_size):
                end = min(start + batch_size, len(X_past))
                xb = torch.from_numpy(X_past[start:end]).float().to(self.device)
                wb = torch.from_numpy(W_future[start:end]).float().to(self.device)
                ub = torch.from_numpy(U_future[start:end]).float().to(self.device)
                pred = self.model(xb, wb, ub).cpu().numpy()
                preds.append(pred)

        pred_scaled = np.concatenate(preds, axis=0)
        true_scaled = Y_future

        pred_real = self._inverse_targets(pred_scaled)
        true_real = self._inverse_targets(true_scaled)

        flat_pred = pred_real.reshape(-1, pred_real.shape[-1])
        flat_true = true_real.reshape(-1, true_real.shape[-1])
        final_pred = pred_real[:, -1, :]
        final_true = true_real[:, -1, :]

        metrics = {
            "full_mae": [],
            "full_r2": [],
            "final_mae": [],
            "final_r2": [],
        }
        for idx, name in enumerate(self.target_cols):
            metrics["full_mae"].append(mean_absolute_error(flat_true[:, idx], flat_pred[:, idx]))
            metrics["full_r2"].append(r2_score(flat_true[:, idx], flat_pred[:, idx]))
            metrics["final_mae"].append(mean_absolute_error(final_true[:, idx], final_pred[:, idx]))
            metrics["final_r2"].append(r2_score(final_true[:, idx], final_pred[:, idx]))

        print("\n" + "=" * 80)
        print("Offline Forecast Evaluation")
        print("-" * 80)
        print(f"{'Variable':<12} | {'Full R2':>8} {'Full MAE':>12} | {'Final R2':>8} {'Final MAE':>12}")
        print("-" * 80)
        for idx, name in enumerate(self.target_cols):
            print(
                f"{name:<12} | "
                f"{metrics['full_r2'][idx]:>8.4f} {metrics['full_mae'][idx]:>12.3f} | "
                f"{metrics['final_r2'][idx]:>8.4f} {metrics['final_mae'][idx]:>12.3f}"
            )
        print("=" * 80)

        output_path = Path(output_dir) if output_dir else Path("results/figures")
        forecast_plot = self._plot_prediction_examples(
            pred_real,
            true_real,
            output_path,
            model_name,
            num_examples=num_plot_examples,
        )
        mae_plot = self._plot_horizon_mae(pred_real, true_real, output_path, model_name)
        print(f"Saved forecast figure: {forecast_plot}")
        print(f"Saved horizon MAE figure: {mae_plot}")

        return {
            "metrics": metrics,
            "pred_real": pred_real,
            "true_real": true_real,
            "figure_paths": {
                "forecast_examples": str(forecast_plot),
                "horizon_mae": str(mae_plot),
            },
        }
