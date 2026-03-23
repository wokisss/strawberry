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

    def __init__(self, model, x_scaler, y_scaler, target_cols, past_feature_cols=None, device=None):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.target_cols = target_cols
        self.past_feature_cols = past_feature_cols or []
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _inverse_past(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        inv = self.x_scaler.inverse_transform(flat)
        return inv.reshape(shape)

    def _inverse_targets(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        inv = self.y_scaler.inverse_transform(flat)
        return inv.reshape(shape)

    def _plot_prediction_examples(
        self,
        past_real: np.ndarray,
        pred_real: np.ndarray,
        true_real: np.ndarray,
        output_dir: Path,
        model_name: str,
        num_examples: int = 3,
        history_steps: int = 96,
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
        future_steps = np.arange(1, pred_real.shape[1] + 1)

        for col_idx, target in enumerate(self.target_cols):
            past_idx = self.past_feature_cols.index(target) if target in self.past_feature_cols else None
            for ex_idx, sample_idx in enumerate(indices):
                ax = axes[col_idx][ex_idx]
                if past_idx is not None:
                    hist = past_real[sample_idx, :, past_idx]
                    hist = hist[-min(history_steps, len(hist)) :]
                    hist_steps = np.arange(-len(hist) + 1, 1)
                    ax.plot(hist_steps, hist, label="History", linewidth=1.8, color="0.45")
                ax.plot(future_steps, true_real[sample_idx, :, col_idx], label="True", linewidth=2.0)
                ax.plot(future_steps, pred_real[sample_idx, :, col_idx], label="Pred", linewidth=2.0, linestyle="--")
                ax.axvline(0, color="0.25", linewidth=1.0, linestyle=":")
                ax.set_title(f"{target} | sample {sample_idx}")
                ax.set_xlabel("Step relative to t0")
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

    def _plot_forecast_rollout(
        self,
        pred_real: np.ndarray,
        true_real: np.ndarray,
        output_dir: Path,
        model_name: str,
        rollout_examples: int = 2,
        rollout_steps: int = 96,
        rollout_stride: int = 6,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_forecast_rollout.png"
        max_start = len(pred_real) - rollout_steps
        if max_start < 0:
            return path

        n_examples = min(rollout_examples, max_start + 1)
        start_indices = np.linspace(0, max_start, num=n_examples, dtype=int)
        fig, axes = plt.subplots(
            len(self.target_cols),
            n_examples,
            figsize=(6 * n_examples, 3.5 * len(self.target_cols)),
            squeeze=False,
        )
        long_steps = np.arange(1, rollout_steps + 1)

        for col_idx, target in enumerate(self.target_cols):
            for ex_idx, start in enumerate(start_indices):
                ax = axes[col_idx][ex_idx]
                true_series = np.asarray(
                    [true_real[start + offset, 0, col_idx] for offset in range(rollout_steps)],
                    dtype=np.float32,
                )
                ax.plot(long_steps, true_series, label="True timeline", linewidth=2.2, color="black")

                for launch in range(0, rollout_steps, rollout_stride):
                    sample_idx = start + launch
                    pred_window = pred_real[sample_idx, :, col_idx]
                    horizon = min(len(pred_window), rollout_steps - launch)
                    pred_steps = np.arange(launch + 1, launch + horizon + 1)
                    label = "Rolling forecast window" if launch == 0 else None
                    ax.plot(
                        pred_steps,
                        pred_window[:horizon],
                        linewidth=1.5,
                        alpha=0.55,
                        linestyle="--",
                        color="tab:blue",
                        label=label,
                    )

                ax.set_title(f"{target} | start {start}")
                ax.set_xlabel("Timeline step")
                ax.set_ylabel(target)
                ax.grid(True, alpha=0.25)
                if col_idx == 0 and ex_idx == 0:
                    ax.legend()

        fig.suptitle(
            f"{model_name} rolling multi-step forecast windows | stride={rollout_stride}, horizon={pred_real.shape[1]}",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_first_step_rollout(
        self,
        pred_real: np.ndarray,
        true_real: np.ndarray,
        output_dir: Path,
        model_name: str,
        rollout_examples: int = 2,
        rollout_steps: int = 96,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_forecast_first_step_rollout.png"
        max_start = len(pred_real) - rollout_steps
        if max_start < 0:
            return path

        n_examples = min(rollout_examples, max_start + 1)
        start_indices = np.linspace(0, max_start, num=n_examples, dtype=int)
        fig, axes = plt.subplots(
            len(self.target_cols),
            n_examples,
            figsize=(6 * n_examples, 3.5 * len(self.target_cols)),
            squeeze=False,
        )
        timeline_steps = np.arange(1, rollout_steps + 1)

        for col_idx, target in enumerate(self.target_cols):
            for ex_idx, start in enumerate(start_indices):
                ax = axes[col_idx][ex_idx]
                true_series = np.asarray(
                    [true_real[start + offset, 0, col_idx] for offset in range(rollout_steps)],
                    dtype=np.float32,
                )
                pred_series = np.asarray(
                    [pred_real[start + offset, 0, col_idx] for offset in range(rollout_steps)],
                    dtype=np.float32,
                )
                ax.plot(timeline_steps, true_series, label="True timeline", linewidth=2.2, color="black")
                ax.plot(
                    timeline_steps,
                    pred_series,
                    label="First-step rollout",
                    linewidth=2.0,
                    linestyle="--",
                    color="tab:blue",
                )
                ax.set_title(f"{target} | start {start}")
                ax.set_xlabel("Timeline step")
                ax.set_ylabel(target)
                ax.grid(True, alpha=0.25)
                if col_idx == 0 and ex_idx == 0:
                    ax.legend()

        fig.suptitle(f"{model_name} first-step stitched rollout", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_forecast_error_heatmap(
        self,
        pred_real: np.ndarray,
        true_real: np.ndarray,
        output_dir: Path,
        model_name: str,
        rollout_examples: int = 2,
        rollout_steps: int = 96,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_forecast_error_heatmap.png"
        max_start = len(pred_real) - rollout_steps
        if max_start < 0:
            return path

        n_examples = min(rollout_examples, max_start + 1)
        start_indices = np.linspace(0, max_start, num=n_examples, dtype=int)
        horizon = pred_real.shape[1]
        fig, axes = plt.subplots(
            len(self.target_cols),
            n_examples,
            figsize=(6 * n_examples, 3.8 * len(self.target_cols)),
            squeeze=False,
        )

        for col_idx, target in enumerate(self.target_cols):
            for ex_idx, start in enumerate(start_indices):
                ax = axes[col_idx][ex_idx]
                error_grid = np.full((horizon, rollout_steps), np.nan, dtype=np.float32)
                for launch in range(rollout_steps):
                    sample_idx = start + launch
                    error_grid[:, launch] = np.abs(pred_real[sample_idx, :, col_idx] - true_real[sample_idx, :, col_idx])

                im = ax.imshow(
                    error_grid,
                    aspect="auto",
                    origin="lower",
                    cmap="magma",
                    interpolation="nearest",
                )
                ax.set_title(f"{target} | start {start}")
                ax.set_xlabel("Launch step")
                ax.set_ylabel("Forecast horizon")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{model_name} forecast error heatmap", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def evaluate(
        self,
        X_past,
        W_future,
        U_future,
        Y_future,
        model_name="model",
        output_dir=None,
        num_plot_examples=3,
        plot_history_steps=96,
        forecast_rollout_examples=2,
        forecast_rollout_steps=96,
        forecast_rollout_stride=6,
    ):
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

        past_real = self._inverse_past(X_past)
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

        output_path = Path(output_dir) if output_dir else Path("results/forecasting/figures")
        forecast_plot = self._plot_prediction_examples(
            past_real,
            pred_real,
            true_real,
            output_path,
            model_name,
            num_examples=num_plot_examples,
            history_steps=plot_history_steps,
        )
        mae_plot = self._plot_horizon_mae(pred_real, true_real, output_path, model_name)
        rollout_plot = self._plot_forecast_rollout(
            pred_real,
            true_real,
            output_path,
            model_name,
            rollout_examples=forecast_rollout_examples,
            rollout_steps=min(forecast_rollout_steps, len(pred_real)),
            rollout_stride=forecast_rollout_stride,
        )
        first_step_rollout_plot = self._plot_first_step_rollout(
            pred_real,
            true_real,
            output_path,
            model_name,
            rollout_examples=forecast_rollout_examples,
            rollout_steps=min(forecast_rollout_steps, len(pred_real)),
        )
        error_heatmap_plot = self._plot_forecast_error_heatmap(
            pred_real,
            true_real,
            output_path,
            model_name,
            rollout_examples=forecast_rollout_examples,
            rollout_steps=min(forecast_rollout_steps, len(pred_real)),
        )
        print(f"Saved forecast figure: {forecast_plot}")
        print(f"Saved horizon MAE figure: {mae_plot}")
        print(f"Saved forecast rollout figure: {rollout_plot}")
        print(f"Saved first-step rollout figure: {first_step_rollout_plot}")
        print(f"Saved forecast heatmap figure: {error_heatmap_plot}")

        return {
            "metrics": metrics,
            "pred_real": pred_real,
            "true_real": true_real,
            "figure_paths": {
                "forecast_examples": str(forecast_plot),
                "horizon_mae": str(mae_plot),
                "forecast_rollout": str(rollout_plot),
                "forecast_first_step_rollout": str(first_step_rollout_plot),
                "forecast_error_heatmap": str(error_heatmap_plot),
            },
        }
