# -*- coding: utf-8 -*-
"""Offline predictor diagnostics for the Transformer model."""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score


class PredictorEvaluator:
    def __init__(self, model, scaler, target_indices, feature_order, config, device=None):
        self.model = model
        self.scaler = scaler
        self.target_indices = target_indices
        self.feature_order = feature_order
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _inverse_transform_targets(self, targets_norm):
        orig_shape = targets_norm.shape
        if len(orig_shape) == 3:
            batch, steps, dims = orig_shape
            flat_targets = targets_norm.reshape(-1, dims)
        else:
            flat_targets = targets_norm

        dummy = np.zeros((len(flat_targets), len(self.scaler.scale_)))
        for i, idx in enumerate(self.target_indices):
            dummy[:, idx] = flat_targets[:, i]

        inv = self.scaler.inverse_transform(dummy)[:, self.target_indices]
        if len(orig_shape) == 3:
            return inv.reshape(batch, steps, dims)
        return inv

    def evaluate(self, X_p, X_f, y_true):
        if len(X_p) == 0 or len(X_f) == 0 or len(y_true) == 0:
            raise ValueError("Predictor evaluation requires a non-empty test split.")

        self.model.eval()
        horizon = self.config.horizon
        batch_size = 128
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch_start in range(0, len(X_p), batch_size):
                batch_end = min(batch_start + batch_size, len(X_p))
                bx_p = torch.tensor(X_p[batch_start:batch_end], dtype=torch.float32, device=self.device)
                bx_f = torch.tensor(X_f[batch_start:batch_end], dtype=torch.float32, device=self.device)
                pred = self.model(bx_p, bx_f)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(y_true[batch_start:batch_end])

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        all_preds_real = self._inverse_transform_targets(all_preds)
        all_trues_real = self._inverse_transform_targets(all_trues)

        sample_count = all_preds_real.shape[0]
        preds_flat = all_preds_real.reshape(-1, 3)
        trues_flat = all_trues_real.reshape(-1, 3)

        r2_full = [r2_score(trues_flat[:, i], preds_flat[:, i]) for i in range(3)]
        mae_full = [mean_absolute_error(trues_flat[:, i], preds_flat[:, i]) for i in range(3)]

        preds_final = all_preds_real[:, -1, :]
        trues_final = all_trues_real[:, -1, :]
        r2_final = [r2_score(trues_final[:, i], preds_final[:, i]) for i in range(3)]
        mae_final = [mean_absolute_error(trues_final[:, i], preds_final[:, i]) for i in range(3)]

        plot_idx = sample_count // 2
        plot_pred = all_preds_real[plot_idx]
        plot_true = all_trues_real[plot_idx]

        labels = ["Temperature", "Humidity", "CO2"]
        units = ["C", "%", "ppm"]

        print("\n" + "=" * 75)
        print(
            f"[Diagnostics] Transformer direct multi-step evaluation "
            f"(horizon={horizon})"
        )
        print("-" * 75)
        print(f"  {'Variable':<12} | {'Full R2':>8} {'MAE':>10} | {'Final R2':>9} {'MAE':>10}")
        print("-" * 75)
        for i in range(3):
            print(
                f"  {labels[i]:<12} | {r2_full[i]:>8.4f} {mae_full[i]:>9.2f}{units[i]} | "
                f"{r2_final[i]:>9.4f} {mae_final[i]:>9.2f}{units[i]}"
            )
        print("=" * 75)
        print(
            f"  test_samples={sample_count}, horizon={horizon}, "
            f"total_points={sample_count * horizon}\n"
        )

        return {
            "r2_full": r2_full,
            "mae_full": mae_full,
            "r2_final": r2_final,
            "mae_final": mae_final,
            "plot_true_ar": plot_true,
            "plot_pred_ar": plot_pred,
            "plot_steps": horizon,
        }
