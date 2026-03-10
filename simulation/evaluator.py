# -*- coding: utf-8 -*-
"""
simulation/evaluator.py
-------------------------
预测大脑 (Transformer) 独立离线诊断模块

双模式评估:
  - Panel A (Teacher Forcing): 每步都给真实历史，只预测下一步 → 公平的泛化 R²
  - Panel B (Autoregressive):  切断真实目标，模型自回归盲推 150 步 → 压力测试
"""

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error


class PredictorEvaluator:
    def __init__(self, model, scaler, target_indices, feature_order, config, device=None):
        self.model = model
        self.scaler = scaler
        self.target_indices = target_indices
        self.feature_order = feature_order
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _inverse_transform_targets(self, targets_norm):
        """反归一化目标变量"""
        orig_shape = targets_norm.shape
        if len(orig_shape) == 3:
            B, S, D = orig_shape
            flat_targets = targets_norm.reshape(-1, D)
        else:
            flat_targets = targets_norm
            
        dummy = np.zeros((len(flat_targets), len(self.scaler.scale_)))
        for i, idx in enumerate(self.target_indices):
            dummy[:, idx] = flat_targets[:, i]
            
        inv = self.scaler.inverse_transform(dummy)[:, self.target_indices]
        if len(orig_shape) == 3:
            return inv.reshape(B, S, D)
        return inv

    def evaluate(self, X_p, X_f, y_true):
        """
        双模式综合评估。

        Returns:
            dict: 包含两种模式的指标和轨迹数据
        """
        self.model.eval()
        
        plot_steps = min(150, len(X_p))
        start_idx = len(X_p) // 2 - plot_steps // 2 if len(X_p) > plot_steps else 0
        
        # ============================================================
        # Panel A: Teacher Forcing 单步滑动评估 (公平泛化指标)
        # 每步都用真实历史，只取 horizon 第 0 步预测
        # ============================================================
        y_true_real_norm_a = y_true[start_idx : start_idx + plot_steps, 0, :]
        y_true_real_a = self._inverse_transform_targets(y_true_real_norm_a)
        
        y_preds_norm_a = []
        
        with torch.no_grad():
            # 批量推理提速: 一次性送入所有样本
            batch_size = 256
            for batch_start in range(0, plot_steps, batch_size):
                batch_end = min(batch_start + batch_size, plot_steps)
                idx_range = range(start_idx + batch_start, start_idx + batch_end)
                
                bx_p = torch.tensor(X_p[list(idx_range)], dtype=torch.float32, device=self.device)
                bx_f = torch.tensor(X_f[list(idx_range)], dtype=torch.float32, device=self.device)
                
                pred = self.model(bx_p, bx_f)  # (batch, horizon, 3)
                # 取 horizon 第 0 步
                step_preds = pred[:, 0, :].cpu().numpy()
                y_preds_norm_a.append(step_preds)
        
        y_preds_norm_a = np.concatenate(y_preds_norm_a, axis=0)
        y_pred_real_a = self._inverse_transform_targets(y_preds_norm_a)
        
        # Panel A 指标
        r2_a = [r2_score(y_true_real_a[:, i], y_pred_real_a[:, i]) for i in range(3)]
        mae_a = [mean_absolute_error(y_true_real_a[:, i], y_pred_real_a[:, i]) for i in range(3)]
        
        # ============================================================
        # Panel B: 自回归滚动评估 (压力测试)
        # 给定初始历史，切断真实目标，模型用自己的预测拼接历史
        # ============================================================
        y_true_real_norm_b = y_true[start_idx : start_idx + plot_steps, 0, :]
        y_true_real_b = self._inverse_transform_targets(y_true_real_norm_b)
        
        current_past_seq = X_p[start_idx].copy()
        y_preds_norm_b = []
        
        with torch.no_grad():
            for t in range(plot_steps):
                curr_future = X_f[start_idx + t]
                
                bx_p = torch.tensor(current_past_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                bx_f = torch.tensor(curr_future, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                pred = self.model(bx_p, bx_f)
                step_pred_norm = pred[0, 0, :].cpu().numpy()
                y_preds_norm_b.append(step_pred_norm)
                
                # 自回归拼接: 抛弃最老一步，用模型预测替换目标维度
                new_step = np.zeros(current_past_seq.shape[1])
                if start_idx + t + 1 < len(X_p):
                    real_next_features = X_p[start_idx + t + 1][-1, :].copy()
                else:
                    real_next_features = current_past_seq[-1, :].copy()
                    
                for i, idx in enumerate(self.target_indices):
                    real_next_features[idx] = step_pred_norm[i]
                    
                current_past_seq = np.vstack([current_past_seq[1:, :], real_next_features])
        
        y_preds_norm_b = np.array(y_preds_norm_b)
        y_pred_real_b = self._inverse_transform_targets(y_preds_norm_b)
        
        # Panel B 指标
        r2_b = [r2_score(y_true_real_b[:, i], y_pred_real_b[:, i]) for i in range(3)]
        mae_b = [mean_absolute_error(y_true_real_b[:, i], y_pred_real_b[:, i]) for i in range(3)]
        
        # ============================================================
        # 打印对比
        # ============================================================
        labels = ['🌡️ Temperature', '💧 Humidity   ', '☁️ CO2        ']
        units  = ['°C', '%', 'ppm']
        
        print("\n" + "=" * 65)
        print("💡 [诊断] Transformer 预测大脑 — 双模式评估")
        print("-" * 65)
        print(f"  {'变量':<16} | {'Panel A (TF) R²':>14} {'MAE':>8} | {'Panel B (AR) R²':>14} {'MAE':>8}")
        print("-" * 65)
        for i in range(3):
            print(f"  {labels[i]} | {r2_a[i]:>14.4f} {mae_a[i]:>7.2f}{units[i]} | {r2_b[i]:>14.4f} {mae_b[i]:>7.2f}{units[i]}")
        print("=" * 65 + "\n")

        return {
            # Panel A: Teacher Forcing
            'r2_tf': r2_a,
            'mae_tf': mae_a,
            'plot_true_tf': y_true_real_a,
            'plot_pred_tf': y_pred_real_a,
            # Panel B: Autoregressive
            'r2_ar': r2_b,
            'mae_ar': mae_b,
            'plot_true_ar': y_true_real_b,
            'plot_pred_ar': y_pred_real_b,
            # 共享
            'plot_steps': plot_steps,
        }
