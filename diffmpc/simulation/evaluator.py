# -*- coding: utf-8 -*-
"""
simulation/evaluator.py
-------------------------
预测大脑 (Transformer) 独立离线诊断模块

Direct Multi-Step 评估:
  模型一次前传输出 horizon=120 步的全部预测值，
  直接与真实值对比，不做自回归循环，不存在误差累积。
  与旧版 baseline 完全等价的评估方式。
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
        Direct Multi-Step 评估。

        模型一次前传输出 horizon 步预测 → 直接与真实值对比。
        不做自回归循环，不存在误差累积。

        评估指标:
          1. 全 horizon R²/MAE: 对 120 步全部时间点计算
          2. 终值 R²/MAE:      只对第 120 步（最远点）计算
          3. 可视化:            随机抽取样本的完整 120 步预测轨迹

        Returns:
            dict: 包含指标和轨迹数据
        """
        self.model.eval()
        horizon = self.config.horizon
        
        # ============================================================
        # 批量前传推理 (一次性输出 horizon 步)
        # ============================================================
        batch_size = 128
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch_start in range(0, len(X_p), batch_size):
                batch_end = min(batch_start + batch_size, len(X_p))
                
                bx_p = torch.tensor(X_p[batch_start:batch_end], dtype=torch.float32, device=self.device)
                bx_f = torch.tensor(X_f[batch_start:batch_end], dtype=torch.float32, device=self.device)
                
                pred = self.model(bx_p, bx_f)  # (batch, horizon, 3)
                all_preds.append(pred.cpu().numpy())
                all_trues.append(y_true[batch_start:batch_end])
        
        all_preds = np.concatenate(all_preds, axis=0)  # (N, horizon, 3)
        all_trues = np.concatenate(all_trues, axis=0)   # (N, horizon, 3)
        
        # 反归一化
        all_preds_real = self._inverse_transform_targets(all_preds)
        all_trues_real = self._inverse_transform_targets(all_trues)
        
        # ============================================================
        # 指标 1: 全 horizon 平均 R²/MAE
        # 将 (N, horizon, 3) 展平为 (N*horizon, 3)
        # ============================================================
        N = all_preds_real.shape[0]
        preds_flat = all_preds_real.reshape(-1, 3)
        trues_flat = all_trues_real.reshape(-1, 3)
        
        r2_full = [r2_score(trues_flat[:, i], preds_flat[:, i]) for i in range(3)]
        mae_full = [mean_absolute_error(trues_flat[:, i], preds_flat[:, i]) for i in range(3)]
        
        # ============================================================
        # 指标 2: 终值 R²/MAE (只看最后一步 t=horizon-1)
        # ============================================================
        preds_final = all_preds_real[:, -1, :]  # (N, 3)
        trues_final = all_trues_real[:, -1, :]  # (N, 3)
        
        r2_final = [r2_score(trues_final[:, i], preds_final[:, i]) for i in range(3)]
        mae_final = [mean_absolute_error(trues_final[:, i], preds_final[:, i]) for i in range(3)]
        
        # ============================================================
        # 可视化数据: 从测试集中间抽取一个样本，展示完整 horizon 步轨迹
        # ============================================================
        plot_idx = len(X_p) // 2
        plot_pred = all_preds_real[plot_idx]  # (horizon, 3)
        plot_true = all_trues_real[plot_idx]  # (horizon, 3)
        
        # ============================================================
        # 打印诊断
        # ============================================================
        labels = ['🌡️ Temperature', '💧 Humidity   ', '☁️ CO2        ']
        units  = ['°C', '%', 'ppm']
        
        print("\n" + "=" * 75)
        print(f"💡 [诊断] Transformer 预测大脑 — Direct Multi-Step 评估 (horizon={horizon})")
        print("-" * 75)
        print(f"  {'变量':<16} | {'全程 R²':>10} {'MAE':>8} | {'终值(t={}) R²'.format(horizon):>14} {'MAE':>8}")
        print("-" * 75)
        for i in range(3):
            print(f"  {labels[i]} | {r2_full[i]:>10.4f} {mae_full[i]:>7.2f}{units[i]} | {r2_final[i]:>14.4f} {mae_final[i]:>7.2f}{units[i]}")
        print("=" * 75)
        print(f"  测试样本数: {N}, 每样本预测 {horizon} 步, 共 {N*horizon} 个预测点\n")

        return {
            # 全 horizon 指标
            'r2_full': r2_full,
            'mae_full': mae_full,
            # 终值指标
            'r2_final': r2_final,
            'mae_final': mae_final,
            # 可视化数据 (单样本完整轨迹)
            'plot_true_ar': plot_true,    # (horizon, 3) - 键名保持兼容 visualizer
            'plot_pred_ar': plot_pred,    # (horizon, 3)
            'plot_steps': horizon,
        }
