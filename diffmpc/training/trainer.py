# -*- coding: utf-8 -*-
"""
training/trainer.py
---------------------
模型训练逻辑 (含 Early Stopping + 趋势差分匹配 + 最优权重管理)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    """
    模型训练器

    特性:
        - 自动从训练集中划出 20% 作为验证集
        - Early Stopping: 验证集 loss 连续 patience 轮不下降自动停止
        - 趋势惩罚使用目标差分匹配 (而非简单的平滑度惩罚)
        - 训练结束后自动回滚到验证集上最优的权重

    Args:
        model: 预测模型
        config: Config 对象 (可选)
        device: 计算设备
    """

    def __init__(self, model, config=None, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            self._lr = config.learning_rate
            self._epochs = config.num_epochs
            self._batch_size = config.batch_size
            self._lambda_trend = config.lambda_trend
            self._patience = getattr(config, 'early_stop_patience', 15)
            self._model_save_path = getattr(config, 'model_save_path', 'best_model_A2.pth')
        else:
            self._lr = 0.0001
            self._epochs = 200
            self._batch_size = 256
            self._lambda_trend = 0.3
            self._patience = 15
            self._model_save_path = 'best_model_A2.pth'

    def train(self, X_train_p, X_train_f, y_train):
        """
        训练模型 (含验证集划分 + Early Stopping)

        Args:
            X_train_p: 训练集历史序列 (numpy)
            X_train_f: 训练集未来序列 (numpy)
            y_train:   训练集标签 (numpy)

        Returns:
            训练好的模型 (已回滚到最优权重)
        """
        # ==================== 验证集划分 ====================
        n = len(X_train_p)
        val_ratio = 0.2
        val_size = int(n * val_ratio)
        train_size = n - val_size

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_p[:train_size]),
            torch.FloatTensor(X_train_f[:train_size]),
            torch.FloatTensor(y_train[:train_size])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_train_p[train_size:]),
            torch.FloatTensor(X_train_f[train_size:]),
            torch.FloatTensor(y_train[train_size:])
        )

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

        print(f"---> 开始训练 (max_epochs={self._epochs}, lr={self._lr}, "
              f"lambda_trend={self._lambda_trend}, patience={self._patience})")
        print(f"     训练集: {train_size} 样本, 验证集: {val_size} 样本, 设备: {self.device}")

        self.model.to(self.device)

        # ==================== Early Stopping 状态 ====================
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(self._epochs):
            # ---------- 训练阶段 ----------
            self.model.train()
            epoch_loss = 0.0

            for batch_p, batch_f, batch_y in train_loader:
                batch_p = batch_p.to(self.device)
                batch_f = batch_f.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_p, batch_f)

                # MSE Loss
                loss_mse = criterion(outputs, batch_y)

                # [修正] 趋势惩罚: 目标差分匹配 (而非简单平滑度)
                # 要求预测的变化方向与真实目标的变化方向一致
                if outputs.shape[1] > 1:
                    pred_diff = outputs[:, 1:] - outputs[:, :-1]
                    target_diff = batch_y[:, 1:] - batch_y[:, :-1]
                    loss_trend = criterion(pred_diff, target_diff)
                else:
                    loss_trend = 0.0

                loss = loss_mse + self._lambda_trend * loss_trend
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)

            # ---------- 验证阶段 ----------
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_p, batch_f, batch_y in val_loader:
                    batch_p = batch_p.to(self.device)
                    batch_f = batch_f.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_p, batch_f)
                    val_loss += criterion(outputs, batch_y).item()

            avg_val_loss = val_loss / len(val_loader)

            # ---------- Early Stopping 检查 ----------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
                # 保存最优权重
                torch.save(self.model.state_dict(), self._model_save_path)
            else:
                epochs_no_improve += 1

            # 打印日志 (每10轮 + 首轮 + 末轮 + 改善时)
            if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_no_improve == 0:
                marker = " ★" if epochs_no_improve == 0 else ""
                print(f"    Epoch {epoch + 1:3d}/{self._epochs} | "
                      f"Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | "
                      f"Best: {best_val_loss:.5f} (ep{best_epoch}){marker}")

            if epochs_no_improve >= self._patience:
                print(f"    [Early Stop] 验证集 loss 连续 {self._patience} 轮未改善，在 Epoch {epoch + 1} 停止")
                break

        # ==================== 回滚到最优权重 ====================
        if os.path.exists(self._model_save_path):
            self.model.load_state_dict(torch.load(self._model_save_path, weights_only=True))
            print(f"---> 已回滚到最优权重 (Epoch {best_epoch}, Val Loss: {best_val_loss:.5f})")
        
        self.model.eval()
        print("---> 模型训练完成。")
        return self.model
