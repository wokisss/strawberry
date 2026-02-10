# -*- coding: utf-8 -*-
"""
training/trainer.py
---------------------
模型训练逻辑

包含带趋势惩罚的训练循环和模型评估。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    """
    模型训练器

    Args:
        model: 预测模型 (SegmentedHybridModel)
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
        else:
            self._lr = 0.001
            self._epochs = 30
            self._batch_size = 256
            self._lambda_trend = 0.3

    def train(self, X_train_p, X_train_f, y_train):
        """
        训练模型

        Args:
            X_train_p: 训练集历史序列 (numpy)
            X_train_f: 训练集未来序列 (numpy)
            y_train:   训练集标签 (numpy)

        Returns:
            训练好的模型
        """
        # 构建 DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(X_train_p),
            torch.FloatTensor(X_train_f),
            torch.FloatTensor(y_train)
        )
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

        print(f"---> 开始训练 (epochs={self._epochs}, lambda_trend={self._lambda_trend}, device={self.device})...")
        self.model.to(self.device)

        for epoch in range(self._epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_p, batch_f, batch_y in loader:
                batch_p = batch_p.to(self.device)
                batch_f = batch_f.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_p, batch_f)

                # MSE Loss
                loss_mse = criterion(outputs, batch_y)

                # 趋势惩罚 (Smoothness Penalty)
                if outputs.shape[1] > 1:
                    diff = outputs[:, 1:] - outputs[:, :-1]
                    loss_smooth = torch.mean(diff ** 2)
                else:
                    loss_smooth = 0.0

                loss = loss_mse + self._lambda_trend * loss_smooth
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"    Epoch {epoch + 1}, Loss: {avg_loss:.5f}")

        self.model.eval()
        print("---> 模型训练完成。")
        return self.model
