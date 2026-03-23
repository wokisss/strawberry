# -*- coding: utf-8 -*-
"""Training loop for AGC forecasting baselines."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    """Simple trainer with early stopping on validation loss."""

    def __init__(self, model, config, device=None):
        self.model = model
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, X_past_train, W_future_train, U_future_train, Y_future_train,
              X_past_val, W_future_val, U_future_val, Y_future_val):
        train_dataset = TensorDataset(
            torch.from_numpy(X_past_train).float(),
            torch.from_numpy(W_future_train).float(),
            torch.from_numpy(U_future_train).float(),
            torch.from_numpy(Y_future_train).float(),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_past_val).float(),
            torch.from_numpy(W_future_val).float(),
            torch.from_numpy(U_future_val).float(),
            torch.from_numpy(Y_future_val).float(),
        )

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        save_path = Path(self.cfg.model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        best_epoch = 0
        epochs_no_improve = 0

        self.model.to(self.device)
        print(
            f"---> Start training baseline "
            f"(epochs={self.cfg.num_epochs}, batch_size={self.cfg.batch_size}, lr={self.cfg.learning_rate})"
        )

        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            train_loss = 0.0

            for xb, wb, ub, yb in train_loader:
                xb = xb.to(self.device)
                wb = wb.to(self.device)
                ub = ub.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb, wb, ub)
                loss_mse = criterion(pred, yb)
                if pred.shape[1] > 1:
                    pred_diff = pred[:, 1:] - pred[:, :-1]
                    true_diff = yb[:, 1:] - yb[:, :-1]
                    loss_trend = criterion(pred_diff, true_diff)
                else:
                    loss_trend = 0.0
                loss = loss_mse + self.cfg.lambda_trend * loss_trend
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train = train_loss / max(len(train_loader), 1)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, wb, ub, yb in val_loader:
                    xb = xb.to(self.device)
                    wb = wb.to(self.device)
                    ub = ub.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb, wb, ub)
                    val_loss += criterion(pred, yb).item()

            avg_val = val_loss / max(len(val_loader), 1)
            improved = avg_val < best_val
            if improved:
                best_val = avg_val
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                epochs_no_improve += 1

            marker = " *" if improved else ""
            print(
                f"    Epoch {epoch + 1:2d}/{self.cfg.num_epochs} | "
                f"train={avg_train:.5f} | val={avg_val:.5f} | best={best_val:.5f}{marker}"
            )

            if epochs_no_improve >= self.cfg.early_stop_patience:
                print(f"    [Early Stop] no validation improvement for {self.cfg.early_stop_patience} epochs.")
                break

        state = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"---> Restored best checkpoint from epoch {best_epoch}.")
        return self.model

