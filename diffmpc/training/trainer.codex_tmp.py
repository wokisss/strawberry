# -*- coding: utf-8 -*-
"""Training loop with validation split and early stopping."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self, model, config=None, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config is not None:
            self._lr = config.learning_rate
            self._epochs = config.num_epochs
            self._batch_size = config.batch_size
            self._lambda_trend = config.lambda_trend
            self._patience = getattr(config, "early_stop_patience", 15)
            self._model_save_path = getattr(config, "model_save_path", "best_model_A2.pth")
        else:
            self._lr = 0.0001
            self._epochs = 200
            self._batch_size = 256
            self._lambda_trend = 0.3
            self._patience = 15
            self._model_save_path = "best_model_A2.pth"

    def train(self, X_train_p, X_train_f, y_train):
        n = len(X_train_p)
        val_ratio = 0.2
        val_size = int(n * val_ratio)
        train_size = n - val_size

        if n == 0 or train_size <= 0 or val_size <= 0:
            raise ValueError(
                f"Not enough training samples for train/val split: total={n}, "
                f"train={train_size}, val={val_size}"
            )

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_p[:train_size]),
            torch.FloatTensor(X_train_f[:train_size]),
            torch.FloatTensor(y_train[:train_size]),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_train_p[train_size:]),
            torch.FloatTensor(X_train_f[train_size:]),
            torch.FloatTensor(y_train[train_size:]),
        )

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self._lr)

        print(
            f"---> Start training (max_epochs={self._epochs}, lr={self._lr}, "
            f"lambda_trend={self._lambda_trend}, patience={self._patience})"
        )
        print(
            f"     train_samples={train_size}, val_samples={val_size}, device={self.device}"
        )

        self.model.to(self.device)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(self._epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_p, batch_f, batch_y in train_loader:
                batch_p = batch_p.to(self.device)
                batch_f = batch_f.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_p, batch_f)
                loss_mse = criterion(outputs, batch_y)

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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self._model_save_path)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_no_improve == 0:
                marker = " *" if epochs_no_improve == 0 else ""
                print(
                    f"    Epoch {epoch + 1:3d}/{self._epochs} | "
                    f"Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | "
                    f"Best: {best_val_loss:.5f} (ep{best_epoch}){marker}"
                )

            if epochs_no_improve >= self._patience:
                print(
                    f"    [Early Stop] validation loss did not improve for "
                    f"{self._patience} epochs; stop at epoch {epoch + 1}"
                )
                break

        if os.path.exists(self._model_save_path):
            try:
                state_dict = torch.load(
                    self._model_save_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                state_dict = torch.load(self._model_save_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(
                f"---> Restored best checkpoint (epoch {best_epoch}, "
                f"val_loss={best_val_loss:.5f})"
            )

        self.model.eval()
        print("---> Training complete.")
        return self.model
