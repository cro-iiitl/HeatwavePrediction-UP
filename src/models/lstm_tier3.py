"""Tier 3 — LSTM, direct multi-step (master §6.5)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LSTMTier3Net(nn.Module):
    def __init__(self, n_features, hidden_units=64, dropout=0.2,
                 dense_units=32, horizon_days=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_units,
                             num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(hidden_units, dense_units)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(dense_units, horizon_days)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n.squeeze(0)
        h = self.dropout(h)
        h = self.relu(self.dense1(h))
        return self.dense2(h)


class LSTMForecastModel:
    def __init__(self, n_features, hidden_units=64, dropout=0.2, dense_units=32,
                 horizon_days=10, lr=1e-3, batch_size=128, max_epochs=30,
                 early_stopping_patience=15, eval_batch_size=1024, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = LSTMTier3Net(n_features, hidden_units, dropout,
                                 dense_units, horizon_days).to(self.device)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.lr = lr
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None

    def _fit_scalers(self, X_train, y_train):
        self.feature_mean = X_train.mean(axis=(0, 1), keepdims=True)
        self.feature_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        self.target_mean = y_train.mean()
        self.target_std = y_train.std() + 1e-8

    def _scale_X(self, X):
        return (X - self.feature_mean) / self.feature_std

    def _scale_y(self, y):
        return (y - self.target_mean) / self.target_std

    def _unscale_y(self, y_scaled):
        return y_scaled * self.target_std + self.target_mean

    def _batched_forward(self, X_scaled: np.ndarray) -> np.ndarray:
        self.net.eval()
        outputs = []
        n = len(X_scaled)
        with torch.no_grad():
            for start in range(0, n, self.eval_batch_size):
                end = min(start + self.eval_batch_size, n)
                xb = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self.device)
                out = self.net(xb).cpu().numpy()
                outputs.append(out)
        return np.concatenate(outputs, axis=0)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            train_dates=None, log_fn=None) -> None:
        # train_dates accepted for interface consistency, unused by LSTM
        self._fit_scalers(X_train, y_train)
        X_train_s = self._scale_X(X_train)
        y_train_s = self._scale_y(y_train)

        train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32),
                                  torch.tensor(y_train_s, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val_s = self._scale_X(X_val)
            y_val_s = self._scale_y(y_val)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_ds)

            val_loss = None
            if use_val:
                val_pred_s = self._batched_forward(X_val_s)
                val_loss = float(np.mean((val_pred_s - y_val_s) ** 2))

                if log_fn:
                    log_fn({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        break
            elif log_fn:
                log_fn({"epoch": epoch, "train_loss": epoch_loss})

        if use_val and best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, X, target_dates=None) -> np.ndarray:
        # target_dates accepted for interface consistency, unused by LSTM
        X_s = self._scale_X(X)
        pred_scaled = self._batched_forward(X_s)
        return self._unscale_y(pred_scaled)

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.net.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }, path)

    @classmethod
    def load(cls, path: str, n_features: int, **kwargs) -> "LSTMForecastModel":
        checkpoint = torch.load(path)
        model = cls(n_features=n_features, **kwargs)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.feature_mean = checkpoint["feature_mean"]
        model.feature_std = checkpoint["feature_std"]
        model.target_mean = checkpoint["target_mean"]
        model.target_std = checkpoint["target_std"]
        return model