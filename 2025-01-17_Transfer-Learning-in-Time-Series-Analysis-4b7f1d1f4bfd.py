# Description: Short example for Transfer Learning in Time Series Analysis.

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class _LSTMForecaster(nn.Module):
    def __init__(self, n_features: int = 1, hidden: int = 32, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=n_layers, batch_first=True, dropout=0.1 if n_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _train_torch(model, x_train, y_train, *, epochs: int = 10) -> None:
    x_t = torch.FloatTensor(x_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()


def _predict_torch(model, x_test) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(x_test)).numpy()


def create_sequences(data: np.ndarray, seq_length: int) -> np.ndarray:
    return np.array([data[i : i + seq_length] for i in range(len(data) - seq_length)])


def synthetic_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (np.sin(t / 12) + 0.1 * rng.normal(size=n)).reshape(-1, 1)


def evaluate_models(models, test_sequences, test_targets) -> pd.DataFrame:
    rows = {}
    for name, model in models.items():
        preds = _predict_torch(model, test_sequences)
        mse = float(np.mean((test_targets - preds.squeeze()) ** 2))
        mae = float(np.mean(np.abs(test_targets - preds.squeeze())))
        rows[name] = {"MSE": mse, "MAE": mae}
    return pd.DataFrame(rows).T


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    seq_length = 24

    source_scaled = MinMaxScaler().fit_transform(synthetic_series(300, 1))
    target_scaled = MinMaxScaler().fit_transform(synthetic_series(300, 2))
    source_sequences = create_sequences(source_scaled, seq_length)
    target_sequences = create_sequences(target_scaled, seq_length)

    source_model = _LSTMForecaster()
    _train_torch(source_model, source_sequences, source_scaled[seq_length:], epochs=5)

    transfer_model = _LSTMForecaster()
    transfer_model.lstm.load_state_dict(source_model.lstm.state_dict())
    for param in transfer_model.lstm.parameters():
        param.requires_grad = False
    _train_torch(transfer_model, target_sequences, target_scaled[seq_length:], epochs=5)

    fine_tune_model = _LSTMForecaster()
    fine_tune_model.load_state_dict(transfer_model.state_dict())
    for param in fine_tune_model.parameters():
        param.requires_grad = True
    _train_torch(fine_tune_model, target_sequences, target_scaled[seq_length:], epochs=5)

    test_x = target_sequences[-50:]
    test_y = target_scaled[seq_length:][-50:].squeeze()
    models = {
        "Base Model": source_model,
        "Transfer Learning": transfer_model,
        "Fine-tuned": fine_tune_model,
    }
    results = evaluate_models(models, test_x, test_y)
    logger.info("Model comparison:\n%s", results)


if __name__ == "__main__":
    main()
