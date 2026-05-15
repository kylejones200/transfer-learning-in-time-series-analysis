# Description: Short example for Transfer Learning in Time Series Analysis.


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_io import read_csv
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Helper function to create time series sequences
class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""
    def __init__(self, n_features: int, hidden: int = 64, output_size: int = 1,
                 n_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 50, batch_size: int = 32,
                 lr: float = 0.001, validation_split: float = 0.2,
                 patience: int = 15) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : (i + seq_length)])
    return np.array(sequences)


# Load and prepare source domain data (e.g., energy consumption)
source_data = read_csv("energy_consumption.csv")
source_scaler = MinMaxScaler()
source_scaled = source_scaler.fit_transform(source_data[["consumption"]])
source_sequences = create_sequences(source_scaled, seq_length=24)

# Load and prepare target domain data (e.g., solar production)
target_data = read_csv("solar_production.csv")
target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target_data[["production"]])
target_sequences = create_sequences(target_scaled, seq_length=24)


def create_base_model(sequence_length, n_features=1):
    model = Sequential(
        [
            LSTM(64, input_shape=(sequence_length, n_features), return_sequences=True),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
        return model


# Train base model on source domain
source_model = create_base_model(24)
_train_torch(source_model, source_sequences[:-1], source_scaled[24:])


# Extract features from intermediate layer
def create_feature_extractor(base_model, layer_name="lstm_1"):
    return Model(
        inputs=base_model.input, outputs=base_model.get_layer(layer_name).output
    )


feature_extractor = create_feature_extractor(source_model)


# Create new model using transferred features
def create_transfer_model(feature_extractor, sequence_length):
    inputs = Input(shape=(sequence_length, 1))
    features = feature_extractor(inputs)
    x = LSTM(16)(features)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
        return model


transfer_model = create_transfer_model(feature_extractor, 24)


def create_fine_tuning_model(base_model, trainable_layers=1):
    # Freeze early layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    return base_model


# Clone source model for fine-tuning
fine_tune_model = clone_model(source_model)
fine_tune_model.set_weights(source_model.get_weights())
fine_tune_model = create_fine_tuning_model(fine_tune_model)

# Fine-tune on target domain
_train_torch(fine_tune_model, target_sequences[:-1], target_scaled[24:])


class DomainAdapter:
    def __init__(self, source_scaler, target_scaler):
        self.source_scaler = source_scaler
        self.target_scaler = target_scaler

    def adapt_sequence(self, sequence, from_domain="source", to_domain="target"):
        if from_domain == "source" and to_domain == "target":
            # Inverse transform to original scale
            sequence_orig = self.source_scaler.inverse_transform(sequence)
            # Transform to target scale
            return self.target_scaler.transform(sequence_orig)
        sequence_orig = self.target_scaler.inverse_transform(sequence)
        return self.source_scaler.transform(sequence_orig)


# Create and use domain adapter
adapter = DomainAdapter(source_scaler, target_scaler)
adapted_sequences = adapter.adapt_sequence(source_sequences)


def evaluate_models(models, test_sequences, test_targets):
    results = {}
    for name, model in models.items():
        predictions = _predict_torch(model, test_sequences)
        mse = losses.MSE(test_targets, predictions)
        mae = losses.MAE(test_targets, predictions)
        results[name] = {"MSE": float(mse), "MAE": float(mae)}
    return pd.DataFrame(results).T


# Compare different approaches
models = {
    "Base Model": source_model,
    "Transfer Learning": transfer_model,
    "Fine-tuned": fine_tune_model,
}

results = evaluate_models(models, target_sequences[-100:], target_scaled[-100:])
logger.info("\nModel Comparison:")
logger.info(results)


def plot_predictions(models, test_sequences, true_values, scaler, plot: bool = False):
    if plot:
        plt.figure(figsize=(15, 6))

        # Plot true values
        plt.plot(scaler.inverse_transform(true_values), label="Actual", linewidth=2)

        # Plot predictions from each model
        for name, model in models.items():
            predictions = _predict_torch(model, test_sequences)
            plt.plot(
                scaler.inverse_transform(predictions),
                label=f"{name} Predictions",
                linestyle="--",
            )

        plt.title("Model Predictions Comparison")
        plt.legend()
        plt.show()


# Visualize results
plot_predictions(models, target_sequences[-100:], target_scaled[-100:], target_scaler)
