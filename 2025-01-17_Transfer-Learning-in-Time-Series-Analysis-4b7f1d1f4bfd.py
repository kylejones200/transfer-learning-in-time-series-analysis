# Description: Short example for Transfer Learning in Time Series Analysis.



from data_io import read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential, Model
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Helper function to create time series sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:(i + seq_length)])
    return np.array(sequences)

# Load and prepare source domain data (e.g., energy consumption)
source_data = read_csv('energy_consumption.csv')
source_scaler = MinMaxScaler()
source_scaled = source_scaler.fit_transform(source_data[['consumption']])
source_sequences = create_sequences(source_scaled, seq_length=24)

# Load and prepare target domain data (e.g., solar production)
target_data = read_csv('solar_production.csv')
target_scaler = MinMaxScaler()
target_scaled = target_scaler.fit_transform(target_data[['production']])
target_sequences = create_sequences(target_scaled, seq_length=24)

def create_base_model(sequence_length, n_features=1):
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, n_features), return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train base model on source domain
source_model = create_base_model(24)
source_model.fit(
    source_sequences[:-1], 
    source_scaled[24:], 
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Extract features from intermediate layer
def create_feature_extractor(base_model, layer_name='lstm_1'):
    return Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer_name).output
    )

feature_extractor = create_feature_extractor(source_model)

# Create new model using transferred features
def create_transfer_model(feature_extractor, sequence_length):
    inputs = Input(shape=(sequence_length, 1))
    features = feature_extractor(inputs)
    x = LSTM(16)(features)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

transfer_model = create_transfer_model(feature_extractor, 24)

def create_fine_tuning_model(base_model, trainable_layers=1):
    # Freeze early layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    return base_model

# Clone source model for fine-tuning
fine_tune_model = tf.keras.models.clone_model(source_model)
fine_tune_model.set_weights(source_model.get_weights())
fine_tune_model = create_fine_tuning_model(fine_tune_model)

# Fine-tune on target domain
fine_tune_model.fit(
    target_sequences[:-1],
    target_scaled[24:],
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

class DomainAdapter:
    def __init__(self, source_scaler, target_scaler):
        self.source_scaler = source_scaler
        self.target_scaler = target_scaler
    
    def adapt_sequence(self, sequence, from_domain='source', to_domain='target'):
        if from_domain == 'source' and to_domain == 'target':
            # Inverse transform to original scale
            sequence_orig = self.source_scaler.inverse_transform(sequence)
            # Transform to target scale
            return self.target_scaler.transform(sequence_orig)
        else:
            sequence_orig = self.target_scaler.inverse_transform(sequence)
            return self.source_scaler.transform(sequence_orig)

# Create and use domain adapter
adapter = DomainAdapter(source_scaler, target_scaler)
adapted_sequences = adapter.adapt_sequence(source_sequences)

def evaluate_models(models, test_sequences, test_targets):
    results = {}
    for name, model in models.items():
        predictions = model.predict(test_sequences)
        mse = tf.keras.losses.MSE(test_targets, predictions)
        mae = tf.keras.losses.MAE(test_targets, predictions)
        results[name] = {'MSE': float(mse), 'MAE': float(mae)}
    return pd.DataFrame(results).T

# Compare different approaches
models = {
    'Base Model': source_model,
    'Transfer Learning': transfer_model,
    'Fine-tuned': fine_tune_model
}

results = evaluate_models(
    models,
    target_sequences[-100:],
    target_scaled[-100:]
)
logger.info("\nModel Comparison:")
logger.info(results)


def plot_predictions(models, test_sequences, true_values, scaler):
    plt.figure(figsize=(15, 6))
    
    # Plot true values
    plt.plot(scaler.inverse_transform(true_values), 
             label='Actual', linewidth=2)
    
    # Plot predictions from each model
    for name, model in models.items():
        predictions = model.predict(test_sequences)
        plt.plot(scaler.inverse_transform(predictions), 
                label=f'{name} Predictions', linestyle='--')
    
    plt.title('Model Predictions Comparison')
    plt.legend()
        plt.show()

# Visualize results
plot_predictions(
    models,
    target_sequences[-100:],
    target_scaled[-100:],
    target_scaler
)
