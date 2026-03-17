"""
Complete LSTM Time Series Forecasting Example
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 50)
print("LSTM Time Series Forecasting")
print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print("=" * 50)

# 1. Generate sample time series data
print("\n1. Generating sample time series data...")
def generate_time_series(n_steps=1000):
    """Generate synthetic time series data with trend and seasonality"""
    time = np.arange(n_steps)
    # Trend component
    trend = 0.005 * time
    # Seasonal component
    seasonality = 2 * np.sin(2 * np.pi * time / 50)
    # Noise component
    noise = 0.5 * np.random.randn(n_steps)
    
    series = trend + seasonality + noise
    return series

# Generate data
n_steps = 1000
series = generate_time_series(n_steps)

# Plot the series
plt.figure(figsize=(15, 6))
plt.plot(series, label='Time Series Data', alpha=0.7)
plt.title('Generated Time Series Data with Trend and Seasonality')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('timeseries_original.png')
plt.show()
print(f"✓ Generated {n_steps} time steps")

# 2. Prepare data for LSTM
print("\n2. Preparing data for LSTM...")

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Define sequence length (how many past time steps to use)
seq_length = 50

# Create sequences
X, y = create_sequences(series_scaled, seq_length)

# Reshape X for LSTM input (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test sets (maintaining temporal order)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"✓ Training data shape: {X_train.shape}")
print(f"✓ Testing data shape: {X_test.shape}")
print(f"✓ Sequence length: {seq_length}")

# 3. Build LSTM Model
print("\n3. Building LSTM model...")

model = Sequential([
    # First LSTM layer with return sequences for stacking
    LSTM(units=64, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    
    # Output layer
    Dense(units=1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Model summary
model.summary()

# 4. Train the model
print("\n4. Training the model...")

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title('Model MAE')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# 5. Make predictions
print("\n5. Making predictions...")

# Make predictions
train_predictions = model.predict(X_train, verbose=0)
test_predictions = model.predict(X_test, verbose=0)

# Inverse transform to original scale
train_predictions = scaler.inverse_transform(train_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

test_predictions = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 6. Evaluate model performance
print("\n6. Evaluating model performance...")

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculate and print evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape}

# Calculate metrics
train_metrics = calculate_metrics(y_train_actual, train_predictions, "Training Set")
test_metrics = calculate_metrics(y_test_actual, test_predictions, "Test Set")

# 7. Plot results
print("\n7. Plotting results...")

plt.figure(figsize=(15, 10))

# Plot 1: Full time series with predictions
plt.subplot(2, 2, 1)
plt.plot(range(len(y_train_actual)), y_train_actual, 
         label='Actual Training', alpha=0.7, linewidth=1)
plt.plot(range(len(train_predictions)), train_predictions, 
         label='Predicted Training', alpha=0.7, linewidth=1)
plt.title('Training Data: Actual vs Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Test set predictions
plt.subplot(2, 2, 2)
test_start = len(y_train_actual)
test_end = test_start + len(y_test_actual)
plt.plot(range(test_start, test_end), y_test_actual, 
         label='Actual Test', alpha=0.7, linewidth=1)
plt.plot(range(test_start, test_end), test_predictions, 
         label='Predicted Test', alpha=0.7, linewidth=1)
plt.axvline(x=len(y_train_actual), color='red', linestyle='--', 
            label='Train/Test Split', alpha=0.5)
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Scatter plot actual vs predicted (test set)
plt.subplot(2, 2, 3)
plt.scatter(y_test_actual, test_predictions, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Set: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Residuals
plt.subplot(2, 2, 4)
residuals = y_test_actual.flatten() - test_predictions.flatten()
plt.scatter(range(len(residuals)), residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.title('Prediction Residuals (Test Set)')
plt.xlabel('Sample')
plt.ylabel('Residual')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_results.png')
plt.show()

# 8. Future predictions
print("\n8. Forecasting future values...")

def forecast_future(model, last_sequence, n_steps, scaler):
    """Forecast future values"""
    current_sequence = last_sequence.copy()
    future_predictions = []
    
    for i in range(n_steps):
        # Reshape for prediction
        current_input = current_sequence.reshape((1, seq_length, 1))
        
        # Make prediction
        next_value = model.predict(current_input, verbose=0)
        
        # Store prediction
        future_predictions.append(next_value[0, 0])
        
        # Update sequence
        current_sequence = np.append(current_sequence[1:], next_value)
    
    # Inverse transform
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    return future_predictions.flatten()

# Get the last sequence from test data
last_sequence = X_test[-1].flatten()

# Forecast next 50 steps
future_steps = 50
future_forecast = forecast_future(model, last_sequence, future_steps, scaler)

# Plot historical and forecasted values
plt.figure(figsize=(15, 6))

# Plot historical data (last 200 points)
historical = series[-200:]
plt.plot(range(len(historical)), historical, 
         label='Historical Data', linewidth=2, alpha=0.7)

# Plot forecast
forecast_start = len(historical)
forecast_end = forecast_start + len(future_forecast)
plt.plot(range(forecast_start, forecast_end), future_forecast, 
         label='50-Step Forecast', color='red', linewidth=2, marker='o', markersize=4)

plt.title('Time Series Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('future_forecast.png')
plt.show()

print("\n" + "=" * 50)
print("✅ Analysis Complete!")
print("=" * 50)
print("\nGenerated files:")
print("  - timeseries_original.png")
print("  - training_history.png")
print("  - prediction_results.png")
print("  - future_forecast.png")
print("\nModel performance saved in console output above.")