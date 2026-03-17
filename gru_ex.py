import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Generate sample data (sine wave prediction)
def generate_sine_wave_data(seq_length=50, num_samples=1000):
    x = np.linspace(0, 100, num_samples)
    y = np.sin(x)
    
    # Create sequences
    X, y_true = [], []
    for i in range(len(y) - seq_length):
        X.append(y[i:i + seq_length])
        y_true.append(y[i + seq_length])
    
    X = np.array(X).reshape(-1, seq_length, 1)
    y_true = np.array(y_true)
    
    return X, y_true

# Parameters
seq_length = 50
num_features = 1
num_samples = 1000

# Generate data
X, y = generate_sine_wave_data(seq_length, num_samples)

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create GRU model
model = Sequential([
    GRU(50, activation='tanh', return_sequences=True, input_shape=(seq_length, num_features)),
    GRU(50, activation='tanh', return_sequences=False),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mse', 
              metrics=['mae'])

# Display model architecture
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions
predictions = model.predict(X_test)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.plot(y_test[:100], label='Actual')
plt.plot(predictions[:100], label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.title('Predictions vs Actual')

plt.tight_layout()
plt.show()