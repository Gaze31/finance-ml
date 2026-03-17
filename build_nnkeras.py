import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

# Keras/TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("=" * 80)

# ============================================================================
# 1. SIMPLE SEQUENTIAL MODEL - BINARY CLASSIFICATION
# ============================================================================
print("\n1. BINARY CLASSIFICATION - BREAST CANCER DATASET")
print("=" * 80)

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model_binary = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model_binary.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\nModel Architecture:")
model_binary.summary()

# Train model
print("\nTraining...")
history_binary = model_binary.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
test_loss, test_acc = model_binary.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred_prob = model_binary.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# ============================================================================
# 2. MULTI-CLASS CLASSIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("2. MULTI-CLASS CLASSIFICATION - IRIS DATASET")
print("=" * 80)

# Load data
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split and scale
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

# One-hot encode labels
y_train_iris_cat = to_categorical(y_train_iris, num_classes=3)
y_test_iris_cat = to_categorical(y_test_iris, num_classes=3)

# Build model
model_multiclass = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])

# Compile
model_multiclass.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model_multiclass.summary()

# Train
print("\nTraining...")
history_multiclass = model_multiclass.fit(
    X_train_iris_scaled, y_train_iris_cat,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

# Evaluate
test_loss_mc, test_acc_mc = model_multiclass.evaluate(
    X_test_iris_scaled, y_test_iris_cat, verbose=0
)
print(f"\nTest Accuracy: {test_acc_mc:.4f}")

# Predictions
y_pred_iris = model_multiclass.predict(X_test_iris_scaled, verbose=0)
y_pred_iris_classes = np.argmax(y_pred_iris, axis=1)
print("\nClassification Report:")
print(classification_report(y_test_iris, y_pred_iris_classes, 
                          target_names=iris.target_names))

# ============================================================================
# 3. REGRESSION WITH NEURAL NETWORK
# ============================================================================
print("\n" + "=" * 80)
print("3. REGRESSION - CALIFORNIA HOUSING DATASET")
print("=" * 80)

# Load data
housing = fetch_california_housing()
X_house, y_house = housing.data, housing.target

# Split and scale
X_train_house, X_test_house, y_train_house, y_test_house = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

scaler_house = StandardScaler()
X_train_house_scaled = scaler_house.fit_transform(X_train_house)
X_test_house_scaled = scaler_house.transform(X_test_house)

# Build model
model_regression = models.Sequential([
    layers.Input(shape=(X_train_house_scaled.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation for regression
])

# Compile
model_regression.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nModel Architecture:")
model_regression.summary()

# Train
print("\nTraining...")
history_regression = model_regression.fit(
    X_train_house_scaled, y_train_house,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

# Evaluate
test_loss_reg, test_mae = model_regression.evaluate(
    X_test_house_scaled, y_test_house, verbose=0
)
print(f"\nTest MSE: {test_loss_reg:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# ============================================================================
# 4. FUNCTIONAL API - COMPLEX ARCHITECTURE
# ============================================================================
print("\n" + "=" * 80)
print("4. FUNCTIONAL API - BRANCHED NETWORK")
print("=" * 80)

# Input layer
inputs = layers.Input(shape=(X_train_scaled.shape[1],))

# Branch 1
branch1 = layers.Dense(32, activation='relu')(inputs)
branch1 = layers.Dropout(0.3)(branch1)
branch1 = layers.Dense(16, activation='relu')(branch1)

# Branch 2
branch2 = layers.Dense(32, activation='relu')(inputs)
branch2 = layers.Dropout(0.3)(branch2)
branch2 = layers.Dense(16, activation='relu')(branch2)

# Concatenate branches
concatenated = layers.concatenate([branch1, branch2])

# Additional layers
x = layers.Dense(32, activation='relu')(concatenated)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

# Create model
model_functional = models.Model(inputs=inputs, outputs=outputs)

# Compile
model_functional.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nFunctional Model Architecture:")
model_functional.summary()

# ============================================================================
# 5. CALLBACKS - EARLY STOPPING & MODEL CHECKPOINT
# ============================================================================
print("\n" + "=" * 80)
print("5. USING CALLBACKS")
print("=" * 80)

# Build simple model
model_callbacks = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model_callbacks.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\nTraining with callbacks...")
history_callbacks = model_callbacks.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

print(f"\nTraining stopped at epoch: {len(history_callbacks.history['loss'])}")

# ============================================================================
# 6. CUSTOM OPTIMIZER AND LEARNING RATE
# ============================================================================
print("\n" + "=" * 80)
print("6. CUSTOM OPTIMIZER CONFIGURATION")
print("=" * 80)

model_custom_opt = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Custom optimizer with learning rate
custom_optimizer = keras.optimizers.Adam(learning_rate=0.001)

model_custom_opt.compile(
    optimizer=custom_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled with custom Adam optimizer (lr=0.001)")

# ============================================================================
# 7. DIFFERENT ACTIVATION FUNCTIONS
# ============================================================================
print("\n" + "=" * 80)
print("7. ACTIVATION FUNCTIONS COMPARISON")
print("=" * 80)

activations = ['relu', 'tanh', 'sigmoid', 'elu']

print("\nCommon Activation Functions:")
print("- ReLU: Most common, f(x) = max(0, x)")
print("- Tanh: f(x) = tanh(x), range [-1, 1]")
print("- Sigmoid: f(x) = 1/(1 + e^-x), range [0, 1]")
print("- ELU: Exponential Linear Unit")
print("- Softmax: For multi-class output layer")

# ============================================================================
# 8. REGULARIZATION TECHNIQUES
# ============================================================================
# ============================================================================
# 8. REGULARIZATION TECHNIQUES
# ============================================================================
print("\n" + "=" * 80)
print("8. REGULARIZATION TECHNIQUES")
print("=" * 80)

from tensorflow.keras import regularizers

model_regularized = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    
    # L2 regularization
    layers.Dense(64, activation='relu', 
                kernel_regularizer=regularizers.l2(0.001)),
    
    # Dropout
    layers.Dropout(0.4),
    
    # Batch Normalization
    layers.BatchNormalization(),
    
    layers.Dense(32, activation='relu',
                kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='sigmoid')
])

model_regularized.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Regularization techniques applied:")
print("✓ L2 regularization (kernel_regularizer)")
print("✓ Dropout layers")
print("✓ Batch Normalization")

# ============================================================================
# 9. PRACTICAL TIPS
# ============================================================================
print("\n" + "=" * 80)
print("9. KERAS NEURAL NETWORK - BEST PRACTICES")
print("=" * 80)

tips = """
ARCHITECTURE DESIGN:
--------------------
✓ Start simple, add complexity gradually
✓ Input layer: Use Input() to define input shape
✓ Hidden layers: Typically 1-3 layers for simple problems
✓ Layer sizes: Often decrease toward output (e.g., 128→64→32)
✓ Output layer: 
  - Binary: 1 neuron, sigmoid activation
  - Multi-class: n neurons, softmax activation
  - Regression: 1 neuron, no activation

ACTIVATION FUNCTIONS:
---------------------
✓ Hidden layers: ReLU (most common), tanh, ELU
✓ Output layer:
  - Binary classification: sigmoid
  - Multi-class: softmax
  - Regression: none (linear)

OPTIMIZATION:
-------------
✓ Optimizer: Adam (good default), SGD, RMSprop
✓ Learning rate: 0.001 (default), tune if needed
✓ Batch size: 32 (common), 64, 128 for larger datasets
✓ Epochs: Start with 50-100, use early stopping

REGULARIZATION:
---------------
✓ Dropout: 0.2-0.5 (prevents overfitting)
✓ Batch Normalization: Stabilizes training
✓ L1/L2 regularization: Penalizes large weights
✓ Early stopping: Prevents overfitting

DATA PREPROCESSING:
-------------------
✓ Always scale features (StandardScaler, MinMaxScaler)
✓ One-hot encode categorical variables
✓ Handle missing values
✓ Split data: train/validation/test

LOSS FUNCTIONS:
---------------
✓ Binary classification: binary_crossentropy
✓ Multi-class: categorical_crossentropy
✓ Regression: mse, mae, huber

MONITORING:
-----------
✓ Track training and validation metrics
✓ Use callbacks (EarlyStopping, ModelCheckpoint)
✓ Plot learning curves
✓ Check for overfitting (val_loss > train_loss)

SAVING/LOADING MODELS:
----------------------
✓ Save: model.save('model_name.keras')
✓ Load: model = keras.models.load_model('model_name.keras')
"""

print(tips)

print("\n" + "=" * 80)
print("KERAS NEURAL NETWORK TUTORIAL COMPLETE!")
print("=" * 80)

import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
