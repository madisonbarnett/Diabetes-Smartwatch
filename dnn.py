''' Code provided by Isaac Arnold, updated by Hunter Enders '''
''' 10/17/2025 '''
import pandas as pd
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ---- CONFIG ----
FILTERED_FILE = './processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv'
CASEID_COL = 'caseid'
TARGET_COL = 'preop_gluc'
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
DROPOUT = 0.2
DNN_LAYERS = [128, 64, 32]
physical_devices = tf.config.list_physical_devices('GPU')
DEVICE = '/GPU:0' if physical_devices else '/CPU:0'

print("Loading filtered dataset...")
df = pd.read_csv(FILTERED_FILE)
df = df.dropna()
print(f"Loaded shape: {df.shape}")

# ----- Feature/target selection -----
features_to_use = [col for col in df.columns if col not in [CASEID_COL, TARGET_COL]]
X = df[features_to_use].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)
caseids = df[CASEID_COL].values

print(f"Using {len(features_to_use)} features: {features_to_use[:10]} ...")

# ----- Subject-wise train/test split -----
unique_caseids = np.unique(caseids)
train_ids, test_ids = train_test_split(unique_caseids, test_size=0.2, random_state=42)
train_mask = np.isin(caseids, train_ids)
test_mask = ~train_mask
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ----- Fit scaler only on train! -----
x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled  = x_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
y_train_orig, y_test_orig = y_train, y_test

# Convert numpy arrays to TensorFlow tensors
X_train_t = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
y_train_t = tf.convert_to_tensor(y_train_scaled, dtype=tf.float32)[:, tf.newaxis]  # Equivalent to unsqueeze(1)
X_test_t = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
y_test_t = tf.convert_to_tensor(y_test_scaled, dtype=tf.float32)[:, tf.newaxis]  # Equivalent to unsqueeze(1)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_t, y_train_t))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_t, y_test_t))

# Configure the datasets with batching and shuffling
train_loader = train_dataset.shuffle(buffer_size=len(X_train_t)).batch(BATCH_SIZE)
test_loader = test_dataset.batch(BATCH_SIZE)

# ----- DNN Model -----
class DNNRegressor(tf.keras.Model):
    def __init__(self, in_features, layer_sizes, dropout):
        super(DNNRegressor, self).__init__()
        self.seq = models.Sequential([
            layers.Dense(layer_sizes[0], input_shape=(in_features,)),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(layer_sizes[1]),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(layer_sizes[2]),
            layers.ReLU(),
            layers.Dense(1)
        ])

    def call(self, x, training=False):
        return self.seq(x, training=training)

# Initialize model
model = DNNRegressor(X_train_t.shape[1], DNN_LAYERS, DROPOUT)

# Configure optimizer and loss
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
criterion = tf.keras.losses.MeanSquaredError()

# Compile model
model.compile(optimizer=optimizer, loss=criterion)

# ----- Training Loop -----
train_losses, val_losses = [], []
print("Training DNN model...")

@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        pred = model(xb, training=True)  # Forward pass with training=True for BatchNorm/Dropout
        loss = criterion(yb, pred)  # Compute loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def val_step(xb, yb):
    pred = model(xb, training=False)  # Forward pass with training=False
    return criterion(yb, pred)

for epoch in range(EPOCHS):
    # Training
    epoch_train_loss = 0.0
    num_train_samples = 0
    for xb, yb in train_loader:
        batch_size = tf.shape(xb)[0]
        loss = train_step(xb, yb)
        epoch_train_loss += loss * tf.cast(batch_size, tf.float32)
        num_train_samples += batch_size
    epoch_train_loss /= tf.cast(num_train_samples, tf.float32)
    train_losses.append(float(epoch_train_loss))

    # Validation
    val_loss = 0.0
    num_val_samples = 0
    for xb, yb in test_loader:
        batch_size = tf.shape(xb)[0]
        loss = val_step(xb, yb)
        val_loss += loss * tf.cast(batch_size, tf.float32)
        num_val_samples += batch_size
    val_loss /= tf.cast(num_val_samples, tf.float32)
    val_losses.append(float(val_loss))

    # Print progress
    if (epoch + 1) % 10 == 0 or epoch < 5:
        print(f"Epoch {epoch + 1}: train {epoch_train_loss:.4f} val {val_loss:.4f}")

# Evaluation
y_pred_scaled = []
for xb, _ in test_loader:
    preds = model(xb, training=False)  # Forward pass with training=False
    y_pred_scaled.append(preds.numpy().flatten())  # Convert to numpy and flatten

y_pred_scaled = np.concatenate(y_pred_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Compute metrics
mae = mean_absolute_error(y_test_orig, y_pred)
mape = mean_absolute_percentage_error(y_test_orig, y_pred) * 100
print(f"\nTest MAE: {mae:.2f} mg/dL")
print(f"Test MAPE: {mape:.2f}%")

# ----- Plots -----
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss (MSE)')
plt.plot(val_losses, label='Val Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(7, 7))
plt.scatter(y_test_orig, y_pred, alpha=0.5, label='Test Predictions')
slope, intercept = np.polyfit(y_test_orig, y_pred, 1)
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'k--', label='Ideal (y=x)')
plt.plot([y_test_orig.min(), y_test_orig.max()],
         [slope * y_test_orig.min() + intercept, slope * y_test_orig.max() + intercept],
         color='red', lw=2, label=f'Trendline (slope={slope:.2f})')
plt.xlabel("Actual BG (mg/dL)")
plt.ylabel("Predicted BG (mg/dL)")
plt.title("Actual vs. Predicted BG (TensorFlow DNN)")
plt.legend(); plt.tight_layout(); plt.show()
print(f"Scatter plot trendline slope: {slope:.2f}")

# Save TensorFlow model weights
save_path = './dnn_model_15s.weights.h5'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save_weights(save_path)
print(f"Model weights saved to {save_path}")
