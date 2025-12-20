import pandas as pd
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
EXCLUDED_COL = [CASEID_COL, TARGET_COL, 'ecg_mean', 'ecg_std', 'ecg_mean_pp_interval_s', 'ecg_std_pp_interval_s', 'ecg_freq', 'ecg_auc', 'ecg_first_deriv_max', 'ecg_first_deriv_min', 'ecg_entropy']
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
DROPOUT = 0.2
DNN_LAYERS = [256, 128, 64]
physical_devices = tf.config.list_physical_devices('GPU')
DEVICE = '/GPU:0' if physical_devices else '/CPU:0'

print("Hyperparameters:")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Dropout: {DROPOUT}")
print(f"DNN Layers: {DNN_LAYERS}")

print("Loading filtered dataset...")
df = pd.read_csv(FILTERED_FILE)
df = df.dropna()
print(f"Loaded shape: {df.shape}")

# ----- Feature/target selection -----
features_to_use = [col for col in df.columns if col not in EXCLUDED_COL]
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
            layers.BatchNormalization(),
            layers.Dropout(dropout),
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


# Clarke Error Grid Analysis
def get_clarke_zone(ref, pred):
    # Force numeric (helps when using pandas)
    r, p = float(ref), float(pred)

    if (r <= 70 and p <= 70) or (0.8*r <= p <= 1.2*r):
        return 'A'

    if (130 < r <= 180 and 1.4*(r-130) >= p) or (70 < r <= 280 and p >= (r+110)):
        return 'C'

    if (r <= 70 and 70 < p <= 180) or (r >= 240 and 70 <= p <= 180):
        return 'D'

    if (r <= 70 and p > 180) or (r > 180 and p <= 70):
        return 'E'

    return 'B'      # everything else

zones_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
total_points = len(y_test_orig)
points = []  # Store (ref, pred, zone) for plotting

for ref, pred in zip(y_test_orig, y_pred):
    zone = get_clarke_zone(ref, pred)
    zones_count[zone] += 1
    points.append((ref, pred, zone))

print("Clarke Error Grid Analysis:")
for zone, count in zones_count.items():
    percentage = (count / total_points) * 100
    print(f"Zone {zone}: {percentage:.2f}% ({count}/{total_points} points)")

# --- Plotting Clarke Error Grid ---
plt.figure(figsize=(10, 10))

# Define zone colors
colors = {'A': 'green', 'B': 'yellow', 'C': 'orange', 'D': 'red', 'E': 'purple'}
labels = {
    'A': 'A: Clinically Accurate',
    'B': 'B: Benign Errors',
    'C': 'C: Overcorrection',
    'D': 'D: Dangerous Failure to Detect',
    'E': 'E: Erroneous Treatment'
}

# Scatter points by zone
for zone in 'ABCDE':
    zone_points = [(r, p) for r, p, z in points if z == zone]
    if zone_points:
        refs, preds = zip(*zone_points)
        plt.scatter(refs, preds, c=colors[zone], label=f"{labels[zone]} ({zones_count[zone]})", alpha=0.7, edgecolors='k', s=60)

# Draw grid boundaries
max_val = 400
x = np.linspace(0, max_val, 500)

# Perfect line (y = x)
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect Agreement')

# Zone A boundaries: ±20% or within 70
plt.fill_between(x, 0.8*x, 1.2*x, where=(x <= 70) | (x >= 70), color='green', alpha=0.1, label='_nolegend_')
plt.fill_between(x, 0, 70, where=x <= 70, color='green', alpha=0.1)

# Zone B: outside A but safe
# Complex boundaries, draw other zones instead

# Zone C: 
plt.fill([70, 70, 290], [180, 400, 400], color='orange', alpha=0.4)
plt.fill([130, 180, 180], [0, 0, 70], color='orange', alpha=0.4)

# Zone D: 
plt.axhspan(70, 180, xmin=0, xmax=70/400, color='red', alpha=0.1)
plt.axhspan(70, 180, xmin=240/400, xmax=1, color='red', alpha=0.1)

# Zone E: 
plt.axhspan(180, max_val, xmin=0, xmax=70/400, color='purple', alpha=0.1)
plt.axhspan(0, 70, xmin=180/400, xmax=1, color='purple', alpha=0.1)

# Axis limits and labels
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel('Reference Glucose (mg/dL)', fontsize=12)
plt.ylabel('Predicted Glucose (mg/dL)', fontsize=12)
plt.title('Clarke Error Grid Analysis', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

# Legend
plt.legend(loc='upper left')

# Show plot
plt.tight_layout()
plt.show()

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
save_path = './model_weights/dnn_model_15s.weights.h5'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save_weights(save_path)
print(f"Model weights saved to {save_path}")
