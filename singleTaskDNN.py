# TO DO NEXT SEMESTER:

# Incorporate signal processing on the device (ESP 32 chip)
# Update libraries to fit with ESP32 allowed libraries (update minmax scalar)
# Update single task DNN to estimate blood pressure, then estimate blood glucose from the blood pressure estimation value
# Convert single task to TFLite and C-Array
# Improve accuracy and thoroughly test


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
FILTERED_FILE = './processed_data/vitaldb_ppg_ecg_extracted_features_15s_nonlin.csv'
CASEID_COL = 'caseid'

TARGET_COLS = ['preop_gluc']   # single task: BG only

# Removed 'ppg_freq', 'ppg_first_deriv_min',
EXCLUDED_COL = [
    CASEID_COL, 'mean_bp', 'sys_bp', 'dys_bp',
    'ecg_mean', 'ecg_std', 'ecg_mean_pp_interval_s',
    'ecg_std_pp_interval_s', 'ecg_freq', 'ecg_auc',
    'ecg_first_deriv_max', 'ecg_first_deriv_min',
    'ecg_entropy', 'ecg_teager_energy', 'ecg_log_energy',
    'ecg_skew', 'ecg_iqr', 'ecg_spectral_entropy'
]

BATCH_SIZE    = 32
EPOCHS        = 10        # can go higher for the small net
LEARNING_RATE = 2e-4
DROPOUT = 0.03

# VERY SMALL DNN (TFLite-friendly)
DNN_LAYERS = [32, 16, 8]

# ---------------- HELPERS ----------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Compute MAPE (on original scale)
def mape(y_true, y_pred):
    eps = 1e-8  # prevent divide-by-zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

# ---------------- DATA LOAD ----------------
print("Loading filtered dataset...")
df = pd.read_csv(FILTERED_FILE).dropna()
print(f"Loaded shape: {df.shape}")

# Features = all columns except caseid, excluded cols, and target
features_to_use = [c for c in df.columns
                   if c not in EXCLUDED_COL + TARGET_COLS]

print(f"Using {len(features_to_use)} features: {features_to_use}")

X = df[features_to_use].values.astype(np.float32)
y = df[TARGET_COLS].values.astype(np.float32)  # shape (N, 1)

# ---------------- TRAIN / TEST SPLIT BY CASEID ----------------
unique_ids = df[CASEID_COL].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

is_train = df[CASEID_COL].isin(train_ids)
df_train = df[is_train].copy()
df_test  = df[~is_train].copy()

X_train = df_train[features_to_use].values.astype(np.float32)
X_test  = df_test[features_to_use].values.astype(np.float32)
y_train = df_train[TARGET_COLS].values.astype(np.float32)
y_test  = df_test[TARGET_COLS].values.astype(np.float32)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ---------------- SCALING ----------------
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled  = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled  = y_scaler.transform(y_test)

# ---------------- SMALL SINGLE-TASK MODEL ----------------
in_features = X_train_scaled.shape[1]

model = models.Sequential()
model.add(layers.Input(shape=(in_features,)))

for units in DNN_LAYERS:
    model.add(layers.Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-5)))
    model.add(layers.Dropout(DROPOUT))

# Single linear output for BG regression
model.add(layers.Dense(1, activation='linear'))

optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mse')

model.summary()

# ---------------- TRAIN ----------------
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# ---------------- EVALUATE (on original scale) ----------------
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

bg_true = y_test.flatten()
bg_pred = y_pred.flatten()

# ----- Clarke Error Grid (BG only) -----
def get_clarke_zone(ref, pred):
    r, p = float(ref), float(pred)
    if (r <= 70 and p <= 70) or (0.8*r <= p <= 1.2*r): return 'A'
    if (130 < r <= 180 and 1.4*(r-130) >= p) or (70 < r <= 280 and p >= (r+110)): return 'C'
    if (r <= 70 and 70 < p <= 180) or (r >= 240 and 70 <= p <= 180): return 'D'
    if (r <= 70 and p > 180) or (r > 180 and p <= 70): return 'E'
    return 'B'

zones_count = {z: 0 for z in 'ABCDE'}
total_points = len(bg_true)
points = []  # Store (ref, pred, zone) for plotting

for ref, pred in zip(bg_true, bg_pred):
    zone = get_clarke_zone(ref, pred)
    zones_count[zone] += 1
    points.append((ref, pred, zone))

print("\n--- Clarke Error Grid Analysis (BG only) ---")
for zone, count in zones_count.items():
    print(f"Zone {zone}: {count/total_points*100:.2f}% ({count}/{total_points})")
print("\n")

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
plt.fill([0, 70, 70, 0], [70, 70, 180, 180], color='red', alpha=0.1)
plt.fill([240, 400, 400, 240], [70, 70, 180, 180], color='red', alpha=0.1)

# Zone E: 
plt.fill([0, 70, 70, 0], [180, 180, 400, 400], color='purple', alpha=0.1)
plt.fill([180, 400, 400, 180], [0, 0, 70, 70], color='purple', alpha=0.1)

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

print("\n--- Single-Task Regression Metrics (BG only) ---")
print(f"MAE Glucose:  {mean_absolute_error(bg_true, bg_pred):.2f} mg/dL")
print(f"MAPE Glucose:  {mape(bg_true, bg_pred):.2f}%")
print(f"RMSE Glucose: {rmse(bg_true, bg_pred):.2f} mg/dL")

# Plot training/validation loss graph
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss (total)')
plt.plot(history.history['val_loss'], label='Val Loss (total)')
plt.xlabel('Epoch'); plt.ylabel('Weighted MSE Loss')
plt.title('Training and Validation Loss (Shared Multitask)')
plt.legend(); plt.tight_layout(); plt.show()

# Plot Coefficient of Determination
plt.figure(figsize=(7, 7))
plt.scatter(bg_true, bg_pred, alpha=0.5, label='Test Predictions')
slope, intercept = np.polyfit(bg_true, y_pred, 1)
slope = float(np.squeeze(slope))
intercept = float(np.squeeze(intercept))
plt.plot([bg_true.min(), bg_true.max()], [bg_true.min(), bg_true.max()], 'k--', label='Ideal (y=x)')
plt.plot([bg_true.min(), bg_true.max()],
        [slope * bg_true.min() + intercept, slope * bg_true.max() + intercept],
        color='red', lw=2, label=f'Trendline (slope={slope:.2f})')
plt.xlabel("Actual BG (mg/dL)")
plt.ylabel("Predicted BG (mg/dL)")
plt.title("Actual vs. Predicted BG (TensorFlow DNN)")
plt.legend(); plt.tight_layout(); plt.show()
print(f"R-Squared, Glucose: {slope:.2f}")

# ---------------- SAVE KERAS MODEL (for TFLite conversion) ----------------
os.makedirs('./model_weights', exist_ok=True)
keras_path = './model_weights/bg_regressor_small.keras'
model.save(keras_path)
print(f"Keras model saved to {keras_path}")
