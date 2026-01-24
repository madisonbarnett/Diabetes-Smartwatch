# LSTM model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

# Configuration 
SUFFIX        = '30s'
DATASET       = 'PhysioNet'
FILTERED_FILE = f'./processed_data/physioNet_ppg_extracted_features_{SUFFIX}.csv'
TARGET_COL    = 'glucose_mg_dl'
FEATURE_COLS = [
    'sex',
    'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s', 'ppg_std_pp_interval_s',
    'ppg_freq', 'ppg_auc', 'ppg_first_deriv_max', 'ppg_first_deriv_min',
    'ppg_entropy', 'ppg_teager_energy', 'ppg_log_energy', 'ppg_skew',
    'ppg_iqr', 'ppg_spectral_entropy'
]

HIDDEN_UNITS  = 64
DROPOUT       = 0.15
BATCH_SIZE    = 256
EPOCHS        = 100
PATIENCE_ES   = 15
PATIENCE_LR   = 5
SEED          = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Load dataset
print("Loading data...")
df = pd.read_csv(FILTERED_FILE).dropna()
print(f"Raw rows: {df.shape[0]}, subjects: {df.get('patient_id', pd.Series()).nunique()}")

df["log_gluc"] = np.log1p(df[TARGET_COL])

# Drop patient_id / caseid - no longer needed
if 'patient_id' in df.columns:
    df = df.drop(columns=['patient_id'])

# Features and target
X = df[FEATURE_COLS].values.astype(np.float32)
y_log = df["log_gluc"].values.reshape(-1, 1).astype(np.float32)

print(f"Features shape: {X.shape}, Target shape: {y_log.shape}")
print(f"Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")

# Stratified split by glucose quintiles
gluc_quint = pd.qcut(df[TARGET_COL], q=5, duplicates='drop', labels=False)

train_idx, test_idx = train_test_split(
    range(len(df)),
    test_size=0.2,
    random_state=SEED,
    stratify=gluc_quint
)

X_train, X_test       = X[train_idx], X[test_idx]
y_log_train, y_log_test = y_log[train_idx], y_log[test_idx]

y_orig_train = np.expm1(y_log_train).flatten()
y_orig_test  = np.expm1(y_log_test).flatten()

print(f"Train rows: {len(X_train):,}, Test rows: {len(X_test):,}")

# Scaling
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_log_train)
y_test_s  = scaler_y.transform(y_log_test)

# Define model (simple MLP)
def build_model(n_features):
    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(HIDDEN_UNITS * 2, activation="relu")(inp)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(HIDDEN_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="mae",
        metrics=["mae"]
    )
    return model

model = build_model(len(FEATURE_COLS))
model.summary()

# Callbacks
cb_es = callbacks.EarlyStopping(
    monitor="val_mae",
    patience=PATIENCE_ES,
    restore_best_weights=True,
    min_delta=1e-4
)
cb_lr = callbacks.ReduceLROnPlateau(
    monitor="val_mae",
    factor=0.3,
    patience=PATIENCE_LR,
    min_lr=1e-6
)

# Train
history = model.fit(
    x=X_train,
    y=y_train_s,
    validation_data=(X_test, y_test_s),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[cb_es, cb_lr],
    verbose=2
)

# Predict
y_pred_s = model.predict(X_test, verbose=0).flatten()
y_pred_log = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
y_pred = np.expm1(y_pred_log)

# Metrics
mae  = mean_absolute_error(y_orig_test, y_pred)
mape = mean_absolute_percentage_error(y_orig_test, y_pred) * 100
r2   = r2_score(y_orig_test, y_pred)

print("\n" + "="*60)
print(f"MAE  : {mae:6.2f} mg/dL")
print(f"MAPE : {mape:6.2f}%")
print(f"R²   : {r2:6.3f}")
print("="*60)

# CEGA
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
total_points = len(y_orig_test)
points = []  # Store (ref, pred, zone) for plotting

for ref, pred in zip(y_orig_test, y_pred):
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
# x: left, right, right, left
# y: bottom, bottom, top, top

plt.fill([0, 70, 70, 0], [70, 70, 180, 180], color='red', alpha=0.1)
plt.fill([240, 400, 400, 240], [70, 70, 180, 180], color='red', alpha=0.1)

# Zone E: 
# plt.axhspan(180, max_val, xmin=0, xmax=70/400, color='purple', alpha=0.1)
# plt.axhspan(0, 70, xmin=180/400, xmax=1, color='purple', alpha=0.1)

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

# Plots
# Training curve (train vs val MAE, log scale)
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train MAE')
plt.plot(history.history['val_loss'], label='Val MAE')
plt.title('Training / Validation MAE (log scale)')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# Actual vs Predicted scatter
plt.figure(figsize=(7,7))
plt.scatter(y_orig_test, y_pred, alpha=0.6, s=30, edgecolor='k')
lim = [y_orig_test.min(), y_orig_test.max()]
plt.plot(lim, lim, 'k--', lw=1)
slope, intercept = np.polyfit(y_orig_test, y_pred, 1)
plt.plot(lim, slope*np.array(lim) + intercept, 'r-', lw=2,
         label=f'Trend (slope={slope:.2f})')
plt.xlabel('Actual BG'); plt.ylabel('Predicted BG'); plt.title('Actual vs Predicted')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# Save model and scalers
save_dir = Path("./model_weights")
save_dir.mkdir(exist_ok=True)
model.save(save_dir / f"mlp_model_{SUFFIX}_{DATASET}.keras")

print(f"\nSaved model: {save_dir / f'mlp_model_{SUFFIX}_{DATASET}.keras'}")