# --------------------------------------------------------------
#  lstm2.py
#  Improved LSTM model!
#  Global scaling + log-target scaling 
# --------------------------------------------------------------

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

# --------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------
SUFFIX        = '5s_nonlin'
FILTERED_FILE = f'./processed_data/vitaldb_ppg_ecg_extracted_features_{SUFFIX}.csv'
CASEID_COL    = "caseid"
TARGET_COL    = "preop_gluc"
STATIC_COLS   = ["age", "sex", "preop_dm", "weight", "height", "bmi"]

# Exclude ECG features
EXCLUDED_COLS = ['mean_bp', 'sys_bp', 'dys_bp', 'ecg_mean', 'ecg_std', 'ecg_mean_pp_interval_s', 'ecg_std_pp_interval_s',
                 'ecg_freq', 'ecg_auc', 'ecg_first_deriv_max', 'ecg_first_deriv_min',
                 'ecg_entropy', 'ecg_teager_energy', 'ecg_log_energy', 'ecg_skew',
                 'ecg_iqr', 'ecg_spectral_entropy']

# MAX_T         = 64         # max sequence length (64 time-steps)
MAX_T         = 192         # max sequence length (192 time-steps)
LSTM_UNITS    = 32
DENSE_UNITS   = 32         
DROPOUT       = 0.5
BATCH_SIZE    = 128
EPOCHS        = 200
PATIENCE_ES   = 15
PATIENCE_LR   = 5
SEED          = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------
# 2. LOAD & LOG TARGET
# --------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(FILTERED_FILE).dropna()    # Load CSV, drop rows with missing value
print(f"Raw rows: {df.shape[0]}, subjects: {df[CASEID_COL].nunique()}")

assert df.groupby(CASEID_COL)[TARGET_COL].nunique().max() == 1  # Check one target per subject
df["log_gluc"] = np.log1p(df[TARGET_COL])   # Log-transform target to stabilize variance

dynamic_cols = [c for c in df.columns if c not in (CASEID_COL, TARGET_COL, "log_gluc", *STATIC_COLS, *EXCLUDED_COLS)]
static_cols  = STATIC_COLS
features_to_use = dynamic_cols + static_cols
D, S = len(dynamic_cols), len(static_cols)

print(f"Using {len(features_to_use)} features: {features_to_use[:]} ...")

# --------------------------------------------------------------
# 3. BUILD ARRAYS + PADDING (NO per-subject z-score)
# --------------------------------------------------------------
subjects = df[CASEID_COL].unique()
n_subj   = len(subjects)

# Create three arrays per patient
X_dyn  = np.zeros((n_subj, MAX_T, D), dtype=np.float32) # Padded/truncated time series
X_stat = np.zeros((n_subj, S), dtype=np.float32)    # Static covariates (same for every time step)
y_log  = np.zeros((n_subj, 1), dtype=np.float32)    # Log glucose

print("Padding (global scaling later)...")
for i, subj in enumerate(subjects):
    sub_df = df[df[CASEID_COL] == subj]
    seq = sub_df[dynamic_cols].values.astype(np.float32)

    # Pad / truncate
    if len(seq) >= MAX_T:
        seq = seq[:MAX_T]
    else:
        seq = np.pad(seq, ((0, MAX_T - len(seq)), (0, 0)), mode="constant")

    X_dyn[i]  = seq
    X_stat[i] = sub_df[static_cols].iloc[0].values
    y_log[i]  = sub_df["log_gluc"].iloc[0]

# --------------------------------------------------------------
# 4. STRATIFIED SPLIT
# --------------------------------------------------------------
gluc = df.groupby(CASEID_COL)[TARGET_COL].first()
strata = pd.qcut(gluc, q=5, duplicates="drop").cat.codes.values

train_ids, test_ids = train_test_split(
    subjects, test_size=0.2, random_state=SEED, stratify=strata
)
train_mask = np.isin(subjects, train_ids)

X_dyn_train, X_dyn_test   = X_dyn[train_mask], X_dyn[~train_mask]
X_stat_train, X_stat_test = X_stat[train_mask], X_stat[~train_mask]
y_log_train, y_log_test   = y_log[train_mask], y_log[~train_mask]

y_orig_train = np.expm1(y_log_train).flatten()
y_orig_test  = np.expm1(y_log_test).flatten()

print(f"Train: {X_dyn_train.shape[0]}, Test: {X_dyn_test.shape[0]}")

# --------------------------------------------------------------
# 5. GLOBAL SCALING (dynamic + static + target)
# --------------------------------------------------------------
# Dynamic – fit on TRAIN only
scaler_dyn = StandardScaler()
X_dyn_train_flat = X_dyn_train.reshape(-1, D)
X_dyn_train = scaler_dyn.fit_transform(X_dyn_train_flat).reshape(-1, MAX_T, D)
X_dyn_test  = scaler_dyn.transform(X_dyn_test.reshape(-1, D)).reshape(-1, MAX_T, D)

# Static
scaler_stat = StandardScaler()
X_stat_train = scaler_stat.fit_transform(X_stat_train)
X_stat_test  = scaler_stat.transform(X_stat_test)

# Log-target
scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_log_train)
y_test_s  = scaler_y.transform(y_log_test)

# --------------------------------------------------------------
# 6. MODEL (dual pooling, no attention)
# --------------------------------------------------------------
def build_model(timesteps, n_dyn, n_stat):
    dyn_in  = layers.Input(shape=(timesteps, n_dyn), name="dyn")
    stat_in = layers.Input(shape=(n_stat,), name="stat")

    # Repeat static (broadcast static vector to every time step)
    stat_rep = layers.RepeatVector(timesteps)(stat_in)
    x = layers.Concatenate(axis=-1)([dyn_in, stat_rep])
    x = layers.Masking(mask_value=0.0)(x)  # Ignores padded zeros

    # Bidirectional LSTM (Captures forward + backward temporal patterns; returns full sequence)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, dropout=DROPOUT, recurrent_dropout=DROPOUT,
                    return_sequences=True)
    )(x)   # (B, T, 64)

    # Dual pooling (summarizes whole sequence into fixed-size vector)
    # Mask-aware pooling (drops time dim, keeps mask logic)
    avg_pool = layers.GlobalAveragePooling1D(keepdims=False)(x)
    max_pool = layers.GlobalMaxPooling1D(keepdims=False)(x)
    x = layers.Concatenate()([avg_pool, max_pool])   # (B, 128)

    # Head (non-linear mapping, single regression value)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = Model([dyn_in, stat_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="mae",
        metrics=["mae"]
    )
    return model

model = build_model(MAX_T, D, S)
model.summary()

# --------------------------------------------------------------
# 7. CALLBACKS
# --------------------------------------------------------------
# log_dir = Path("./tb_logs/lstm_fixed")
# log_dir.mkdir(parents=True, exist_ok=True)

# Stop when validation MAE stops improving for PATIENCE_ES epochs, restore best weights
cb_es = callbacks.EarlyStopping(monitor="val_mae", patience=PATIENCE_ES,
                                restore_best_weights=True, min_delta=1e-4)
# Reduce LR when validation MAE plateaus
cb_lr = callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.3,
                                    patience=PATIENCE_LR, min_lr=1e-6)
# cb_tb = callbacks.TensorBoard(log_dir=str(log_dir))

# --------------------------------------------------------------
# 8. TRAIN
# --------------------------------------------------------------
history = model.fit(
    x=[X_dyn_train, X_stat_train],
    y=y_train_s,    # Training on scaled log-target
    validation_data=([X_dyn_test, X_stat_test], y_test_s),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[cb_es, cb_lr],
    verbose=2   # Prints one line per epoch
)

# --------------------------------------------------------------
# 9. PREDICT & INVERSE
# --------------------------------------------------------------
y_pred_s = model.predict([X_dyn_test, X_stat_test], verbose=0).flatten()    # Predict on scaled space
y_pred_log = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()  # Inverse the StnadardScaler -> original log-glucose
y_pred = np.expm1(y_pred_log)   # Original glucose (mg/dL)

# --------------------------------------------------------------
# 10. METRICS
# --------------------------------------------------------------
mae  = mean_absolute_error(y_orig_test, y_pred) # absolute error
mape = mean_absolute_percentage_error(y_orig_test, y_pred) * 100    # relative error in %
r2   = r2_score(y_orig_test, y_pred)    # proportion of variance explained

print("\n" + "="*60)
print(f"MAE  : {mae:6.2f} mg/dL")
print(f"MAPE : {mape:6.2f}%")
print(f"R²   : {r2:6.3f}")
print("="*60)

# --------------------------------------------------------------
# 11. CLARKE GRID
# --------------------------------------------------------------
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

# --------------------------------------------------------------
# 12. PLOTS
# --------------------------------------------------------------
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

# --------------------------------------------------------------
# 13. SAVE
# --------------------------------------------------------------
save_dir = Path("./model_weights")
save_dir.mkdir(exist_ok=True)
model.save(save_dir / f"lstm_model2_{SUFFIX}.keras")

# np.savez(save_dir / "scalers.npz",
#          dyn_mean=scaler_dyn.mean_, dyn_scale=scaler_dyn.scale_,
#          stat_mean=scaler_stat.mean_, stat_scale=scaler_stat.scale_,
#          y_mean=scaler_y.mean_[0], y_scale=scaler_y.scale_[0])

print(f"\nSaved: {save_dir / f'lstm_model2_{SUFFIX}.keras'}")
# print(f"TensorBoard: tensorboard --logdir {log_dir}")