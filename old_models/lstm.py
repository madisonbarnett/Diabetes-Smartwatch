# --------------------------------------------------------------
#  lstm_improved.py
#  LSTM + Static Repeat + Bidir + MAE + Log(Target) + Stratified
#  Full evaluation: Clarke, plots, metrics
# --------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from pathlib import Path

# --------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------
FILTERED_FILE = "./processed_data/vitaldb_ppg_ecg_extracted_features_15s_nonlin.csv"
CASEID_COL    = "caseid"
TARGET_COL    = "preop_gluc"

STATIC_COLS   = ["age", "sex", "preop_dm", "weight", "height"]

# Hyper-parameters
MAX_T         = 64
LSTM_UNITS    = 32          # smaller → less overfit
DENSE_UNITS   = 16
DROPOUT       = 0.5         # heavy
BATCH_SIZE    = 64
EPOCHS        = 50
PATIENCE_ES   = 25
PATIENCE_LR   = 10
SEED          = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(FILTERED_FILE).dropna()
print(f"Raw rows: {df.shape[0]}, subjects: {df[CASEID_COL].nunique()}")

# Sanity
assert df.groupby(CASEID_COL)[TARGET_COL].nunique().max() == 1

dynamic_cols = [c for c in df.columns if c not in (CASEID_COL, TARGET_COL, *STATIC_COLS)]
static_cols  = STATIC_COLS

print(f"Dynamic: {len(dynamic_cols)}, Static: {len(static_cols)}")

# --------------------------------------------------------------
# 3. LOG-TRANSFORM TARGET
# --------------------------------------------------------------
df['log_gluc'] = np.log1p(df[TARGET_COL])

# --------------------------------------------------------------
# 4. BUILD (X_dyn, X_stat, y_log) WITH PADDING + PER-SUBJECT Z-SCORE
# --------------------------------------------------------------
subjects = df[CASEID_COL].unique()
n_subj   = len(subjects)
D = len(dynamic_cols)
S = len(static_cols)

X_dyn  = np.zeros((n_subj, MAX_T, D), dtype=np.float32)
X_stat = np.zeros((n_subj, S), dtype=np.float32)
y_log  = np.zeros((n_subj, 1), dtype=np.float32)

print("Padding + per-subject z-scoring...")
for i, subj in enumerate(subjects):
    sub_df = df[df[CASEID_COL] == subj]
    seq = sub_df[dynamic_cols].values.astype(np.float32)

    # --- Per-subject z-score ---
    mean_seq = seq.mean(axis=0, keepdims=True)
    std_seq  = seq.std(axis=0, keepdims=True) + 1e-8
    seq = (seq - mean_seq) / std_seq

    # --- Pad ---
    if len(seq) >= MAX_T:
        seq = seq[:MAX_T]
    else:
        pad_width = MAX_T - len(seq)
        seq = np.pad(seq, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)

    X_dyn[i] = seq
    X_stat[i] = sub_df[static_cols].iloc[0].values
    y_log[i] = sub_df['log_gluc'].iloc[0]

real_lengths = np.sum(np.any(X_dyn != 0, axis=-1), axis=1)
print(f"Real lengths → min: {real_lengths.min()}, mean: {real_lengths.mean():.1f}")

# --------------------------------------------------------------
# 5. STRATIFIED SUBJECT SPLIT (by glucose bins)
# --------------------------------------------------------------
bins = pd.qcut(df.groupby(CASEID_COL)[TARGET_COL].first(), q=5, duplicates='drop')
strata = df.groupby(CASEID_COL).ngroup().map(dict(enumerate(bins.cat.codes)))

train_ids, test_ids = train_test_split(
    subjects, test_size=0.20, random_state=SEED, stratify=strata.loc[subjects]
)
train_mask = np.isin(subjects, train_ids)

X_dyn_train, X_dyn_test   = X_dyn[train_mask], X_dyn[~train_mask]
X_stat_train, X_stat_test = X_stat[train_mask], X_stat[~train_mask]
y_log_train, y_log_test   = y_log[train_mask], y_log[~train_mask]
y_orig_train = np.expm1(y_log_train).flatten()
y_orig_test  = np.expm1(y_log_test).flatten()

print(f"Train: {X_dyn_train.shape[0]}, Test: {X_dyn_test.shape[0]}")

# --------------------------------------------------------------
# 6. SCALING (static only — dynamic already z-scored)
# --------------------------------------------------------------
scaler_stat = StandardScaler()
X_stat_train = scaler_stat.fit_transform(X_stat_train)
X_stat_test  = scaler_stat.transform(X_stat_test)

# Log target already scaled → no StandardScaler needed
y_train_s = y_log_train
y_test_s  = y_log_test

# --------------------------------------------------------------
# 7. MODEL: Bidir LSTM + Static Repeat + Attention
# --------------------------------------------------------------
def build_model(timesteps, n_dyn, n_stat):
    # Inputs
    dyn_in  = layers.Input(shape=(timesteps, n_dyn), name="dyn")
    stat_in = layers.Input(shape=(n_stat,), name="stat")

    # Repeat static across time
    stat_rep = layers.RepeatVector(timesteps)(stat_in)           # (B, T, S)

    # Concatenate
    x = layers.Concatenate(axis=-1)([dyn_in, stat_rep])          # (B, T, D+S)
    x = layers.Masking(mask_value=0.0)(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS,
                    dropout=DROPOUT,
                    recurrent_dropout=DROPOUT,
                    return_sequences=True)
    )(x)                                                         # (B, T, 2*units)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    # Head
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = Model([dyn_in, stat_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
        loss="mae",
        metrics=["mae"]
    )
    return model

model = build_model(MAX_T, D, S)
model.summary()

# --------------------------------------------------------------
# 8. CALLBACKS
# --------------------------------------------------------------
# log_dir = Path("./tb_logs/lstm_improved")
# log_dir.mkdir(exist_ok=True, parents=True)

# cb_tb = callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
cb_es = callbacks.EarlyStopping(monitor="val_mae", patience=PATIENCE_ES,
                                restore_best_weights=True, min_delta=1e-4)
cb_lr = callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.3,
                                    patience=PATIENCE_LR, min_lr=1e-6)
# cb_cp = callbacks.ModelCheckpoint(
#     filepath=str(log_dir / "best_model.h5"),
#     monitor="val_mae", save_best_only=True, save_weights_only=False
# )

# --------------------------------------------------------------
# 9. TRAIN
# --------------------------------------------------------------
history = model.fit(
    x=[X_dyn_train, X_stat_train],
    y=y_train_s,
    validation_data=([X_dyn_test, X_stat_test], y_test_s),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,                      # <--- ADD THIS
    callbacks=[cb_es, cb_lr],
    verbose=2
)

# --------------------------------------------------------------
# 10. PREDICT & INVERSE LOG
# --------------------------------------------------------------
y_pred_log = model.predict([X_dyn_test, X_stat_test], verbose=0).flatten()
y_pred = np.expm1(y_pred_log)  # inverse of log1p

# --------------------------------------------------------------
# 11. METRICS
# --------------------------------------------------------------
mae  = mean_absolute_error(y_orig_test, y_pred)
mape = mean_absolute_percentage_error(y_orig_test, y_pred) * 100
r2   = r2_score(y_orig_test, y_pred)

print("\n" + "="*60)
print("FINAL METRICS (mg/dL)")
print(f"MAE  : {mae:6.2f}")
print(f"MAPE : {mape:6.2f}%")
print(f"R²   : {r2:6.3f}")
print("="*60)

# --------------------------------------------------------------
# 12. CLARKE ERROR GRID
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
# 13. PLOTS: Loss, Scatter, Bland-Altman
# --------------------------------------------------------------
# Loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train MAE')
plt.plot(history.history['val_loss'], label='Val MAE')
plt.title('Training / Validation MAE (log scale)')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# Scatter
plt.figure(figsize=(7,7))
plt.scatter(y_orig_test, y_pred, alpha=0.6, s=30, edgecolor='k')
lim = [y_orig_test.min(), y_orig_test.max()]
plt.plot(lim, lim, 'k--', lw=1)
slope, intercept = np.polyfit(y_orig_test, y_pred, 1)
plt.plot(lim, slope*np.array(lim) + intercept, 'r-', lw=2,
         label=f'Trend (slope={slope:.2f})')
plt.xlabel('Actual BG'); plt.ylabel('Predicted BG'); plt.title('Actual vs Predicted')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# Bland-Altman
# diff = y_pred - y_orig_test
# mean_diff, std_diff = diff.mean(), diff.std()
# plt.figure(figsize=(8,5))
# plt.scatter((y_orig_test + y_pred)/2, diff, alpha=0.6, s=30, edgecolor='k')
# plt.axhline(mean_diff, color='gray', linestyle='--')
# plt.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--')
# plt.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--')
# plt.xlabel('Mean'); plt.ylabel('Pred - Actual'); plt.title('Bland-Altman')
# plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# --------------------------------------------------------------
# 14. SAVE
# --------------------------------------------------------------
save_dir = Path("./model_weights")
save_dir.mkdir(exist_ok=True)

save_file = save_dir / "lstm_model_15s_nonlin.weights.h5"
model.save(save_file)

# np.savez(os.path.join(save_dir, "scalers.npz"),
#          dyn_mean=scaler_dyn.mean_, dyn_scale=scaler_dyn.scale_,
#          stat_mean=scaler_stat.mean_, stat_scale=scaler_stat.scale_,
#          y_mean=scaler_y.mean_, y_scale=scaler_y.scale_)

print(f"\nModel saved to {save_file}")