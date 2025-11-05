# --------------------------------------------------------------
#  vitaldb_glucose_lstm_full.py
#  LSTM + static-merge + full evaluation (Clarke, plots, etc.)
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

# --------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------
FILTERED_FILE = "./processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv"   # <-- change if needed
CASEID_COL    = "caseid"
TARGET_COL    = "preop_gluc"

# static features (constant per subject)
STATIC_COLS   = ["age", "sex", "preop_dm", "weight", "height"]

# hyper-parameters
LSTM_UNITS    = 64
DENSE_UNITS   = 32
DROPOUT       = 0.2
BATCH_SIZE    = 64
EPOCHS        = 200
PATIENCE_ES   = 15
PATIENCE_LR   = 7
SEED          = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------
# 2. LOAD & GROUP BY SUBJECT
# --------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(FILTERED_FILE).dropna()
print(f"Raw rows: {df.shape[0]}")

# sanity – target must be constant per caseid
assert df.groupby(CASEID_COL)[TARGET_COL].nunique().max() == 1

dynamic_cols = [c for c in df.columns if c not in (CASEID_COL, TARGET_COL, *STATIC_COLS)]
static_cols  = STATIC_COLS

print(f"Dynamic features : {len(dynamic_cols)}")
print(f"Static  features : {len(static_cols)}")

subjects = df[CASEID_COL].unique()
n_subj   = len(subjects)

# 64 rows per subject → T = 64
T = df.groupby(CASEID_COL).size().max()
assert T == 64, f"Expected 64 rows per subject, got {T}"
D = len(dynamic_cols)
S = len(static_cols)

X_dyn  = np.zeros((n_subj, T, D), dtype=np.float32)
X_stat = np.zeros((n_subj, S), dtype=np.float32)
y      = np.zeros((n_subj, 1), dtype=np.float32)

for i, subj in enumerate(subjects):
    sub_df = df[df[CASEID_COL] == subj].sort_values(by=CASEID_COL)  # any order ok
    X_dyn[i]   = sub_df[dynamic_cols].values
    X_stat[i]  = sub_df[static_cols].iloc[0].values
    y[i]       = sub_df[TARGET_COL].iloc[0]

# --------------------------------------------------------------
# 3. SUBJECT-LEVEL SPLIT
# --------------------------------------------------------------
train_ids, test_ids = train_test_split(subjects, test_size=0.20, random_state=SEED)
train_mask = np.isin(subjects, train_ids)

X_dyn_train, X_dyn_test   = X_dyn[train_mask], X_dyn[~train_mask]
X_stat_train, X_stat_test = X_stat[train_mask], X_stat[~train_mask]
y_train, y_test           = y[train_mask], y[~train_mask]

print(f"Train subjects: {X_dyn_train.shape[0]}, Test: {X_dyn_test.shape[0]}")

# --------------------------------------------------------------
# 4. SCALING (fit on train only)
# --------------------------------------------------------------
# dynamic
scaler_dyn = StandardScaler()
n_tr, T_tr, D_tr = X_dyn_train.shape
X_dyn_train = scaler_dyn.fit_transform(X_dyn_train.reshape(-1, D_tr)).reshape(n_tr, T_tr, D_tr)
X_dyn_test  = scaler_dyn.transform(X_dyn_test.reshape(-1, D_tr)).reshape(-1, T, D_tr)

# static
scaler_stat = StandardScaler()
X_stat_train = scaler_stat.fit_transform(X_stat_train)
X_stat_test  = scaler_stat.transform(X_stat_test)

# target
scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train)
y_test_s  = scaler_y.transform(y_test)

# keep original for reporting
y_train_orig, y_test_orig = y_train.copy().flatten(), y_test.copy().flatten()

# --------------------------------------------------------------
# 5. MODEL
# --------------------------------------------------------------
def build_lstm(timesteps, dyn_features, static_features):
    # ---- dynamic -------------------------------------------------
    dyn_in = layers.Input(shape=(timesteps, dyn_features), name="dyn")
    x = layers.Masking(mask_value=0.0)(dyn_in)                 # safety
    x = layers.LSTM(LSTM_UNITS,
                    dropout=DROPOUT,
                    recurrent_dropout=DROPOUT,
                    return_sequences=False)(x)         # (batch, LSTM_UNITS)

    # ---- static --------------------------------------------------
    stat_in = layers.Input(shape=(static_features,), name="stat")
    s = layers.Dense(DENSE_UNITS, activation="relu")(stat_in)
    s = layers.Dropout(DROPOUT)(s)

    # ---- merge ---------------------------------------------------
    merged = layers.Concatenate()([x, s])
    merged = layers.Dense(64, activation="relu")(merged)
    merged = layers.Dropout(DROPOUT)(merged)

    out = layers.Dense(1, activation="linear", name="glucose")(merged)

    model = Model(inputs=[dyn_in, stat_in], outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

model = build_lstm(timesteps=T, dyn_features=D, static_features=S)
model.summary()

# --------------------------------------------------------------
# 6. CALLBACKS
# --------------------------------------------------------------
cb_es = callbacks.EarlyStopping(monitor="val_mae", patience=PATIENCE_ES,
                                restore_best_weights=True, min_delta=1e-4)
cb_lr = callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5,
                                    patience=PATIENCE_LR, min_lr=1e-7)

# --------------------------------------------------------------
# 7. TRAIN
# --------------------------------------------------------------
history = model.fit(
    [X_dyn_train, X_stat_train], y_train_s,
    validation_data=([X_dyn_test, X_stat_test], y_test_s),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[cb_es, cb_lr],
    verbose=2
)

# --------------------------------------------------------------
# 8. PREDICT & INVERSE-SCALE
# --------------------------------------------------------------
y_pred_s = model.predict([X_dyn_test, X_stat_test], verbose=0).flatten()
y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()

# --------------------------------------------------------------
# 9. METRICS
# --------------------------------------------------------------
mae  = mean_absolute_error(y_test_orig, y_pred)
mape = mean_absolute_percentage_error(y_test_orig, y_pred) * 100
r2   = r2_score(y_test_orig, y_pred)

print("\n=== FINAL METRICS (mg/dL) ===")
print(f"MAE  : {mae:6.2f}")
print(f"MAPE : {mape:6.2f}%")
print(f"R²   : {r2:6.3f}")

# --------------------------------------------------------------
# 10. CLARKE ERROR GRID
# --------------------------------------------------------------
def clarke_zone(ref, pred):
    r, p = float(ref), float(pred)
    if (r <= 70 and p <= 70) or (0.8*r <= p <= 1.2*r):
        return 'A'
    if (130 < r <= 180 and 1.4*(r-130) >= p) or (70 < r <= 280 and p >= (r+110)):
        return 'C'
    if (r <= 70 and 70 < p <= 180) or (r >= 240 and 70 <= p <= 180):
        return 'D'
    if (r <= 70 and p > 180) or (r > 180 and p <= 70):
        return 'E'
    return 'B'

zones = {'A':0, 'B':0, 'C':0, 'D':0, 'E':0}
points = []
for r, p in zip(y_test_orig, y_pred):
    z = clarke_zone(r, p)
    zones[z] += 1
    points.append((r, p, z))

total = len(y_test_orig)
print("\nClarke Error Grid:")
for z, cnt in zones.items():
    print(f"  Zone {z}: {100*cnt/total:5.2f}% ({cnt}/{total})")

# ---- Plot Clarke Grid ------------------------------------------------
plt.figure(figsize=(10,10))
colors = {'A':'green','B':'yellow','C':'orange','D':'red','E':'purple'}
labels = {k:f"{k}: {v}" for k,v in {
    'A':'Clinically Accurate','B':'Benign Errors','C':'Overcorrection',
    'D':'Dangerous Failure','E':'Erroneous Treatment'}.items()}

max_val = 400
for z in 'ABCDE':
    zp = [(r,p) for r,p,zz in points if zz==z]
    if zp:
        rs, ps = zip(*zp)
        plt.scatter(rs, ps, c=colors[z], label=f"{labels[z]} ({zones[z]})",
                    alpha=0.7, edgecolors='k', s=60)

# perfect line
plt.plot([0,max_val],[0,max_val], 'k--', lw=1, label='y = x')

# Zone A fill
x = np.linspace(0, max_val, 500)
plt.fill_between(x, 0.8*x, 1.2*x, where=(x<=70)|(x>=70), color='green', alpha=0.1)
plt.fill_between(x, 0, 70, where=x<=70, color='green', alpha=0.1)

# Zone C
plt.fill([70,70,290,290], [180,400,400,180], color='orange', alpha=0.4)
plt.fill([130,180,180,130], [0,0,70,70], color='orange', alpha=0.4)

# Zone D
plt.fill([0,70,70,0], [70,70,180,180], color='red', alpha=0.1)
plt.fill([240,400,400,240], [70,70,180,180], color='red', alpha=0.1)

# Zone E
plt.fill([0,70,70,0], [180,180,400,400], color='purple', alpha=0.1)
plt.fill([180,400,400,180], [0,0,70,70], color='purple', alpha=0.1)

plt.xlim(0, max_val); plt.ylim(0, max_val)
plt.xlabel('Reference Glucose (mg/dL)'); plt.ylabel('Predicted Glucose (mg/dL)')
plt.title('Clarke Error Grid – LSTM')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 11. LOSS CURVES
# --------------------------------------------------------------
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'],     label='Train MSE')
plt.plot(history.history['val_loss'], label='Val   MSE')
plt.title('Training / Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE (scaled)')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 12. ACTUAL vs PREDICTED + TREND
# --------------------------------------------------------------
plt.figure(figsize=(7,7))
plt.scatter(y_test_orig, y_pred, alpha=0.6, s=30, edgecolor='k', label='Predictions')
lim = [min(y_test_orig.min(), y_pred.min()),
       max(y_test_orig.max(), y_pred.max())]
plt.plot(lim, lim, 'k--', lw=1, label='Ideal (y=x)')

slope, intercept = np.polyfit(y_test_orig, y_pred, 1)
x_reg = np.array(lim)
plt.plot(x_reg, slope*x_reg + intercept, 'r-', lw=2,
         label=f'Trend (slope={slope:.2f})')

plt.xlabel('Actual BG (mg/dL)')
plt.ylabel('Predicted BG (mg/dL)')
plt.title('Actual vs Predicted – LSTM')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 13. BLAND-ALTMAN
# --------------------------------------------------------------
# diff = y_pred - y_test_orig
# mean_diff = diff.mean()
# std_diff  = diff.std()

# plt.figure(figsize=(8,5))
# plt.scatter((y_test_orig + y_pred)/2, diff, alpha=0.6, s=30, edgecolor='k')
# plt.axhline(mean_diff, color='gray', linestyle='--',
#             label=f'Mean diff = {mean_diff:+.2f}')
# plt.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--',
#             label=f'+1.96σ = {mean_diff+1.96*std_diff:+.2f}')
# plt.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--',
#             label=f'-1.96σ = {mean_diff-1.96*std_diff:+.2f}')
# plt.xlabel('Mean of Actual & Predicted (mg/dL)')
# plt.ylabel('Prediction – Actual (mg/dL)')
# plt.title('Bland-Altman Plot – LSTM')
# plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
# plt.show()

# --------------------------------------------------------------
# 14. SAVE MODEL & SCALERS
# --------------------------------------------------------------
save_dir = "./model_weights"
os.makedirs(save_dir, exist_ok=True)
model_name = "/lstm_model_15s.weights.h5"
save_file = os.path.join(save_dir, model_name)
model.save(save_file)

# np.savez(os.path.join(save_dir, "scalers.npz"),
#          dyn_mean=scaler_dyn.mean_, dyn_scale=scaler_dyn.scale_,
#          stat_mean=scaler_stat.mean_, stat_scale=scaler_stat.scale_,
#          y_mean=scaler_y.mean_, y_scale=scaler_y.scale_)

print(f"\nModel saved to {save_file}")