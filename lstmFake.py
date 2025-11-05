# --------------------------------------------------------------
#  debug_signal.py — 100% FIXED, NO SHAPE ERRORS
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, Model

# --------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------
df = pd.read_csv("./processed_data/vitaldb_ppg_ecg_extracted_features_15s_nonlin.csv").dropna()
CASEID_COL = "caseid"
TARGET_COL = "preop_gluc"
STATIC_COLS = ["age", "sex", "preop_dm", "weight", "height"]

df["log_gluc"] = np.log1p(df[TARGET_COL])
dynamic_cols = [c for c in df.columns if c not in (CASEID_COL, TARGET_COL, "log_gluc", *STATIC_COLS)]

# --------------------------------------------------------------
# 2. BUILD ARRAYS + INJECT FAKE FEATURE
# --------------------------------------------------------------
subjects = df[CASEID_COL].unique()
n_subj = len(subjects)
MAX_T = 64
D = len(dynamic_cols)

X_dyn = np.zeros((n_subj, MAX_T, D + 1), dtype=np.float32)  # +1 for fake
y_log = np.zeros((n_subj, 1), dtype=np.float32)              # ← (n, 1)

for i, subj in enumerate(subjects):
    sub_df = df[df[CASEID_COL] == subj]
    seq = sub_df[dynamic_cols].values.astype(np.float32)
    gluc = sub_df[TARGET_COL].iloc[0]

    # FAKE FEATURE: glucose / 100 + noise
    fake = (gluc / 100.0) + np.random.normal(0, 0.1, size=(len(seq), 1))
    seq = np.hstack([seq, fake])

    # Pad/truncate
    if len(seq) >= MAX_T:
        seq = seq[:MAX_T]
    else:
        pad_width = MAX_T - len(seq)
        seq = np.pad(seq, ((0, pad_width), (0, 0)), mode="constant")

    X_dyn[i] = seq
    y_log[i] = sub_df["log_gluc"].iloc[0]

# --------------------------------------------------------------
# 3. SPLIT
# --------------------------------------------------------------
train_ids, test_ids = train_test_split(subjects, test_size=0.2, random_state=42)
train_mask = np.isin(subjects, train_ids)

X_dyn_train, X_dyn_test = X_dyn[train_mask], X_dyn[~train_mask]
y_log_train, y_log_test = y_log[train_mask], y_log[~train_mask]
y_orig_test = np.expm1(y_log_test).flatten()

# --------------------------------------------------------------
# 4. SCALE
# --------------------------------------------------------------
scaler_dyn = StandardScaler()
X_dyn_train = scaler_dyn.fit_transform(X_dyn_train.reshape(-1, D+1)).reshape(-1, MAX_T, D+1)
X_dyn_test  = scaler_dyn.transform(X_dyn_test.reshape(-1, D+1)).reshape(-1, MAX_T, D+1)

scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_log_train)  # ← (n_train, 1)
y_test_s  = scaler_y.transform(y_log_test)        # ← (n_test, 1)

# --------------------------------------------------------------
# 5. MODEL — OUTPUT (batch, 1)
# --------------------------------------------------------------
def build_model():
    inp = layers.Input(shape=(MAX_T, D+1))
    x = layers.GlobalAveragePooling1D()(inp)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)  # ← (batch, 1)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    return model

model = build_model()

# --------------------------------------------------------------
# 6. TRAIN
# --------------------------------------------------------------
history = model.fit(
    X_dyn_train, y_train_s,
    validation_data=(X_dyn_test, y_test_s),
    epochs=50,
    batch_size=64,
    verbose=1
)

# --------------------------------------------------------------
# 7. PREDICT
# --------------------------------------------------------------
y_pred_s = model.predict(X_dyn_test, verbose=0)                    # ← (n, 1)
y_pred_log = scaler_y.inverse_transform(y_pred_s)                  # ← (n, 1)
y_pred = np.expm1(y_pred_log).flatten()                            # ← (n,)

mae = mean_absolute_error(y_orig_test, y_pred)
slope, _ = np.polyfit(y_orig_test, y_pred, 1)

print(f"\nFAKE FEATURE TEST → MAE: {mae:.2f}, Slope: {slope:.2f}")

# --------------------------------------------------------------
# 8. PLOT
# --------------------------------------------------------------
plt.figure(figsize=(7,7))
plt.scatter(y_orig_test, y_pred, alpha=0.6)
lim = [y_orig_test.min(), y_orig_test.max()]
plt.plot(lim, lim, 'k--')
plt.plot(lim, slope*np.array(lim) + _, 'r-', label=f"Slope={slope:.2f}")
plt.xlabel("Actual BG"); plt.ylabel("Predicted BG")
plt.legend(); plt.grid(alpha=0.3); plt.show()