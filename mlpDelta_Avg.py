# Neural network blood glucose regression (MLP)
# 10 trials + delta dataset + log transform + histogram weighting + stability fixes

import os
import random
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from helper_scripts.cega import cega


# =========================
# CONFIG
# =========================
N_TRIALS = 5

DATAFILE = "processed_data/delta_vitaldb_ppg_extracted_features_15s_5minwin.csv"

GLUC = "gluc"
ID   = "caseid"

FEATURES = "important"     # "all" or "important"
SCALE    = "log"           # "log" or "standard"

DERIVE_FEATS = False
NUM_FEATS    = 15

SUFFIX = "15s"


# =========================
# Utilities
# =========================
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def mean_std(x):
    x = np.array(x)
    return float(x.mean()), float(x.std(ddof=1))


# =========================
# Load Dataset
# =========================
bg_df = pd.read_csv(DATAFILE)

print(f"Loaded dataset: {DATAFILE}")
print(f"Shape: {bg_df.shape}")

groups_all = bg_df[ID].values


# =========================
# Train/Test Split (group safe)
# =========================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(
    gss.split(bg_df, groups=groups_all)
)

df_train = bg_df.iloc[train_idx].copy()
df_test  = bg_df.iloc[test_idx].copy()

groups_dev = df_train[ID].values

print(f"Dev patients:  {df_train[ID].nunique()}")
print(f"Test patients: {df_test[ID].nunique()}")
print(f"Dev rows:      {len(df_train):,}")
print(f"Test rows:     {len(df_test):,}")


# =========================
# Drop unwanted columns
# =========================
drop_cols = [
    GLUC, ID,
    "ppg_freq", "ppg_std", "ppg_mean",
    "ppg_first_deriv_min", "bmi",
    "ppg_spectral_entropy",
    "pwm_rise_time", "pwm_decay_time",
    "pwm_max_slope", "pwm_min_slope",
    "delta_ppg_freq"
]

all_features = [c for c in bg_df.columns if c not in drop_cols]


# =========================
# Optional RandomForest feature derivation
# =========================
if DERIVE_FEATS:

    print("\nComputing RandomForest feature importance...")

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(df_train[all_features], df_train[GLUC])

    importances = pd.Series(
        rf.feature_importances_,
        index=all_features
    ).sort_values(ascending=False)

    important_features = importances.head(NUM_FEATS).index.tolist()

else:

    important_features = [
        'height','weight','age','preop_dm',
        'ppg_mean_pp_interval_s','pwm_pulse_amplitude',
        'ppg_teager_energy','ppg_skew','ppg_iqr',
        'pwm_pulse_width','ppg_first_deriv_max',
        'ppg_entropy','ppg_std_pp_interval_s',
        'sex','delta_ppg_mean_pp_interval_s'
    ]

features = all_features if FEATURES == "all" else important_features


# =========================
# Build arrays
# =========================
X_train = df_train[features].values.astype(np.float32)
y_train = df_train[GLUC].values.astype(np.float32)

X_test  = df_test[features].values.astype(np.float32)
y_test  = df_test[GLUC].values.astype(np.float32)


# =========================
# Feature scaling
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training samples: {X_train_scaled.shape[0]}")
print(f"Testing samples:  {X_test_scaled.shape[0]}")


# =========================
# Train / Validation Split
# =========================
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_actual_idx, val_idx = next(
    gss_val.split(X_train_scaled, groups=groups_dev)
)

X_train_actual = X_train_scaled[train_actual_idx]
y_train_actual = y_train[train_actual_idx]

X_val = X_train_scaled[val_idx]
y_val = y_train[val_idx]


# =========================
# Optional Log Transform
# =========================
if SCALE == "log":

    y_train_actual_log = np.log1p(y_train_actual)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)


# =========================
# Model
# =========================
def build_model(input_dim):

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.35),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(16, activation="relu"),

        layers.Dense(1)
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=1e-4,
            clipnorm=1.0     # gradient clipping
        ),
        loss="mae",
        metrics=["mae","mape"]
    )

    return model


callbacks_list = [

    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=12,
        min_lr=1e-6
    )
]


# =========================
# Trial Loop
# =========================
input_dim = X_train_actual.shape[1]

trial_r2   = []
trial_mae  = []
trial_mape = []

best_model = None
best_mae = float("inf")
best_r2 = None
best_mape = None
best_trial_idx = None
best_y_pred = None


print("\nStarting trials...\n")

for t in range(N_TRIALS):

    print(f"Trial {t+1}/{N_TRIALS}")

    K.clear_session()
    set_all_seeds(1000 + t)

    model = build_model(input_dim)

    y_train_fit = y_train_actual_log if SCALE == "log" else y_train_actual
    y_val_fit   = y_val_log if SCALE == "log" else y_val


    # =========================
    # Histogram weighting (recomputed each trial)
    # =========================
    NUM_BINS = 80
    ALPHA = 0.25

    target_for_weights = y_train_fit

    counts, bin_edges = np.histogram(target_for_weights, bins=NUM_BINS)

    counts = counts + 1

    bin_weights = (1.0 / counts) ** ALPHA
    bin_weights = bin_weights / np.mean(bin_weights)

    bin_indices = np.digitize(target_for_weights, bin_edges[:-1], right=True)

    sample_weights = bin_weights[bin_indices - 1]

    # Clip extreme weights (stability fix)
    sample_weights = np.clip(sample_weights, 0.25, 4.0)


    # =========================
    # Train
    # =========================
    model.fit(
        X_train_actual,
        y_train_fit,
        validation_data=(X_val, y_val_fit),
        epochs=60,
        batch_size=32,
        callbacks=callbacks_list,
        sample_weight=sample_weights,
        verbose=0
    )


    # =========================
    # Evaluate
    # =========================
    preds = model.predict(X_test_scaled).flatten()

    if SCALE == "log":
        preds = np.clip(preds, 0, 5.9) # Clip predictions to prevent exploding gradients after exponentiation

        preds = np.expm1(preds)

    r2   = r2_score(y_test, preds)
    mae  = np.mean(np.abs(y_test - preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    trial_r2.append(r2)
    trial_mae.append(mae)
    trial_mape.append(mape)

    print(f"R²  : {r2:.3f}")
    print(f"MAE : {mae:.2f}")
    print(f"MAPE: {mape:.2f}\n")


    if mae < best_mae:

        best_mae = mae
        best_r2 = r2
        best_mape = mape
        best_model = model
        best_y_pred = preds
        best_trial_idx = t + 1


# =========================
# Average results
# =========================
r2_mean, r2_std = mean_std(trial_r2)
mae_mean, mae_std = mean_std(trial_mae)
mape_mean, mape_std = mean_std(trial_mape)

print("\n===== AVERAGED RESULTS =====")

print(f"R²   : {r2_mean:.3f} ± {r2_std:.3f}")
print(f"MAE  : {mae_mean:.2f} ± {mae_std:.2f}")
print(f"MAPE : {mape_mean:.2f} ± {mape_std:.2f}")


# =========================
# Best Trial
# =========================
print("\n===== BEST TRIAL RESULTS =====")

print(f"Trial : {best_trial_idx}")
print(f"R²    : {best_r2:.3f}")
print(f"MAE   : {best_mae:.2f} mg/dL")
print(f"MAPE  : {best_mape:.2f}%")

print("\nGenerating Clarke Error Grid Analysis...")
cega(y_test, best_y_pred)


# Early exit to prevent quantization
exit()

# =========================
# Quantization
# =========================
def representative_dataset():

    for i in range(min(300, len(X_val))):

        yield [X_val[i:i+1].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

os.makedirs("model_weights", exist_ok=True)

outfile = f"model_weights/delta_mlp_{SUFFIX}_int8.tflite"

with open(outfile, "wb") as f:
    f.write(tflite_model)

print(f"\nSaved quantized model: {outfile}")
print(f"Model size: {os.path.getsize(outfile)/1024:.1f} KB")