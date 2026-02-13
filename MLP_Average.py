# Neural network, blood glucose regression (MLP) — 10 trials + averaged metrics (group-safe splits)
import os
import time
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from cega import cega

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


# =========================
# Config
# =========================
N_TRIALS = 10

DATASET  = 'vitaldb'     # 'vitaldb' or 'physionet'
FEATURES = 'all'   # 'all' or 'important' (only applies to vitaldb)

SUFFIX = '15s'

DATAFILE = (
    f'processed_data/new_vitaldb_ppg_extracted_features_{SUFFIX}_5minwin.csv'
    if DATASET == 'vitaldb'
    else 'processed_data/physioNet_ppg_extracted_features_30s.csv'
)

GLUC = 'gluc' if DATASET == 'vitaldb' else 'glucose_mg_dl'   # Target variable
ID   = 'caseid' if DATASET == 'vitaldb' else 'patient_id'    # Grouping variable (no leakage)


# =========================
# Utilities
# =========================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def mean_std(x):
    x = np.array(x, dtype=np.float64)
    if len(x) <= 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))


# =========================
# Load data
# =========================
bg_df = pd.read_csv(DATAFILE)
print(f"Successfully loaded data from {DATASET} (shape: {bg_df.shape})")

groups_all = bg_df[ID].values  # used only for splitting, never fed into model


# =========================
# Split into dev (train+val) and test by group (patient/caseid)
# =========================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(bg_df, groups=groups_all))

df_train = bg_df.iloc[train_idx].copy()
df_test  = bg_df.iloc[test_idx].copy()

# IMPORTANT: Save group labels for dev set BEFORE dropping ID column
groups_dev = df_train[ID].values

print(f"Dev patients:   {df_train[ID].nunique()}")
print(f"Test patients:  {df_test[ID].nunique()}")
print(f"Dev rows:       {len(df_train):,}")
print(f"Test rows:      {len(df_test):,}")


# =========================
# Drop unwanted columns (from feature tables)
# NOTE: We can safely drop ID from df_train/df_test because groups_dev is saved above.
# =========================
drop_cols_vdb = [col for col in bg_df.columns if 'ecg' in col.lower()]
drop_cols_vdb.extend(['ppg_freq', 'ppg_first_deriv_min', 'caseid', 'bmi'])

drop_cols_physio = ['ppg_freq', 'patient_id']  # ppg_freq redundant, patient_id not a feature

drop_cols = drop_cols_vdb if DATASET == 'vitaldb' else drop_cols_physio

df_train = df_train.drop(columns=drop_cols, errors='ignore')
df_test  = df_test.drop(columns=drop_cols, errors='ignore')


# =========================
# Feature selection
# =========================
all_features_vdb = [
    'age', 'sex', 'preop_dm', 'weight', 'height',
    'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s',
    'ppg_std_pp_interval_s', 'ppg_auc',
    'ppg_first_deriv_max', 'ppg_entropy',
    'ppg_teager_energy', 'ppg_log_energy',
    'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy'
]

important_features_vdb = [
    'age', 'weight', 'height', 'preop_dm', 'ppg_mean_pp_interval_s',
    'ppg_std', 'ppg_teager_energy', 'ppg_skew',
    'ppg_iqr', 'ppg_entropy', 'ppg_first_deriv_max', 'ppg_std_pp_interval_s'
]

features_physio = [
    'sex', 'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s', 'ppg_std_pp_interval_s',
    'ppg_auc', 'ppg_first_deriv_max', 'ppg_first_deriv_min', 'ppg_entropy',
    'ppg_teager_energy', 'ppg_log_energy', 'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy'
]

if DATASET == 'vitaldb':
    features = all_features_vdb if FEATURES == 'all' else important_features_vdb
else:
    features = features_physio

# Safety check: ensure all features exist
missing_train = [c for c in features if c not in df_train.columns]
missing_test  = [c for c in features if c not in df_test.columns]
if missing_train or missing_test:
    raise ValueError(
        f"Missing feature columns.\n"
        f"Missing in train: {missing_train}\n"
        f"Missing in test : {missing_test}"
    )


# =========================
# Build X/y arrays
# =========================
X_train = df_train[features].values.astype(np.float32)
y_train = df_train[GLUC].values.astype(np.float32)

X_test  = df_test[features].values.astype(np.float32)
y_test  = df_test[GLUC].values.astype(np.float32)

# Scale only on training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training samples (dev): {X_train_scaled.shape[0]}, Features: {X_train_scaled.shape[1]}")
print(f"Testing samples:        {X_test_scaled.shape[0]}")


# =========================
# Split dev into train_actual + val (grouped, no leakage)
# =========================
print("Splitting dev data into actual train + validation sets...")

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_actual_idx, val_idx = next(gss_val.split(X_train_scaled, groups=groups_dev))

X_train_actual = X_train_scaled[train_actual_idx]
y_train_actual = y_train[train_actual_idx]

X_val = X_train_scaled[val_idx]
y_val = y_train[val_idx]

print(f"Actual training samples: {X_train_actual.shape[0]}")
print(f"Validation samples:      {X_val.shape[0]}")
print(f"Test samples:            {X_test_scaled.shape[0]}")


# =========================
# Model
# =========================
def build_small_glucose_model(input_dim: int):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.30),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.20),

        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0015),
        loss='mae',
        metrics=['mae', 'mape']
    )
    return model


callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=12,
        min_lr=1e-6,
        verbose=1
    )
]


# =========================
# 10 trials + average metrics
# =========================
input_dim = X_train_actual.shape[1]

trial_r2   = []
trial_mae  = []
trial_mape = []

best_trial_idx = None
best_mae = float("inf")
best_model = None
best_y_pred_test = None

print("\n" + "=" * 70)
print(f"Starting {N_TRIALS} MLP trials...")
print("=" * 70)

for t in range(N_TRIALS):
    print("\n" + "#" * 70)
    print(f"TRIAL {t+1}/{N_TRIALS}")
    print("#" * 70)

    # Reset TF graph state so trials don't accumulate memory
    K.clear_session()

    # Different seed each trial (keeps splits constant; changes init/shuffle/dropout)
    seed = 1000 + t
    set_all_seeds(seed)

    model = build_small_glucose_model(input_dim)

    # Print model summary only once
    if t == 0:
        model.summary()

    print("Starting model training...")
    model.fit(
        X_train_actual, y_train_actual,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        verbose=1,
        callbacks=callbacks_list
    )

    # Evaluate on test
    y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()

    r2_test = r2_score(y_test, y_pred_test)
    mae_test = float(np.mean(np.abs(y_test - y_pred_test)))
    mape_test = float(mean_absolute_percentage_error(y_test, y_pred_test) * 100)

    trial_r2.append(r2_test)
    trial_mae.append(mae_test)
    trial_mape.append(mape_test)

    print("\n" + "-" * 60)
    print(f"Trial {t+1} TEST metrics:")
    print(f"    R²  : {r2_test:.3f}")
    print(f"    MAE : {mae_test:.2f} mg/dL")
    print(f"    MAPE: {mape_test:.2f}%")
    print("-" * 60)

    # Track best model (by lowest MAE on test)
    if mae_test < best_mae:
        best_mae = mae_test
        best_trial_idx = t
        best_model = model
        best_y_pred_test = y_pred_test


# =========================
# Summary across trials
# =========================
r2_mean, r2_std     = mean_std(trial_r2)
mae_mean, mae_std   = mean_std(trial_mae)
mape_mean, mape_std = mean_std(trial_mape)

print("\n" + "=" * 70)
print(f"AVERAGED RESULTS over {N_TRIALS} trials (TEST set):")
print(f"    R²   : {r2_mean:.3f} ± {r2_std:.3f}")
print(f"    MAE  : {mae_mean:.2f} ± {mae_std:.2f} mg/dL")
print(f"    MAPE : {mape_mean:.2f} ± {mape_std:.2f} %")
print(f"Best trial: {best_trial_idx + 1} (MAE={best_mae:.2f} mg/dL)")
print("=" * 70)


# =========================
# CEGA (best trial only)
# =========================
print("\nGenerating Clarke Error Grid Analysis plot (best trial only)...")
cega(y_test, best_y_pred_test)


# =========================
# Quantization (best trial only)
# =========================
def representative_dataset():
    """
    Generator that yields validation samples for calibration.
    Must return list of arrays with shape [1, n_features], float32.
    """
    NUM_CALIBRATION_SAMPLES = 300
    num_samples = min(NUM_CALIBRATION_SAMPLES, len(X_val))
    for i in range(num_samples):
        yield [X_val[i:i+1].astype(np.float32)]

print("Starting TFLite quantization (best trial only)...")
start_quantization = time.time()

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

os.makedirs("model_weights", exist_ok=True)
OUTFILE = f"model_weights/mlp_allfeats_{SUFFIX}_{DATASET}_int8.tflite"

with open(OUTFILE, "wb") as f:
    f.write(tflite_model)

end_quantization = time.time()
print(f"Saved model as {OUTFILE}")
print(f"Model quantization took {end_quantization - start_quantization:.1f} seconds")
print(f"To convert .tflite to C, use: xxd -i {OUTFILE} > mlp.h")

if os.path.exists(OUTFILE):
    print(f"Quantized model size: {os.path.getsize(OUTFILE)/1024:.1f} KB")
else:
    print("Error: Unable to determine file size (does the quantized model exist?)")