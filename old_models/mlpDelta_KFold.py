# Blood Glucose Regression with Group K-Fold Validation

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from helper_scripts.cega import cega


# =========================
# CONFIG
# =========================
DATAFILE = "processed_data/delta_vitaldb_ppg_extracted_features_15s_5minwin.csv"

GLUC = "gluc"
ID = "caseid"

N_FOLDS = 5

SCALE = "log"     # "log" or "standard"
FEATURES = "important"

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
# Load dataset
# =========================
bg_df = pd.read_csv(DATAFILE)

print(f"Loaded dataset: {DATAFILE}")
print(f"Shape: {bg_df.shape}")

groups = bg_df[ID].values


# =========================
# Feature setup
# =========================
drop_cols = [
    GLUC, ID,
    "ppg_freq","ppg_std","ppg_mean",
    "ppg_first_deriv_min","bmi",
    "ppg_spectral_entropy",
    "pwm_rise_time","pwm_decay_time",
    "pwm_max_slope","pwm_min_slope",
    "delta_ppg_freq"
]

all_features = [c for c in bg_df.columns if c not in drop_cols]

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
X = bg_df[features].values.astype(np.float32)
y = bg_df[GLUC].values.astype(np.float32)


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
            clipnorm=1.0
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
# Group K-Fold
# =========================
gkf = GroupKFold(n_splits=N_FOLDS)

fold_r2 = []
fold_mae = []
fold_mape = []

best_fold = None
best_model = None
best_mae = float("inf")
best_pred = None
best_y = None


print("\nStarting Group K-Fold Validation\n")


for fold,(train_idx,test_idx) in enumerate(gkf.split(X,y,groups)):

    print(f"\n===== Fold {fold+1}/{N_FOLDS} =====")

    set_all_seeds(1000 + fold)
    K.clear_session()

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]


    # =========================
    # Scaling
    # =========================
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # =========================
    # Log transform
    # =========================
    if SCALE == "log":

        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)

        y_train_fit = y_train_log

    else:

        y_train_fit = y_train


    # =========================
    # Histogram weights
    # =========================
    NUM_BINS = 80
    ALPHA = 0.25

    target_for_weights = y_train_fit

    counts,bin_edges = np.histogram(target_for_weights,bins=NUM_BINS)

    counts += 1

    bin_weights = (1.0/counts)**ALPHA
    bin_weights /= np.mean(bin_weights)

    bin_indices = np.digitize(target_for_weights,bin_edges[:-1],right=True)

    sample_weights = bin_weights[bin_indices-1]

    sample_weights = np.clip(sample_weights,0.25,4.0)


    # =========================
    # Train
    # =========================
    model = build_model(X_train.shape[1])

    model.fit(
        X_train,
        y_train_fit,
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        callbacks=callbacks_list,
        sample_weight=sample_weights,
        verbose=0
    )


    # =========================
    # Predict
    # =========================
    preds = model.predict(X_test).flatten()

    if SCALE == "log":
        preds = np.expm1(preds)


    # =========================
    # Metrics
    # =========================
    r2 = r2_score(y_test,preds)
    mae = np.mean(np.abs(y_test-preds))
    mape = mean_absolute_percentage_error(y_test,preds)*100

    fold_r2.append(r2)
    fold_mae.append(mae)
    fold_mape.append(mape)

    print(f"R²  : {r2:.3f}")
    print(f"MAE : {mae:.2f}")
    print(f"MAPE: {mape:.2f}")


    if mae < best_mae:

        best_mae = mae
        best_model = model
        best_pred = preds
        best_y = y_test
        best_fold = fold + 1


# =========================
# Final results
# =========================
r2_mean,r2_std = mean_std(fold_r2)
mae_mean,mae_std = mean_std(fold_mae)
mape_mean,mape_std = mean_std(fold_mape)

print("\n===== CROSS VALIDATION RESULTS =====")

print(f"R²   : {r2_mean:.3f} ± {r2_std:.3f}")
print(f"MAE  : {mae_mean:.2f} ± {mae_std:.2f}")
print(f"MAPE : {mape_mean:.2f} ± {mape_std:.2f}")


print("\n===== BEST FOLD =====")

print(f"Fold : {best_fold}")
print(f"MAE  : {best_mae:.2f}")


# =========================
# CEGA
# =========================
print("\nGenerating Clarke Error Grid...")
cega(best_y,best_pred)

# Early exit to prevent quantization
exit()

# =========================
# Quantization
# =========================
def representative_dataset():

    for i in range(300):
        yield [X_train[i:i+1].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

os.makedirs("model_weights",exist_ok=True)

outfile = f"model_weights/delta_mlp_{SUFFIX}_int8.tflite"

with open(outfile,"wb") as f:
    f.write(tflite_model)

print(f"\nSaved quantized model: {outfile}")
print(f"Model size: {os.path.getsize(outfile)/1024:.1f} KB")