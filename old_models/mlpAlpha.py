# ============================================================
# Neural Network Blood Glucose Regression
# Histogram-Based Sample Weighting + Alpha Tuning (Coarse + Fine)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from helper_scripts.cega import cega
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import random
import os

# ============================================================
# Reproducibility
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================================================
# Configuration
# ============================================================

DATAFILE = 'processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv'
GLUC = 'gluc'
ID = 'caseid'

IMPORTANT_FEATURES = [
    'age','weight','height','preop_dm',
    'ppg_mean_pp_interval_s','ppg_std','ppg_teager_energy',
    'ppg_skew','ppg_iqr','ppg_entropy',
    'ppg_first_deriv_max','ppg_std_pp_interval_s'
]

NUM_BINS = 50
ALPHA_COARSE = np.linspace(0.2, 1.0, 7)
FINE_RANGE = 0.15
FINE_POINTS = 5

# ============================================================
# Load Data
# ============================================================

bg_df = pd.read_csv(DATAFILE)
print(f"Loaded dataset shape: {bg_df.shape}")

groups = bg_df[ID].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(bg_df, groups=groups))

df_train = bg_df.iloc[train_idx].copy()
df_test = bg_df.iloc[test_idx].copy()

X_train = df_train[IMPORTANT_FEATURES].values.astype(np.float32)
y_train = df_train[GLUC].values.astype(np.float32)

X_test = df_test[IMPORTANT_FEATURES].values.astype(np.float32)
y_test = df_test[GLUC].values.astype(np.float32)

# ============================================================
# Scale Features
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# Train/Validation Split (Group Safe)
# ============================================================

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_actual_idx, val_idx = next(
    gss_val.split(X_train, groups=df_train[ID].values)
)

X_train_actual = X_train[train_actual_idx]
y_train_actual = y_train[train_actual_idx]

X_val = X_train[val_idx]
y_val = y_train[val_idx]

# ============================================================
# Log Transform Target
# ============================================================

y_train_log = np.log1p(y_train_actual)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

# ============================================================
# Model Builder
# ============================================================

def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.35),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.30),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mae',
        metrics=['mae']
    )
    return model

# ============================================================
# Histogram-Based Sample Weighting
# ============================================================

def compute_sample_weights(y_log, alpha):
    counts, bin_edges = np.histogram(y_log, bins=NUM_BINS)
    counts = counts + 1

    bin_weights = (1.0 / counts) ** alpha
    bin_weights /= np.mean(bin_weights)

    bin_indices = np.digitize(y_log, bin_edges[:-1], right=True)
    weights = bin_weights[bin_indices - 1]

    weights = np.clip(weights, 0.5, 5.0)
    return weights

# ============================================================
# Alpha Evaluation Function
# ============================================================

def evaluate_alpha(alpha):
    sample_weights = compute_sample_weights(y_train_log, alpha)

    model = build_model(X_train_actual.shape[1])

    model.fit(
        X_train_actual, y_train_log,
        validation_data=(X_val, y_val_log),
        epochs=60,
        batch_size=32,
        verbose=0,
        sample_weight=sample_weights
    )

    y_val_pred = np.expm1(
        model.predict(X_val, verbose=0).flatten()
    )

    overall_mae = np.mean(np.abs(y_val - y_val_pred))

    tail_mask = (y_val < 70) | (y_val > 180)
    if np.sum(tail_mask) > 0:
        tail_mae = np.mean(
            np.abs(y_val[tail_mask] - y_val_pred[tail_mask])
        )
    else:
        tail_mae = overall_mae

    score = 0.5 * overall_mae + 0.5 * tail_mae
    return score

# ============================================================
# Coarse Alpha Search
# ============================================================

print("\nStarting coarse alpha search...")

best_alpha = None
best_score = np.inf

for alpha in ALPHA_COARSE:
    score = evaluate_alpha(alpha)
    print(f"Alpha {alpha:.2f} → Score {score:.3f}")

    if score < best_score:
        best_score = score
        best_alpha = alpha

print(f"\nBest coarse alpha: {best_alpha:.3f}")

# ============================================================
# Fine Alpha Refinement
# ============================================================

fine_grid = np.linspace(
    max(0, best_alpha - FINE_RANGE),
    best_alpha + FINE_RANGE,
    FINE_POINTS
)

print("\nStarting fine alpha refinement...")

for alpha in fine_grid:
    score = evaluate_alpha(alpha)
    print(f"[Fine] Alpha {alpha:.3f} → Score {score:.3f}")

    if score < best_score:
        best_score = score
        best_alpha = alpha

print(f"\nFinal selected alpha: {best_alpha:.4f}")
print("="*60)

# ============================================================
# Final Training With Best Alpha
# ============================================================

final_weights = compute_sample_weights(y_train_log, best_alpha)

callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6
    )
]

print("Training final model...")

final_model = build_model(X_train_actual.shape[1])

final_model.fit(
    X_train_actual, y_train_log,
    validation_data=(X_val, y_val_log),
    epochs=120,
    batch_size=32,
    callbacks=callbacks_list,
    sample_weight=final_weights,
    verbose=1
)

# ============================================================
# Test Evaluation
# ============================================================

y_pred = np.expm1(
    final_model.predict(X_test, verbose=0).flatten()
)

r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("\n" + "="*60)
print("FINAL TEST RESULTS")
print(f"Best Alpha : {best_alpha:.4f}")
print(f"R²  : {r2:.3f}")
print(f"MAE : {mae:.2f} mg/dL")
print(f"MAPE: {mape:.2f}%")
print("="*60)

print("Generating CEGA...")
cega(y_test, y_pred)