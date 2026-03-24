# ============================================================
# Neural Network Blood Glucose Regression
# Patient-Specific Scaling (Quick Test)
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import random

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_percentage_error

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

from helper_scripts.cega import cega


# ============================================================
# CONFIG
# ============================================================

SEED = 42
DATAFILE = 'processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv'

GLUC = 'gluc'
ID   = 'caseid'


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

bg_df = pd.read_csv(DATAFILE)

print(f"Loaded dataset: {bg_df.shape}")


# ============================================================
# FEATURES
# ============================================================

demographic_features = [
    'age','weight','height','preop_dm'
]

signal_features = [
    'ppg_mean_pp_interval_s',
    'ppg_std',
    'ppg_teager_energy',
    'ppg_skew',
    'ppg_iqr',
    'ppg_entropy',
    'ppg_first_deriv_max',
    'ppg_std_pp_interval_s'
]

features = demographic_features + signal_features


# ============================================================
# PATIENT-SPECIFIC SCALING
# ============================================================

print("\nApplying patient-specific normalization...")

bg_scaled = bg_df.copy()

for patient_id, group in bg_df.groupby(ID):

    idx = group.index

    means = group[signal_features].mean()
    stds  = group[signal_features].std()

    stds = stds.replace(0,1)

    bg_scaled.loc[idx, signal_features] = (group[signal_features] - means) / stds


print("Patient normalization complete.")


# ============================================================
# GROUP SPLIT (PATIENT SAFE)
# ============================================================

groups = bg_scaled[ID].values

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED
)

train_idx, test_idx = next(gss.split(bg_scaled, groups=groups))

df_train = bg_scaled.iloc[train_idx]
df_test  = bg_scaled.iloc[test_idx]

print(f"\nDev patients : {df_train[ID].nunique()}")
print(f"Test patients: {df_test[ID].nunique()}")


# ============================================================
# MATRICES
# ============================================================

X_train_full = df_train[features].values.astype(np.float32)
y_train_full = df_train[GLUC].values.astype(np.float32)

X_test = df_test[features].values.astype(np.float32)
y_test = df_test[GLUC].values.astype(np.float32)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

gss_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=SEED
)

train_idx, val_idx = next(
    gss_val.split(X_train_full, groups=df_train[ID].values)
)

X_train = X_train_full[train_idx]
y_train = y_train_full[train_idx]

X_val = X_train_full[val_idx]
y_val = y_train_full[val_idx]

print(f"\nTraining samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")


# ============================================================
# MODEL
# ============================================================

def build_model(input_dim):

    model = models.Sequential([

        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.30),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.20),

        layers.Dense(16, activation='relu'),

        layers.Dense(1)
    ])

    model.compile(
        optimizer=Adam(3e-4),
        loss='mae',
        metrics=['mae','mape']
    )

    return model


# ============================================================
# CALLBACKS
# ============================================================

callbacks_list = [

    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),

    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...\n")

tf.keras.backend.clear_session()

model = build_model(X_train.shape[1])

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks_list,
    verbose=1
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating on test set...\n")

y_pred = model.predict(X_test, verbose=0).flatten()

r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("============================================================")
print("TEST PERFORMANCE")
print("============================================================")

print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.2f}")
print(f"MAPE : {mape:.2f}%")


# ============================================================
# CLARKE ERROR GRID
# ============================================================

print("\nGenerating Clarke Error Grid Analysis...")

cega(y_test, y_pred)