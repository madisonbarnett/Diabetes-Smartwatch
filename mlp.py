# Neural network, blood glucose regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from cega import cega
import time

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Define parameters for easy reuse or substitution
DATASET = 'vitaldb' # 'vitaldb' or 'physionet'
DATAFILE = 'processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv' if DATASET == 'vitaldb' else 'processed_data/physioNet_ppg_extracted_features_30s.csv'
GLUC = 'gluc' if DATASET == 'vitaldb' else 'glucose_mg_dl'  # Target variable
ID = 'caseid' if DATASET == 'vitaldb' else 'patient_id'      # Grouping variable to prevent data leakage
SUFFIX        = '15s'
FEATURES = 'important' # 'all' or 'important'

# Load dataset into dataframe
bg_df = pd.read_csv(DATAFILE)
print(f"Successfully loaded data from {DATASET} (shape: {bg_df.shape})")
groups = bg_df[ID].values

# Split into train+val and test  
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(bg_df, groups=groups))

df_train  = bg_df.iloc[train_idx].copy()
df_test = bg_df.iloc[test_idx].copy()

print(f"Dev patients:   {df_train[ID].nunique()}")
print(f"Test patients:  {df_test[ID].nunique()}")
print(f"Dev rows:       {len(df_train):,}")
print(f"Test rows:      {len(df_test):,}")

# Drop unwanted columns
drop_cols_vdb = [col for col in bg_df.columns if 'ecg' in col.lower()]
drop_cols_vdb.extend(['ppg_freq', 'ppg_first_deriv_min', 'caseid', 'bmi'])

drop_cols_physio = ['ppg_freq', 'patient_id'] # ppg_freq redundant, patient_id not a feature

drop_cols = drop_cols_vdb if DATASET == 'vitaldb' else drop_cols_physio
bg_df = bg_df.drop(columns=drop_cols)

# # Features & target
# All VDB Features
all_features_vdb = ['age', 'sex', 'preop_dm', 'weight', 'height',
            'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s',
            'ppg_std_pp_interval_s', 'ppg_auc',
            'ppg_first_deriv_max', 'ppg_entropy',
            'ppg_teager_energy', 'ppg_log_energy',
            'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy']

# Only top 12 VDB Features + sex
important_features_vdb = ['age', 'weight', 'height', 'preop_dm', 'ppg_mean_pp_interval_s',
                'ppg_std', 'ppg_teager_energy', 'ppg_skew',
                'ppg_iqr', 'ppg_entropy', 'ppg_first_deriv_max', 'ppg_std_pp_interval_s']

features_physio = ['sex', 'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s', 'ppg_std_pp_interval_s', 
                'ppg_auc', 'ppg_first_deriv_max', 'ppg_first_deriv_min', 'ppg_entropy', 
                'ppg_teager_energy', 'ppg_log_energy', 'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy']

if DATASET == 'vitaldb':
    if FEATURES == 'all':
        features = all_features_vdb
    else:
        features = important_features_vdb
elif DATASET == 'physionet':
    features = features_physio

X_train  = df_train[features].values.astype(np.float32)
y_train  = df_train[GLUC].values.astype(np.float32)

X_test = df_test[features].values.astype(np.float32)
y_test = df_test[GLUC].values.astype(np.float32)

# Scale only on training set
scaler = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train_scaled.shape[0]}, Features: {X_train_scaled.shape[1]}")
print(f"Testing samples:  {X_test_scaled.shape[0]}")

print("Splitting training data into actual train + validation sets...")

# Split training into actual train + validation (val used for representative data set during quantization)
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_actual_idx, val_idx = next(gss_val.split(
    X_train_scaled, 
    groups=df_train[ID].values
))

X_train_actual = X_train_scaled[train_actual_idx]
y_train_actual = y_train[train_actual_idx]

X_val = X_train_scaled[val_idx]
y_val = y_train[val_idx]

# Log transform target variable
y_train_actual_log = np.log1p(y_train_actual)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

print(f"Actual training samples: {X_train_actual.shape[0]}")
print(f"Validation   samples:    {X_val.shape[0]}")
print(f"Test         samples:    {X_test_scaled.shape[0]}")

# Custom weighted MAE loss function
def weighted_mae(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    
    # Base weight = 1
    weights = tf.ones_like(y_true)
    
    # Increase penalty in hypo/hyper ranges
    weights = tf.where(y_true < 80,  weights * 3.0, weights) 
    weights = tf.where(y_true < 70,  weights * 30.0, weights)
    weights = tf.where(y_true < 55, weights * 60.0, weights)
    weights = tf.where(y_true > 180, weights * 5.0, weights)
    weights = tf.where(y_true > 220, weights * 50.0, weights)
    weights = tf.where(y_true > 300, weights * 100.0, weights)
    
    # Optional: make continuous/smooth transition
    hypo_weight = 1 + 7 * tf.sigmoid((70 - y_true) / 15)
    hyper_weight = 1 + 5 * tf.sigmoid((y_true - 180) / 30)
    weights = hypo_weight * hyper_weight
    
    weighted_error = error * weights
    return tf.reduce_mean(weighted_error)

# Another custom loss option aiming to reduce error in hypo/hyper ranges
def asymmetric_weighted_mae(y_true, y_pred):
    diff = y_pred - y_true   # positive = over-prediction
    abs_diff = tf.abs(diff)

    weights = tf.ones_like(y_true, dtype=tf.float32)

    # Base clinical weighting
    weights = tf.where(y_true < 80,  weights * 3.0, weights) 
    weights = tf.where(y_true < 70,  weights * 30.0, weights)
    weights = tf.where(y_true < 55, weights * 60.0, weights)
    weights = tf.where(y_true > 180, weights * 5.0, weights)
    weights = tf.where(y_true > 220, weights * 50.0, weights)
    weights = tf.where(y_true > 300, weights * 100.0, weights)

    # Extra penalty when under-predicting in hyper
    under_penalty = tf.where((y_true > 180) & (diff < 0), 2.5, 1.0)  # 2.5× more cost to under-predict highs

    return tf.reduce_mean(abs_diff * weights * under_penalty)

# Small MLP
def build_small_glucose_model(input_dim):
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
        # optimizer=Adam(learning_rate=0.0015),
        optimizer=Adam(learning_rate=0.0001),
        loss='mae',
        metrics=['mae', 'mape']
    )
    
    return model

# Create model
input_dim = X_train_actual.shape[1]
model = build_small_glucose_model(input_dim)
model.summary()

# Training callbacks
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
        patience=15,
        min_lr=1e-6,
        verbose=1
    )
    # Optional: ModelCheckpoint if you want to save best model during training
    # callbacks.ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)
]



sample_weights = np.ones_like(y_train_actual, dtype=np.float32)
sample_weights = tf.where(y_train_actual < 55, sample_weights * 3.0, sample_weights)
sample_weights = tf.where(y_train_actual > 180, sample_weights * 3.0, sample_weights)
sample_weights = tf.where(y_train_actual > 240, sample_weights * 2.0, sample_weights)
sample_weights = tf.where(y_train_actual > 300, sample_weights * 2.0, sample_weights)


# Train model
print("Starting model training!")
history = model.fit(
    X_train_actual, y_train_actual_log,
    validation_data=(X_val, y_val_log),           
    epochs=60,
    batch_size=32,
    verbose=1,
    callbacks=callbacks_list,
    sample_weight=sample_weights
)

# Evaluate model on test set
y_pred_log = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = np.expm1(y_pred_log)  # Inverse of log1p to get back to original scale

r2_test = r2_score(y_test, y_pred)
mae_test = np.mean(np.abs(y_test - y_pred))
mape_test = mean_absolute_percentage_error(y_test, y_pred) * 100

print("\n" + "="*60)
print(f"MLP results on TEST set:")
print(f"    R²  : {r2_test:.3f}")
print(f"    MAE : {mae_test:.2f} mg/dL")
print(f"    MAPE: {mape_test:.2f}%")
print("="*60)

# # Optional: quick comparison table
# import pandas as pd
# results_df = pd.DataFrame({
#     'Actual': y_test,
#     'Predicted': y_pred_test.round(1)
# })
# print(results_df.sample(12))

# Perform CEGA analysis and plot results
print("\nGenerating Clarke Error Grid Analysis plot...")
cega(y_test, y_pred)

print("Exiting for now. Comment this out later to quantize and save the model!")
exit()

# # Save trained model
# model.save(f"model_weights/mlp_{SUFFIX}_{DATASET}.keras")
# print(f"Keras model saved as mlp_{SUFFIX}_{DATASET}.keras")

# # Save scalers
# import joblib
# joblib.dump(scaler, f"model_weights/mlp_feature_scaler_{SUFFIX}_{DATASET}.pkl")
# print(f"Feature scaler saved as feature_scaler_{SUFFIX}_{DATASET}.pkl")

# Convert model (not scalers yet) to C
# import emlearn
# start_conversion = time.time()
# cmodel = emlearn.convert(model, method='loadable')
# cmodel.save(file=f"model_weights/mlp.h", name="mlp")
# end_conversion = time.time()
# print(f"Saved model as .h file (conversion took {end_conversion - start_conversion:.1f} seconds)")

def representative_dataset():
    """
    Generator that yields validation samples for calibration.
    Must return list of arrays with shape [1, n_features], float32.
    200–300 samples is typically enough for good quantization quality.
    """
    NUM_CALIBRATION_SAMPLES = 300
    num_samples = min(NUM_CALIBRATION_SAMPLES, len(X_val))
    
    for i in range(num_samples):
        # Yield single sample with batch dimension [1, features]
        yield [X_val[i:i+1].astype(np.float32)]

print("Starting TFLite quantization...")
start_quantization = time.time()

# Quantization settings
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

# Save quantized model
OUTFILE = f'model_weights/mlp_{SUFFIX}_{DATASET}_int8.tflite'
with open(OUTFILE, "wb") as f:
    f.write(tflite_model)

end_quantization = time.time()
print(f"Saved model as {OUTFILE}")
print(f"Model quantization took {end_quantization - start_quantization:.1f} seconds")
print(f"To convert .tflite to C, use the command 'xxd -i {OUTFILE} > mlp.h'")

# Check model size
import os
if os.path.exists(OUTFILE):
    print(f"Quantized model size: {os.path.getsize(OUTFILE)/1024:.1f} KB")
else:
    print(f"Error: Unable to determine file size (does the quantized model exist?)")
