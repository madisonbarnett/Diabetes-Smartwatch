# Neural network, blood glucose regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from cega import cega

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

# Define parameters for easy reuse or substitution
DATASET = 'vitaldb' # 'vitaldb' or 'physionet'
DATAFILE = 'processed_data/vitaldb_ppg_ecg_extracted_features_30s.csv' if DATASET == 'vitaldb' else 'processed_data/physioNet_ppg_extracted_features_30s.csv'
GLUC = 'preop_gluc' if DATASET == 'vitaldb' else 'glucose_mg_dl'  # Target variable
ID = 'caseid' if DATASET == 'vitaldb' else 'patient_id'      # Grouping variable to prevent data leakage
SUFFIX        = '30s'

# Load dataset into dataframe
bg_df = pd.read_csv(DATAFILE)
print(f"Successfully loaded data from {DATASET} (shape: {bg_df.shape})")
groups = bg_df[ID].values

# Split into train + test  
gss = GroupShuffleSplit(n_splits=1, test_size=0.1875, random_state=42)
train_idx, test_idx = next(gss.split(bg_df, groups=groups))

df_train  = bg_df.iloc[train_idx].copy()
df_test = bg_df.iloc[test_idx].copy()

print(f"Dev patients:   {df_train[ID].nunique()}")
print(f"Test patients:  {df_test[ID].nunique()}")
print(f"Dev rows:       {len(df_train):,}")
print(f"Test rows:      {len(df_test):,}")

# Drop unwanted columns
drop_cols_vdb = [col for col in bg_df.columns if 'ecg' in col.lower()]
drop_cols_vdb.extend(['mean_bp', 'sys_bp', 'dys_bp', 'ppg_freq', 
                  'first_deriv_min', 'caseid'])

drop_cols_physio = ['ppg_freq', 'patient_id'] # ppg_freq redundant, patient_id not a feature

drop_cols = drop_cols_vdb if DATASET == 'vitaldb' else drop_cols_physio
bg_df = bg_df.drop(columns=drop_cols)

# # Features & target
features_vdb = [
    'age', 'sex', 'preop_dm', 'weight', 'height',
    'ppg_mean', 'ppg_std', 'mean_pp_interval_s', 'std_pp_interval_s',
    'auc', 'first_deriv_max', 'entropy'
]

features_physio = ['sex', 'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s', 'ppg_std_pp_interval_s', 
                'ppg_auc', 'ppg_first_deriv_max', 'ppg_first_deriv_min', 'ppg_entropy', 
                'ppg_teager_energy', 'ppg_log_energy', 'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy']

if DATASET == 'vitaldb':
    features = features_vdb
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

# Small MLP
def build_small_glucose_model(input_dim):
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


# Create model
input_dim = X_train_scaled.shape[1]
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
        patience=12,
        min_lr=1e-6,
        verbose=1
    )
    # Optional: ModelCheckpoint if you want to save best model during training
    # callbacks.ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)
]

# Train model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.15,          # small validation set from training
    epochs=100,
    batch_size=64,                  # relatively small batch → better generalization on small data
    verbose=1,
    callbacks=callbacks_list
)

# Evaluate model on test set
y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()

r2_test = r2_score(y_test, y_pred_test)
mae_test = np.mean(np.abs(y_test - y_pred_test))
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

print("\n" + "="*60)
print(f"Neural Network Results on Test set:")
print(f"    R²  : {r2_test:.3f}")
print(f"    MAE : {mae_test:.2f} mg/dL")
print(f"    MAPE: {mape_test:.2f}%")
print("="*60)

# Optional: quick comparison table
import pandas as pd
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test.round(1)
})
print(results_df.sample(12))

# Perform CEGA analysis and plot results
print("\nGenerating Clarke Error Grid Analysis plot...")
cega(y_test, y_pred_test)

# Save trained model
model.save(f"model_weights/mlp_{SUFFIX}_{DATASET}.keras")
print(f"Keras model saved as mlp_{SUFFIX}_{DATASET}.keras")

# Save scalers
import joblib
joblib.dump(scaler, f"model_weights/mlp_feature_scaler_{SUFFIX}_{DATASET}.pkl")
print(f"Feature scaler saved as feature_scaler_{SUFFIX}_{DATASET}.pkl")