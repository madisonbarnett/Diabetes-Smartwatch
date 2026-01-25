# Neural network, blood glucose regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from cega import cega

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

# Load dataset into dataframe
SUFFIX        = '30s'
DATASET       = 'PhysioNet'
# bg_df = pd.read_csv('processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv')
bg_df = pd.read_csv('processed_data/physioNet_ppg_extracted_features_30s.csv')


# Drop unwanted columns
# drop_cols = [col for col in bg_df.columns if 'ecg' in col.lower()]
# drop_cols.extend(['mean_bp', 'sys_bp', 'dys_bp', 'ppg_freq', 
#                   'first_deriv_min', 'caseid'])

drop_cols = ['ppg_freq', 'patient_id'] # ppg_freq redundant, patient_id not a feature

bg_df = bg_df.drop(columns=drop_cols)

# # Features & target
# feature_cols = [
#     'age', 'sex', 'preop_dm', 'weight', 'height',
#     'ppg_mean', 'ppg_std', 'mean_pp_interval_s', 'std_pp_interval_s',
#     'auc', 'first_deriv_max', 'entropy'
# ]

feature_cols = ['sex', 'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s', 'ppg_std_pp_interval_s', 
                'ppg_auc', 'ppg_first_deriv_max', 'ppg_first_deriv_min', 'ppg_entropy', 
                'ppg_teager_energy', 'ppg_log_energy', 'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy']

X = bg_df[feature_cols].values.astype(np.float32)
# y = bg_df['preop_gluc'].values.astype(np.float32)
y = bg_df['glucose_mg_dl'].values.astype(np.float32)

# Perform feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
print(f"Testing samples:  {X_test.shape[0]}")

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
input_dim = X_train.shape[1]
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
    X_train, y_train,
    validation_split=0.15,          # small validation set from training
    epochs=100,
    batch_size=64,                  # relatively small batch → better generalization on small data
    verbose=1,
    callbacks=callbacks_list
)

# Evaluate model on test set
y_pred_test = model.predict(X_test, verbose=0).flatten()

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