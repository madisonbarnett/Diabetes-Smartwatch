# Small Neural Network for Blood Glucose Regression
# Target: small enough for microcontroller deployment after quantization
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

print("TensorFlow version:", tf.__version__)

# ────────────────────────────────────────────────────────────────
# 1. Load and prepare data (same as your original)
# ────────────────────────────────────────────────────────────────
bg_df = pd.read_csv('processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv')

# Drop unwanted columns (same logic as yours)
drop_cols = [col for col in bg_df.columns if 'ecg' in col.lower()]
drop_cols.extend(['mean_bp', 'sys_bp', 'dys_bp', 'ppg_freq', 
                  'first_deriv_min', 'caseid'])

bg_df = bg_df.drop(columns=drop_cols)

# Features & target
feature_cols = [
    'age', 'sex', 'preop_dm', 'weight', 'height',
    'ppg_mean', 'ppg_std', 'mean_pp_interval_s', 'std_pp_interval_s',
    'auc', 'first_deriv_max', 'entropy'
]

X = bg_df[feature_cols].values.astype(np.float32)
y = bg_df['preop_gluc'].values.astype(np.float32)

# Very important for neural networks: Feature scaling!
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ────────────────────────────────────────────────────────────────
# 2. Define a small, microcontroller-friendly MLP
# ────────────────────────────────────────────────────────────────
def build_small_glucose_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(48, activation='relu'),   # wider first layer helps capture more
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
        loss='mse',
        metrics=['mae']
    )
    
    return model


# Create model
input_dim = X_train.shape[1]
model = build_small_glucose_model(input_dim)
model.summary()

# ────────────────────────────────────────────────────────────────
# 3. Callbacks - very helpful for small/medium datasets
# ────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────
# 4. Train
# ────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_split=0.15,          # small validation set from training
    epochs=100,
    batch_size=64,                  # relatively small batch → better generalization on small data
    verbose=1,
    callbacks=callbacks_list
)

# ────────────────────────────────────────────────────────────────
# 5. Evaluate
# ────────────────────────────────────────────────────────────────
y_pred_test = model.predict(X_test, verbose=0).flatten()

r2_test = r2_score(y_test, y_pred_test)
mae_test = np.mean(np.abs(y_test - y_pred_test))

print("\n" + "="*60)
print(f"Neural Network Results on Test set:")
print(f"    R²  : {r2_test:.3f}")
print(f"    MAE : {mae_test:.2f} mg/dL")
print("="*60)

# Optional: quick comparison table
import pandas as pd
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test.round(1)
})
print(results_df.sample(12))

# ────────────────────────────────────────────────────────────────
# 6. Save model (for later conversion to TFLite / C code)
# ────────────────────────────────────────────────────────────────
model.save("model_weights/glucose_mlp.keras")
print("Keras model saved → glucose_mlp.keras")

# Also save the scaler (critical!)
import joblib
joblib.dump(scaler, "model_weights/feature_scaler.pkl")
print("Feature scaler saved → feature_scaler.pkl")