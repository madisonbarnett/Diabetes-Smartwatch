import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ---- CONFIG ----
FILTERED_FILE = './processed_data/vitaldb_ppg_ecg_extracted_features_5s_nonlin.csv'
CASEID_COL = 'caseid'
MULTITASK = True  # <<---- Toggle between True (BG + BP) or False (BG only)

# Targets
if MULTITASK:
    TARGET_COLS = ['preop_gluc', 'mean_bp', 'sys_bp', 'dys_bp']
else:
    TARGET_COLS = ['preop_gluc']

# EXCLUDED_COL = [CASEID_COL]
EXCLUDED_COL = [CASEID_COL, 'ppg_freq', 'ppg_first_deriv_min', 'ecg_mean', 'ecg_std', 'ecg_mean_pp_interval_s', 
                'ecg_std_pp_interval_s', 'ecg_freq', 'ecg_auc', 'ecg_first_deriv_max', 
                'ecg_first_deriv_min', 'ecg_entropy', 'ecg_teager_energy', 'ecg_log_energy', 
                'ecg_skew', 'ecg_iqr', 'ecg_spectral_entropy']
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 15e-4
DROPOUT = 0.5
# DNN_LAYERS = [512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32, 16, 16, 16, 8, 8, 8] # 20 Layer
# DNN_LAYERS = [256, 256, 128, 128, 64, 64, 32, 32, 16, 16]       # 10 Layer
DNN_LAYERS = [256, 128, 64, 32, 16]                               # 5 Layer
# DNN_LAYERS = [128, 64, 32]

physical_devices = tf.config.list_physical_devices('GPU')
DEVICE = '/GPU:0' if physical_devices else '/CPU:0'

# Root Mean Squared Error Helper Function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("Hyperparameters:")
print(f"Mode: {'Multi-Task' if MULTITASK else 'Single-Task'}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Dropout: {DROPOUT}")
print(f"DNN Layers: {DNN_LAYERS}")

print("Loading filtered dataset...")
df = pd.read_csv(FILTERED_FILE).dropna()
print(f"Loaded shape: {df.shape}")

# ----- Feature/target selection -----
features_to_use = [c for c in df.columns if c not in EXCLUDED_COL + TARGET_COLS]
X = df[features_to_use].values.astype(np.float32)
y = df[TARGET_COLS].values.astype(np.float32)
caseids = df[CASEID_COL].values

# To only print the first 10 features use: features: {features_to_use[:10]} ...
print(f"Using {len(features_to_use)} features: {features_to_use}")

# ----- Train/test split -----
unique_caseids = np.unique(caseids)
train_ids, test_ids = train_test_split(unique_caseids, test_size=0.2, random_state=42)
train_mask = np.isin(caseids, train_ids)
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ----- Scaling -----
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled  = x_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled  = y_scaler.transform(y_test)
y_train_orig, y_test_orig = y_train, y_test

# ----- Model -----
class MultiTaskRegressor(tf.keras.Model):
    def __init__(self, in_features, layer_sizes, dropout, out_dim):
        super(MultiTaskRegressor, self).__init__()
        # Build shared layers dynamically
        shared_layers = []
        for n, size in enumerate(layer_sizes):
            if n == 0:
                # First hidden layer includes input shape
                shared_layers.append(layers.Dense(layer_sizes[0], input_shape=(in_features,)))
            else:
                shared_layers.append(layers.Dense(size, kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
            shared_layers.append(layers.ReLU())
            shared_layers.append(layers.BatchNormalization())
            shared_layers.append(layers.Dropout(dropout))

        # Sequential container for shared layers
        self.shared = models.Sequential(shared_layers)
        
        # Output Layer
        self.output_head = layers.Dense(out_dim, activation='linear')

    def call(self, x, training=False):
        x = self.shared(x, training=training)
        return self.output_head(x)

out_dim = len(TARGET_COLS)
model = MultiTaskRegressor(X_train_scaled.shape[1], DNN_LAYERS, DROPOUT, out_dim)

# Weighted multitask loss
def multitask_loss(y_true, y_pred):
    if MULTITASK:
        weights = tf.constant([0.55, 0.15, 0.15, 0.15][:out_dim], dtype=tf.float32)
        return tf.reduce_mean(tf.reduce_sum(weights * tf.square(y_true - y_pred), axis=1))
    else:
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=multitask_loss)

# ----- Training -----
print("Training DNN model...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

# ----- Evaluation -----
y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

if MULTITASK:
    bg_pred, meanbp_pred, sysbp_pred, dysbp_pred = y_pred.T
    bg_true, meanbp_true, sysbp_true, dysbp_true = y_test_orig.T

    print("\n--- Multi-Task Regression Metrics ---")
    print(f"MAE Glucose:  {mean_absolute_error(bg_true, bg_pred):.2f} mg/dL")
    print(f"RMSE Glucose: {rmse(bg_true, bg_pred):.2f} mg/dL")
    print(f"MAE Mean BP:  {mean_absolute_error(meanbp_true, meanbp_pred):.2f} mmHg")
    print(f"RMSE Mean BP: {rmse(meanbp_true, meanbp_pred):.2f} mmHg")
    print(f"MAE Sys BP:   {mean_absolute_error(sysbp_true, sysbp_pred):.2f} mmHg")
    print(f"RMSE Sys BP:  {rmse(sysbp_true, sysbp_pred):.2f} mmHg")
    print(f"MAE Dys BP:   {mean_absolute_error(dysbp_true, dysbp_pred):.2f} mmHg")
    print(f"RMSE Dys BP:  {rmse(dysbp_true, dysbp_pred):.2f} mmHg")
else:
    bg_pred = y_pred.flatten()
    bg_true = y_test_orig.flatten()
    print("\n--- Single-Task Regression Metrics ---")
    print(f"MAE Glucose: {mean_absolute_error(bg_true, bg_pred):.2f} mg/dL")
    print(f"RMSE Glucose: {rmse(bg_true, bg_pred):.2f} mg/dL")

# ----- Clarke Error Grid (BG only) -----
def get_clarke_zone(ref, pred):
    r, p = float(ref), float(pred)
    if (r <= 70 and p <= 70) or (0.8*r <= p <= 1.2*r): return 'A'
    if (130 < r <= 180 and 1.4*(r-130) >= p) or (70 < r <= 280 and p >= (r+110)): return 'C'
    if (r <= 70 and 70 < p <= 180) or (r >= 240 and 70 <= p <= 180): return 'D'
    if (r <= 70 and p > 180) or (r > 180 and p <= 70): return 'E'
    return 'B'

zones_count = {z: 0 for z in 'ABCDE'}
total_points = len(bg_true)
points = []  # Store (ref, pred, zone) for plotting

for ref, pred in zip(bg_true, bg_pred):
    zone = get_clarke_zone(ref, pred)
    zones_count[zone] += 1
    points.append((ref, pred, zone))

print("\n--- Clarke Error Grid Analysis (BG only) ---")
for zone, count in zones_count.items():
    print(f"Zone {zone}: {count/total_points*100:.2f}% ({count}/{total_points})")
print("\n")

# --- Plotting Clarke Error Grid ---
plt.figure(figsize=(10, 10))

# Define zone colors
colors = {'A': 'green', 'B': 'yellow', 'C': 'orange', 'D': 'red', 'E': 'purple'}
labels = {
    'A': 'A: Clinically Accurate',
    'B': 'B: Benign Errors',
    'C': 'C: Overcorrection',
    'D': 'D: Dangerous Failure to Detect',
    'E': 'E: Erroneous Treatment'
}

# Scatter points by zone
for zone in 'ABCDE':
    zone_points = [(r, p) for r, p, z in points if z == zone]
    if zone_points:
        refs, preds = zip(*zone_points)
        plt.scatter(refs, preds, c=colors[zone], label=f"{labels[zone]} ({zones_count[zone]})", alpha=0.7, edgecolors='k', s=60)

# Draw grid boundaries
max_val = 400
x = np.linspace(0, max_val, 500)

# Perfect line (y = x)
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect Agreement')

# Zone A boundaries: ±20% or within 70
plt.fill_between(x, 0.8*x, 1.2*x, where=(x <= 70) | (x >= 70), color='green', alpha=0.1, label='_nolegend_')
plt.fill_between(x, 0, 70, where=x <= 70, color='green', alpha=0.1)

# Zone B: outside A but safe
# Complex boundaries, draw other zones instead

# Zone C: 
plt.fill([70, 70, 290], [180, 400, 400], color='orange', alpha=0.4)
plt.fill([130, 180, 180], [0, 0, 70], color='orange', alpha=0.4)

# Zone D: 
# x: left, right, right, left
# y: bottom, bottom, top, top

plt.fill([0, 70, 70, 0], [70, 70, 180, 180], color='red', alpha=0.1)
plt.fill([240, 400, 400, 240], [70, 70, 180, 180], color='red', alpha=0.1)

# Zone E: 
# plt.axhspan(180, max_val, xmin=0, xmax=70/400, color='purple', alpha=0.1)
# plt.axhspan(0, 70, xmin=180/400, xmax=1, color='purple', alpha=0.1)

plt.fill([0, 70, 70, 0], [180, 180, 400, 400], color='purple', alpha=0.1)
plt.fill([180, 400, 400, 180], [0, 0, 70, 70], color='purple', alpha=0.1)

# Axis limits and labels
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel('Reference Glucose (mg/dL)', fontsize=12)
plt.ylabel('Predicted Glucose (mg/dL)', fontsize=12)
plt.title('Clarke Error Grid Analysis', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

# Legend
plt.legend(loc='upper left')

# Show plot
plt.tight_layout()
plt.show()

# ----- Plot total training/validation loss -----
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss (total)')
plt.plot(history.history['val_loss'], label='Val Loss (total)')
plt.xlabel('Epoch'); plt.ylabel('Weighted MSE Loss')
plt.title('Training and Validation Loss (Shared Multitask)')
plt.legend(); plt.tight_layout(); plt.show()

# ----- Plot Per-Task Scatter Trendlines -----
if MULTITASK:
    task_names = ['Glucose (mg/dL)', 'Mean BP (mmHg)', 'Systolic BP (mmHg)', 'Diastolic BP (mmHg)']
    preds_list = [bg_pred, meanbp_pred, sysbp_pred, dysbp_pred]
    trues_list = [bg_true, meanbp_true, sysbp_true, dysbp_true]

    plt.figure(figsize=(10, 6))
    for i, (true_vals, pred_vals, name) in enumerate(zip(trues_list, preds_list, task_names)):
        plt.subplot(2, 2, i + 1)
        plt.scatter(true_vals, pred_vals, alpha=0.5, label=f"{name} Predictions")
        slope, intercept = np.polyfit(true_vals, pred_vals, 1)
        plt.plot([true_vals.min(), true_vals.max()],
                 [true_vals.min(), true_vals.max()], 'k--', label='Ideal (y=x)')
        plt.plot([true_vals.min(), true_vals.max()],
                 [slope*true_vals.min()+intercept, slope*true_vals.max()+intercept],
                 color='red', lw=2, label=f'Trendline (slope={slope:.2f})')
        plt.xlabel(f"Actual {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(name)
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
        # plt.tight_layout()
        print(f"R-Squared, {name}: {slope:.2f}")
    plt.suptitle("Multi-Task DNN: Actual vs Predicted", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    plt.figure(figsize=(7, 7))
    plt.scatter(bg_true, bg_pred, alpha=0.5, label='Test Predictions')
    slope, intercept = np.polyfit(bg_true, y_pred, 1)
    slope = float(np.squeeze(slope))
    intercept = float(np.squeeze(intercept))
    plt.plot([bg_true.min(), bg_true.max()], [bg_true.min(), bg_true.max()], 'k--', label='Ideal (y=x)')
    plt.plot([bg_true.min(), bg_true.max()],
            [slope * bg_true.min() + intercept, slope * bg_true.max() + intercept],
            color='red', lw=2, label=f'Trendline (slope={slope:.2f})')
    plt.xlabel("Actual BG (mg/dL)")
    plt.ylabel("Predicted BG (mg/dL)")
    plt.title("Actual vs. Predicted BG (TensorFlow DNN)")
    plt.legend(); plt.tight_layout(); plt.show()
    print(f"R-Squared, Glucose: {slope:.2f}")

# ----- Save model weights -----
suffix = '_multitask' if MULTITASK else '_single'
save_path = f'./model_weights/dnn_model_15s{suffix}.weights.h5'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save_weights(save_path)
print(f"\nModel weights saved to {save_path}")
