import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Config
FILTERED_FILE = './processed_data/vitaldb_ppg_extracted_features_1.csv'
EPOCHS = 80
BATCH_SIZE = 32

# Load CSV file
df = pd.read_csv(FILTERED_FILE)

# Define features
sequential_features = ['ppg_mean', 'ppg_std', 'mean_pp_interval_s', 'std_pp_interval_s',
                       'ppg_freq', 'auc', 'first_deriv_max', 'first_deriv_min', 'entropy']
static_features = ['age', 'sex', 'preop_dm', 'weight', 'height']
target = 'preop_gluc' 

# Aggregate sequential features per caseid by taking mean
def aggregate_features(df, seq_features, static_features, target):
    # Group by caseid and computer mean for sequential features
    agg_funcs = {feat: 'mean' for feat in seq_features}
    # Keep static features (constants per caseid), taking first value
    for feat in static_features + [target]:
        agg_funcs[feat] = 'first'
    aggregated = df.groupby('caseid').agg(agg_funcs).reset_index()
    return aggregated

# Aggregate data
agg_data = aggregate_features(df, sequential_features, static_features, target)

# Prepare features and target
X = agg_data[sequential_features + static_features]
y = agg_data[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training, test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define dense nn
def build_dense_model(n_features):
    model = models.Sequential([
        # layers.Dense(64, activation='relu', input_shape=(n_features,)),
        layers.Dense(32, activation='relu', input_shape=(n_features,)),
        #layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build and train model
model = build_dense_model(X_scaled.shape[1])
model.summary() 

history = model.fit(
    X_train, y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_split = 0.2,
    verbose = 1
)

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Make prediction
sample = X_test[0:1]
predicted_glucose = model.predict(sample)[0][0]
print(f"Predicted preop_glucose for sample: {predicted_glucose:.2f}, Actual: {y_test.iloc[0]:.2f}")

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual preop_gluc (mg/dL)')
plt.ylabel('Predicted preop_gluc (mg/dL)')
plt.title('Predicted vs Actual Blood Glucose')
plt.show()
