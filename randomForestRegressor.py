# Random Forest Regressor (R^2: ~0.95)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from cega import cega

# Load data into dataframe
bg_df = pd.read_csv('processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv')

# Drop unwanted features
bg_df = bg_df.drop(columns=[col for col in bg_df.columns if 'ecg' in col.lower()])
bg_df = bg_df.drop('mean_bp', axis=1)   # Drop BP (no BP sensor on device)
bg_df = bg_df.drop('sys_bp', axis=1)
bg_df = bg_df.drop('dys_bp', axis=1)
bg_df = bg_df.drop('ppg_freq', axis=1)
bg_df = bg_df.drop('first_deriv_min', axis=1)
bg_df = bg_df.drop('caseid', axis=1)    # Drop ID column (not a feature)

# Split feature variables and target variable
X = bg_df[['age', 'sex', 'preop_dm', 'weight', 'height', 'ppg_mean', 'ppg_std', 'mean_pp_interval_s', 'std_pp_interval_s',
            'auc', 'first_deriv_max', 'entropy']]
y = bg_df['preop_gluc']

# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# print("Training and testing data prepared.")
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

rf_model = RandomForestRegressor(
    n_estimators=300,          # Good default: more trees = more stable predictions
    random_state=42,           # Always set for reproducibility
    max_depth=None,            # Let trees grow fully (good default for RF)
    min_samples_split=2,       # Default is fine
    min_samples_leaf=1,        # Default is fine
    max_features='sqrt',       # Key parameter: sqrt(12) ≈ 3-4 features per split
    bootstrap=True,            # Default: use bootstrapping
    n_jobs=-1,                 # Use all CPU cores for faster training
    warm_start=False           # Not needed unless incrementally training
)

# rf_model = RandomForestRegressor(
#     n_estimators=100,      
#     max_depth=15,          
#     min_samples_leaf=5,    
#     max_features='sqrt',
#     random_state=42,
#     n_jobs=-1
# )

# rf_model = RandomForestRegressor(
#     n_estimators=200,        
#     max_depth=15,             
#     min_samples_leaf=5,        
#     min_samples_split=10,      
#     max_features='sqrt',
#     random_state=42,
#     n_jobs=-1
# )

# Train model
rf_model.fit(X_train, y_train)

# Run model predictions
y_pred_test_rf = rf_model.predict(X_test)

# Evaluate model
pred_test_rf_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test_rf})
print(pred_test_rf_df)

r2_rf_model_test = round(rf_model.score(X_test, y_test),2)
print("R^2 Test: {}".format(r2_rf_model_test))

# Perform CEGA classification and plotting
cega(y_test, y_pred_test_rf)

import joblib

# Save the trained model (note: currently really large as of 12/26/2025)
joblib.dump(rf_model, './model_weights/rf_glucose_model.pkl', compress=3)
print("Model saved to model_weights folder as rf_glucose_model.pkl")
