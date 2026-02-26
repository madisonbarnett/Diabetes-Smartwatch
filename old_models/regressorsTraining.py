# Program to test training using different regressors
import pandas as pd
from sklearn.model_selection import train_test_split

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
print("Training and testing data prepared.")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Model 1: Linear Regressor (R^2: ~0.12)
# from sklearn.linear_model import LinearRegression
# lr_model = LinearRegression()

# # Train model
# lr_model.fit(X_train, y_train)

# # Run model predictions
# y_pred_test = lr_model.predict(X_test)

# # Evaluate model
# pred_test_lr_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
# print(pred_test_lr_df)

# r2_lr_model_test = round(lr_model.score(X_test, y_test),2)
# print("R^2 Test: {}".format(r2_lr_model_test))

# Model 2: Random Forest Regressor (R^2: ~0.95)
from sklearn.ensemble import RandomForestRegressor
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

# Train model
rf_model.fit(X_train, y_train)

# Run model predictions
y_pred_test_rf = rf_model.predict(X_test)

# Evaluate model
pred_test_rf_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test_rf})
print(pred_test_rf_df)

r2_rf_model_test = round(rf_model.score(X_test, y_test),2)
print("R^2 Test: {}".format(r2_rf_model_test))

# Model 3: Gradient Boosting Regressor (R^2: ~0.87)
# from xgboost import XGBRegressor
# xgb_model = XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.1,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     n_jobs=-1
# )

# # Train model
# xgb_model.fit(X_train, y_train)

# # Run model predictions
# y_pred_test_xgb = xgb_model.predict(X_test)

# # Evaluate model
# pred_test_xgb_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test_xgb})
# print(pred_test_xgb_df)

# r2_xgb_model_test = round(xgb_model.score(X_test, y_test),2)
# print("R^2 Test: {}".format(r2_xgb_model_test))










