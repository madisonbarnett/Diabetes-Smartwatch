# Random Forest Regressor (R^2: ~0.95)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from cega import cega
import matplotlib.pyplot as plt

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
y_pred_test = rf_model.predict(X_test)

# Evaluate model using R^2, MAE, MAPE
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
print(pred_df)  # Print handful of actual v predicted values in dataframe

r2_score = round(rf_model.score(X_test, y_test),2)
print("R^2 Test: {}".format(r2_score))

mae = round(mean_absolute_error(y_test, y_pred_test),2)
print("MAE Test: {}".format(mae), "mg/dL")

mape = round(mean_absolute_percentage_error(y_test, y_pred_test)*100,2)
print("MAPE Test: {}%".format(mape), "\n")

# Perform CEGA classification and plotting
cega(y_test, y_pred_test)

# === Improve model by selecting important features only ===

# Plot feature importances
plt.close('all')
plt.figure(figsize=(10,6))
feat_importances = pd.Series(rf_model.feature_importances_, index = X_train.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.title('Random Forest Regressor Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
plt.close('all')

train_x_if = X_train[['age', 'weight', 'height', 'preop_dm', 'mean_pp_interval_s']]  # Top 5 important features
test_x_if = X_test[['age', 'weight', 'height', 'preop_dm', 'mean_pp_interval_s']]

# Retrain model with important features only
rf_model_if = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    warm_start=False
)

# Train model
rf_model_if.fit(train_x_if, y_train)

# Run model predictions
y_pred_test_if = rf_model_if.predict(test_x_if)

# Evaluate model
updated_pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test_if})
print(updated_pred_df)

r2_score_updated = round(rf_model_if.score(test_x_if, y_test),2)
print("R^2 Test: {}".format(r2_score_updated))

mae_updated = round(mean_absolute_error(y_test, y_pred_test_if),2)
print("MAE Test: {}".format(mae_updated), "mg/dL")

mape_updated = round(mean_absolute_percentage_error(y_test, y_pred_test_if)*100,2)
print("MAPE Test: {}%".format(mape_updated), "\n")

# Perform CEGA classification and plotting
plt.close('all')
cega(y_test, y_pred_test_if)

import joblib 

# Save model (can be compressed further with compress=9)
joblib.dump(rf_model_if, './model_weights/rf_glucose_model.pkl', compress=3)
print("Model saved to model_weights folder as rf_glucose_model.pkl")