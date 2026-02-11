# Random Forest Regressor 
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from cega import cega
import matplotlib.pyplot as plt

# Load data into dataframe
bg_df = pd.read_csv(
    'processed_data/new_vitaldb_ppg_extracted_features_30s_5minwin.csv'
)

# -----------------------------
# KEEP caseid for grouping
# -----------------------------
groups = bg_df['caseid']

# Drop unwanted features (NOT caseid yet)
bg_df = bg_df.drop('ppg_freq', axis=1)
bg_df = bg_df.drop('ppg_first_deriv_min', axis=1)

# Split feature variables and target variable
X = bg_df[['age', 'sex', 'preop_dm', 'weight', 'height', 'bmi',
            'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s',
            'ppg_std_pp_interval_s', 'ppg_auc',
            'ppg_first_deriv_max', 'ppg_entropy',
            'ppg_teager_energy', 'ppg_log_energy',
            'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy']]

y = bg_df['gluc']

# ----------------------------------------------------
# GroupShuffleSplit (80/20 split by PATIENT)
# ----------------------------------------------------
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Unique train patients:", len(groups.iloc[train_idx].unique()))
print("Unique test patients:", len(groups.iloc[test_idx].unique()))

# ----------------------------------------------------
# Random Forest
# ----------------------------------------------------
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train, y_train)

# Predictions
y_pred_test = rf_model.predict(X_test)

# Evaluation
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
print(pred_df)

r2_score = round(rf_model.score(X_test, y_test), 2)
print("R^2 Test:", r2_score)

mae = round(mean_absolute_error(y_test, y_pred_test), 2)
print("MAE Test:", mae, "mg/dL")

mape = round(mean_absolute_percentage_error(y_test, y_pred_test)*100, 2)
print("MAPE Test:", mape, "%\n")

# CEGA
cega(y_test, y_pred_test)

# ----------------------------------------------------
# Feature Importance Plot
# ----------------------------------------------------
plt.close('all')
plt.figure(figsize=(10,6))

feat_importances = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
)

feat_importances.nlargest(12).plot(kind='barh')
plt.title('Random Forest Regressor Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# ----------------------------------------------------
# Retrain with important features
# ----------------------------------------------------
most_important_features = feat_importances.nlargest(12).index.tolist()
print("Training again on most important features: ", most_important_features)

train_x_if = X_train[most_important_features]

test_x_if = X_test[most_important_features]

rf_model_if = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1
)

rf_model_if.fit(train_x_if, y_train)

y_pred_test_if = rf_model_if.predict(test_x_if)

updated_pred_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test_if
})
print(updated_pred_df)

print("R^2 Test:", round(rf_model_if.score(test_x_if, y_test), 2))
print("MAE Test:", round(mean_absolute_error(y_test, y_pred_test_if), 2), "mg/dL")
print("MAPE Test:", round(mean_absolute_percentage_error(y_test, y_pred_test_if)*100, 2), "%")

plt.close('all')
cega(y_test, y_pred_test_if)
