import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from cega import cega
import time 

# Define parameters for easy reuse or substitution
DATASET = 'physionet' # 'vitaldb' or 'physionet'
DATAFILE = 'processed_data/physioNet_ppg_extracted_features_30s.csv'
GLUC = 'glucose_mg_dl'  # Target variable
ID = 'patient_id'      # Grouping variable to prevent data leakage

if DATASET == 'vitaldb':
    DATAFILE = 'processed_data/vitaldb_ppg_ecg_extracted_features_30s.csv'
    GLUC = 'preop_gluc'  # Target variable
    ID = 'caseid'      # Grouping variable to prevent data leakage

# Load data into dataframe
bg_df = pd.read_csv(DATAFILE)
print(f"Successfully loaded data from {DATASET} (shape: {bg_df.shape})")
groups = bg_df[ID].values

# Split into development + final test   (patient-disjoint)
gss = GroupShuffleSplit(n_splits=1, test_size=0.1875, random_state=42)
dev_idx, test_idx = next(gss.split(bg_df, groups=groups))

df_dev  = bg_df.iloc[dev_idx].copy()
df_test = bg_df.iloc[test_idx].copy()

print(f"Dev patients:   {df_dev[ID].nunique()}")
print(f"Test patients:  {df_test[ID].nunique()}")
print(f"Dev rows:       {len(df_dev):,}")
print(f"Test rows:      {len(df_test):,}")

# Split feature variables and target variable
feat_vdb = ['age', 'sex', 'preop_dm', 'weight', 'height', 
            'ppg_mean', 'ppg_std', 'mean_pp_interval_s', 
            'std_pp_interval_s', 'auc', 'first_deriv_max', 'entropy'] 
feat_pn = ['sex', 'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s', 'ppg_std_pp_interval_s', 
            'ppg_auc', 'ppg_first_deriv_max', 'ppg_first_deriv_min', 'ppg_entropy', 'ppg_teager_energy', 
            'ppg_log_energy', 'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy'] 

features = feat_vdb if DATASET == 'vitaldb' else feat_pn 

X_dev  = df_dev[features].values.astype(np.float32)
y_dev  = df_dev[GLUC].values.astype(np.float32)

X_test = df_test[features].values.astype(np.float32)
y_test = df_test[GLUC].values.astype(np.float32)

# Scale only on development set
scaler = StandardScaler()
X_dev_scaled  = scaler.fit_transform(X_dev)
X_test_scaled = scaler.transform(X_test)

# Model
rf_model = RandomForestRegressor(
    n_estimators=20,
    max_depth=5,               
    min_samples_split=5,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

# # Perform cross-validation on dev set
# print("\nStarting GroupKFold CV on dev set (10 folds)...")
# gkf = GroupKFold(n_splits=10)

# scoring = {
#     'r2': 'r2',
#     'mae': 'neg_mean_absolute_error',
#     'mape': 'neg_mean_absolute_percentage_error'
# }

# start_time = time.time()
# cv_results = cross_validate(
#     xgb_model,
#     X_dev_scaled, y_dev,
#     cv=gkf,
#     groups=df_dev[ID].values,  # Keep cases together
#     scoring=scoring,
#     return_train_score=True,
#     n_jobs=-1
# )
# elapsed = time.time() - start_time

# print(f"CV completed in {elapsed:.1f} seconds")
# print("CV results (mean ± std over 10 folds):")
# for metric in ['r2', 'mae', 'mape']:
#     train_key = f'train_{metric}'
#     test_key  = f'test_{metric}'
#     train_mean, train_std = cv_results[train_key].mean(), cv_results[train_key].std()
#     test_mean,  test_std  = cv_results[test_key].mean(),  cv_results[test_key].std()
#     sign = '+' if metric in ['mae', 'mape'] else ''  # neg metrics are negative
#     print(f"{metric.upper():<6} - Train: {train_mean:.4f} ± {train_std:.4f} | Test: {test_mean:.4f} ± {test_std:.4f}")

# Train final model on full dev set and evaluate on test set
print("\nTraining final model on full dev set...")
rf_model.fit(X_dev_scaled, y_dev)

y_pred_test = rf_model.predict(X_test_scaled)

r2_test   = r2_score(y_test, y_pred_test)
mae_test  = mean_absolute_error(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100  # in %

print("\nFinal Test Set Performance:")
print("-"*40)
print(f"R²:   {r2_test:.4f}")
print(f"MAE:  {mae_test:.2f} mg/dL")
print(f"MAPE: {mape_test:.2f}%")
print("*"*40)

# Clarke Error Grid Analysis
cega(y_test, y_pred_test)
print("*"*40)

# Baseline (predict mean of dev)
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_dev_scaled, y_dev)
y_dummy = dummy.predict(X_test_scaled)
print(f"Baseline (mean predictor) R²: {r2_score(y_test, y_dummy):.4f}")

# import emlearn 
# # Convert RFR to C code
# print("\nConverting RFR model to C code...")
# conversion_start = time.time()
# c_code = emlearn.convert(rf_model, method='inline')
# conversion_end = time.time()
# print(f"Conversion completed (took {conversion_end - conversion_start:.1f} seconds)")

# # Save C code to file
# OUTFILE = f'model_weights/rf_regressor_{DATASET}.h'
# c_code.save(file=OUTFILE, name=f'rf_regressor_{DATASET}')

# import os
# print(f"Saved model as '{OUTFILE}' (file size: {os.path.getsize(OUTFILE) / 1024:.2f} KB)")

import m2cgen as m2c
# Convert XGB model to C code
print("\nConverting XGB model to C code...")
conversion_start = time.time()
c_code = m2c.export_to_c(rf_model)
conversion_end = time.time()
print(f"Conversion completed (took {conversion_end - conversion_start:.1f} seconds)")

# Save C code to file
OUTFILE = f"model_weights/rf_regressor_{DATASET}.c"
with open(OUTFILE, 'w') as f:
    f.write(c_code)

import os
print(f"Saved model as '{OUTFILE}' (file size: {os.path.getsize(OUTFILE) / 1024:.2f} KB)")