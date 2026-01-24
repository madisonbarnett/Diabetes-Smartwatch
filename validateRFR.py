import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import time 

# Load data into dataframe
bg_df = pd.read_csv('processed_data/physioNet_ppg_extracted_features_30s.csv')
print(f"Successfully loaded data (shape: {bg_df.shape})")

# Split feature variables and target variable
# VitalDB features
# features = ['age', 'sex', 'preop_dm', 'weight', 'height', 'ppg_mean_avg', 'ppg_mean_variability',
#             'ppg_std_avg', 'ppg_std_variability', 'mean_pp_interval_s_avg', 'mean_pp_interval_s_variability',
#             'std_pp_interval_s_avg', 'std_pp_interval_s_variability', 'auc_avg', 'auc_variability',
#             'first_deriv_max_avg', 'first_deriv_max_variability', 'entropy_avg', 'entropy_variability'] 
# PhysioNet features
features = ['sex', 'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s', 'ppg_std_pp_interval_s', 'ppg_freq', 
            'ppg_auc', 'ppg_first_deriv_max', 'ppg_first_deriv_min', 'ppg_entropy', 'ppg_teager_energy', 
            'ppg_log_energy', 'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy'] 
  
X = bg_df[features].copy()
y = bg_df['glucose_mg_dl'].copy()

# Group data by caseid to prevent data leakage
# groups = bg_df['caseid'].values
groups = bg_df['patient_id'].values

rf_model = RandomForestRegressor(
    n_estimators=300,          
    random_state=42,           
    max_depth=6,            
    min_samples_split=10,       
    min_samples_leaf=5,       
    max_features=0.4,       
    bootstrap=True,            
    n_jobs=-1         
)

# kf = KFold(n_splits=10, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=10)
# logo = LeavePGroupsOut(n_groups = 5) 

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'mape': 'neg_mean_absolute_percentage_error'
}

start_time = time.time()
print("Starting cross-validation...")
cv_results = cross_validate(
    rf_model,
    X, y,
    cv = gkf,
    groups = groups,
    scoring = scoring,
    return_train_score = True,
    n_jobs = -1
)

end_time = time.time()
print("Cross-validation results (mean +/- std over folds):")
for metric in scoring.keys():
    train_score = cv_results[f'train_{metric}']
    test_score = cv_results[f'test_{metric}']
    print(f"{metric} - Train: {train_score.mean():.4f} +/- {train_score.std():.4f}, Test: {test_score.mean():.4f} +/- {test_score.std():.4f}")

elapsed_time = end_time - start_time
print(f"Total cross-validation time: {elapsed_time:.2f} seconds")