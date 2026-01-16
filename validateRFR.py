import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_validate, LeavePGroupsOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from cega import cega
import matplotlib.pyplot as plt

# Load data into dataframe
bg_df = pd.read_csv('processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv')

# Split feature variables and target variable
features = ['age', 'sex', 'preop_dm', 'weight', 'height', 'ppg_mean', 'ppg_std', 'mean_pp_interval_s', 'std_pp_interval_s',
            'auc', 'first_deriv_max', 'entropy']
X = bg_df[features].copy()
y = bg_df['preop_gluc'].copy()

# Group data by caseid to prevent data leakage
groups = bg_df['caseid'].values

rf_model = RandomForestRegressor(
    n_estimators=100,          
    random_state=42,           
    max_depth=6,            
    min_samples_split=10,       
    min_samples_leaf=5,       
    max_features=0.4,       
    bootstrap=True,            
    n_jobs=-1         
)

# gkf = GroupKFold(n_splits=10)
logo = LeavePGroupsOut(n_groups = 5) 

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'mape': 'neg_mean_absolute_percentage_error'
}

cv_results = cross_validate(
    rf_model,
    X, y,
    # cv = gkf,
    cv = logo,
    groups = groups,
    scoring = scoring,
    return_train_score = True,
    n_jobs = -1
)

print("Cross-validation results (mean +/- std over folds):")
for metric in scoring.keys():
    train_score = cv_results[f'train_{metric}']
    test_score = cv_results[f'test_{metric}']
    print(f"{metric} - Train: {train_score.mean():.4f} +/- {train_score.std():.4f}, Test: {test_score.mean():.4f} +/- {test_score.std():.4f}")