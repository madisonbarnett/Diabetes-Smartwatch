import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from cega import cega
import matplotlib.pyplot as plt

# Load data into dataframe
bg_df = pd.read_csv('processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv')

# Split feature variables and target variable
features = ['age', 'sex', 'preop_dm', 'weight', 'height', 'ppg_mean', 'ppg_std', 'mean_pp_interval_s', 'std_pp_interval_s',
            'auc', 'first_deriv_max', 'entropy']
X = bg_df[features]
y = bg_df['preop_gluc']

# Group data by caseid to prevent data leakage
groups = bg_df['caseid']

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

gkf = GroupKFold(n_splits=10)

scoring = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'mape': 'neg_mean_absolute_percentage_error'
}

cv_results = cross_validate(
    rf_model,
    X, y,
    cv = gkf,
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