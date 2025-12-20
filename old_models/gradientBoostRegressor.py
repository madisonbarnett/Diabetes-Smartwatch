''' Gradient Boosting Baseline for Glucose Prediction from PPG Features '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---- CONFIG ----
CSV = './processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv'
CASEID_COL = 'caseid'
TARGET_COL = 'preop_gluc'

# Selected PPG-based features (edit to match your dataset)
PPG_FEATURES = [
    'ppg_mean', 'ppg_std', 'mean_pp_interval_s', 'std_pp_interval_s',
    'ppg_freq', 'auc', 'first_deriv_max', 'first_deriv_min', 'entropy'
]

# ---- Load and clean ----
df = pd.read_csv(CSV).dropna(subset=[CASEID_COL, TARGET_COL] + PPG_FEATURES)
print(f"Loaded shape: {df.shape}")

# ---- Subject-wise split ----
caseids = df[CASEID_COL].values
unique_ids = np.unique(caseids)
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
train_mask = np.isin(caseids, train_ids)
test_mask  = ~train_mask

X_train = df.loc[train_mask, PPG_FEATURES].values
y_train = df.loc[train_mask, TARGET_COL].values
X_test  = df.loc[test_mask,  PPG_FEATURES].values
y_test  = df.loc[test_mask,  TARGET_COL].values

# Optional: Standardize for stability
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---- Model ----
model = GradientBoostingRegressor(
    n_estimators=300,      # number of boosting stages (trees)
    learning_rate=0.05,    # smaller = slower but more accurate
    max_depth=3,           # depth of individual trees (controls complexity)
    subsample=0.8,         # use 80% of data per tree for regularization
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ---- Evaluation ----
y_pred = model.predict(X_test_scaled)
mae  = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2   = r2_score(y_test, y_pred)

print(f"\nGradient Boosting Performance:")
print(f"  MAE:  {mae:.2f} mg/dL")
print(f"  MAPE: {mape:.2f}%")
print(f"  R²:    {r2:.3f}")

# ---- Feature importance ----
importances = model.feature_importances_
feat_imp = pd.DataFrame({'Feature': PPG_FEATURES, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False)

print("\nTop Feature Importances:")
print(feat_imp)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='crest')
plt.title('Gradient Boosting Feature Importance')
plt.tight_layout()
plt.show()

# ---- Scatter plot ----
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Glucose (mg/dL)')
plt.ylabel('Predicted Glucose (mg/dL)')
plt.title(f'Gradient Boosting: Actual vs Predicted\nMAE={mae:.1f}, MAPE={mape:.1f}%, R²={r2:.2f}')
plt.tight_layout()
plt.show()
