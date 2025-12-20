''' Ridge Regression + Feature Importance + Correlation Analysis '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

# ---- CONFIG ----
CSV = './processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv'
CASEID_COL = 'caseid'
TARGET_COL = 'preop_gluc'   # Replace with 'glucose' or other if needed
EXCLUDED_COL = [CASEID_COL, TARGET_COL, 'ecg_mean', 'ecg_std', 'ecg_mean_pp_interval_s', 'ecg_std_pp_interval_s', 'ecg_freq', 'ecg_auc', 'ecg_first_deriv_max', 'ecg_first_deriv_min', 'ecg_entropy']

# ---- Load and clean ----
df = pd.read_csv(CSV).dropna()
print(f"Loaded shape: {df.shape}")

PPG_FEATURES = [col for col in df.columns if col not in EXCLUDED_COL]

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

# ---- Ridge pipeline ----
ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
])

ridge_pipe.fit(X_train, y_train)
y_pred = ridge_pipe.predict(X_test)

# ---- Metrics ----
mae  = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2   = r2_score(y_test, y_pred)
print(f"\nRidge Regression Performance:")
print(f"  MAE:  {mae:.2f} mg/dL")
print(f"  MAPE: {mape:.2f}%")
print(f"  R²:    {r2:.3f}")

# ---- Feature importance ----
ridge = ridge_pipe.named_steps['ridge']
coef = ridge.coef_

importance_df = pd.DataFrame({
    'Feature': PPG_FEATURES,
    'Coefficient': coef,
    'AbsCoefficient': np.abs(coef)
}).sort_values('AbsCoefficient', ascending=False)

print("\nTop Feature Importance (Ridge Coefficients):")
print(importance_df[['Feature', 'Coefficient']])

# ---- Plot: Feature importance ----
plt.figure(figsize=(8,5))
sns.barplot(x='AbsCoefficient', y='Feature', data=importance_df, palette='crest')
plt.title('Feature Importance (|Ridge Coefficients|)')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ---- Plot: Correlation heatmap ----
corr = df[PPG_FEATURES + [TARGET_COL]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Pearson r'})
plt.title('Correlation Heatmap: PPG Features vs Glucose')
plt.tight_layout()
plt.show()

# ---- Plot: Actual vs Predicted ----
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Glucose (mg/dL)')
plt.ylabel('Predicted Glucose (mg/dL)')
plt.title(f'Ridge Regression: Actual vs Predicted\nMAE={mae:.1f}, MAPE={mape:.1f}%, R²={r2:.2f}')
plt.tight_layout()
plt.show()
