import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data into dataframe
bg_df = pd.read_csv('processed_data/vitaldb_ppg_ecg_extracted_features_15s.csv')

# Drop ECG related columns
bg_df = bg_df.drop(columns=[col for col in bg_df.columns if 'ecg' in col.lower()])

# Check cols, non-null counts, dtypes
# bg_df.info()

# Check shape of dataframe (rows, cols)
# bg_df.shape

# Print first 5 rows of the dataframe
# print(bg_df.head())

# See metrics of dataframe
# print(bg_df.describe())

# Create histograms of each feature
# for col in bg_df.columns:
#     plt.figure(figsize=(10,6))
#     sns.histplot(bg_df[col].dropna(), bins=30, kde=True)
#     plt.title(f'Histogram of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.show()

# Find correlation between features
# corr = bg_df.corr(numeric_only=True)
# print(corr)

# Display correlation heatmap
# plt.figure(figsize=(14, 12))  # Bigger figure
# sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, cmap="coolwarm", center=0)
# plt.title('Feature Correlation Heatmap')
# plt.xticks(rotation=30, ha='right')
# plt.show()

# Drop redundant features based on correlation analysis
bg_df = bg_df.drop('mean_bp', axis=1)   # Drop BP due to hardware limits
bg_df = bg_df.drop('sys_bp', axis=1)
bg_df = bg_df.drop('dys_bp', axis=1)
bg_df = bg_df.drop('ppg_freq', axis=1)
bg_df = bg_df.drop('first_deriv_min', axis=1)
bg_df = bg_df.drop('caseid', axis=1)    # Drop ID column (not a feature)

# Update correlation matrix after dropping features
corr = bg_df.corr(numeric_only=True)
print(corr)

# Display updated correlation heatmap
plt.figure(figsize=(14, 12))  # Bigger figure
sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, cmap="coolwarm", center=0)
plt.title('Feature Correlation Heatmap')
plt.xticks(rotation=30, ha='right') 
plt.show()





