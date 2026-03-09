import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data into dataframe
bg_df = pd.read_csv('processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv')

# Choose number of bins
num_bins = 100

# Compute original histogram manually using numpy
counts, bin_edges = np.histogram(bg_df['gluc'], bins=num_bins)

total = len(bg_df['gluc'])

print("\nTotal count:", total)

print("\nBin Statistics (Original):")
print("-" * 60)

for i in range(len(counts)):
    bin_start = bin_edges[i]
    bin_end = bin_edges[i+1]
    count = counts[i]
    percent = (count / total) * 100

    print(f"Bin {i+1:2d}: [{bin_start:8.2f}, {bin_end:8.2f}) "
          f"Count = {count:6d} "
          f"({percent:6.2f}%)")
    
# Compute log-transformed histogram manually using numpy
counts, bin_edges = np.histogram(np.log1p(bg_df['gluc']), bins=num_bins)

total = len(np.log1p(bg_df['gluc']))

print("\nBin Statistics (Log-Transformed):")
print("-" * 60)

for i in range(len(counts)):
    bin_start = bin_edges[i]
    bin_end = bin_edges[i+1]
    count = counts[i]
    percent = (count / total) * 100

    print(f"Bin {i+1:2d}: [{bin_start:8.2f}, {bin_end:8.2f}) "
          f"Count = {count:6d} "
          f"({percent:6.2f}%)")

glucose_skew = bg_df['gluc'].skew()
log_glucose_skew = np.log1p(bg_df['gluc']).skew()
print(f"Skewness of original glucose: {glucose_skew:.2f}")
print(f"Skewness of log-transformed glucose: {log_glucose_skew:.2f}")

# Plot histogram of glucose
plt.figure(figsize=(10,6))
sns.histplot(bg_df['gluc'], bins=num_bins, kde=True)
plt.title('Histogram of Glucose')
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.show()

# Plot histogram of log-transformed glucose
plt.figure(figsize=(10,6))
sns.histplot(np.log1p(bg_df['gluc']), bins=num_bins, kde=True)
plt.title('Histogram of Log-Transformed Glucose')
plt.xlabel('Log-Transformed Glucose')
plt.ylabel('Frequency')
plt.show()

# Early exit to only plot skewness histogram, remove to continue plotting feature analysis heatmap
exit()

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
corr = bg_df.corr(numeric_only=True)
print(corr)

# Display correlation heatmap
plt.figure(figsize=(14, 12))  # Bigger figure
sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, cmap="coolwarm", center=0)
plt.title('Feature Correlation Heatmap')
plt.xticks(rotation=30, ha='right')
plt.show()

# Drop redundant features based on correlation analysis
# bg_df = bg_df.drop('mean_bp', axis=1)   # Drop BP due to hardware limits
# bg_df = bg_df.drop('sys_bp', axis=1)
# bg_df = bg_df.drop('dys_bp', axis=1)
bg_df = bg_df.drop('ppg_freq', axis=1)
bg_df = bg_df.drop('ppg_first_deriv_min', axis=1)
bg_df = bg_df.drop('bmi', axis=1)
bg_df = bg_df.drop('ppg_spectral_entropy', axis=1)
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

bg_df.info()



