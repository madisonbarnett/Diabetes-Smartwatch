import pandas as pd

# ────────────────────────────────────────────────
# CONFIGURATION - change these to match your file
# ────────────────────────────────────────────────

INPUT_FILE  = "processed_data\\vitaldb_ppg_ecg_extracted_features_15s.csv"          
OUTPUT_FILE = "processed_data\\agg_ppg_data.csv"         

GROUP_COL   = 'caseid'                      

DYNAMIC_COLS = [
    'ppg_mean',
    'ppg_std',
    'mean_pp_interval_s',
    'std_pp_interval_s',
    'auc',
    'first_deriv_max',
    'entropy'
]

STATIC_COLS = ['preop_gluc', 'age', 'sex', 'preop_dm', 'weight', 'height']

# Which statistics to compute for each feature
STATS = ['mean', 'std']       

# ────────────────────────────────────────────────
# Main aggregation
# ────────────────────────────────────────────────

print("Reading data...")
df = pd.read_csv(INPUT_FILE)

print(f"Original shape: {df.shape}")
print(f"Unique caseids: {df[GROUP_COL].nunique()}")

# Build aggregation dictionary
agg_dict = {feat: STATS for feat in DYNAMIC_COLS}

# Perform groupby + aggregation
print("Aggregating...")
aggregated = (
    df.groupby(GROUP_COL, as_index=True)
      .agg(agg_dict)
)

# Flatten the multi-index columns (ppg_mean → ppg_mean_mean, etc.)
aggregated.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0]
    for col in aggregated.columns.values
]

# Rename columns to avoid feature name vs statistic confusion (e.g., ppg_mean_mean → ppg_mean_avg)
# mean statistic -> avg, std statistic -> variability
rename_dict = {}
for col in aggregated.columns:
    new_col = col
    if col.endswith('_mean'):
        new_col = col[:-5] + '_avg'    
    if col.endswith('_std'):
        new_col = col[:-4] + '_variability'
    rename_dict[col] = new_col

aggregated = aggregated.rename(columns=rename_dict)

# Reset index so caseid becomes a regular column again
aggregated = aggregated.reset_index()

# Keep metadata columns (take first value per group - assumes they're constant within subject)
if STATIC_COLS:
    print("Adding metadata columns...")
    meta = df.groupby(GROUP_COL)[STATIC_COLS].first().reset_index()
    aggregated = pd.merge(
        aggregated,
        meta,
        on=GROUP_COL,
        how='left',
        validate='1:1'
    )

# Optional: reorder columns so caseid is first, then metadata, then features
cols_order = [GROUP_COL] + STATIC_COLS + [c for c in aggregated.columns
                                            if c not in [GROUP_COL] + STATIC_COLS]
aggregated = aggregated[cols_order]

# Save
print(f"Saving to {OUTPUT_FILE} ...")
aggregated.to_csv(OUTPUT_FILE, index=False)

print("Done!")
print(f"Final shape: {aggregated.shape}")
print("\nFirst few rows:")
print(aggregated.head())

# Quick sanity check
print("\nMissing values per column (should be very few):")
print(aggregated.isna().sum())