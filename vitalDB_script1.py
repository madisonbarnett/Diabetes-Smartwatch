# script_1.py, provided by Destinie Diggs
# This script filters VitalDB glucose lab measurements to include only valid samples within each case's timeframe, saves the filtered data, and a summary of samples and cases kept or excluded.
import pandas as pd
import vitaldb
import os
from tqdm import tqdm


# Download VitalDB data
VITALDB_LABS_URL = "https://api.vitaldb.net/labs"
df_labs = pd.read_csv(VITALDB_LABS_URL)

VITALDB_DATA_URL = "https://api.vitaldb.net/cases"
df_cases = pd.read_csv(VITALDB_DATA_URL)


# Filter glucose labs
labs_gluc_cases = df_labs[df_labs['name'] == 'gluc'].copy()

# Filter for only cases with demographic data
bg_data_cases = df_cases[df_cases['age'].notna()].copy()
valid_caseids = set(bg_data_cases['caseid'])


# Output files
output_data_file = os.path.join("vitaldb_data", "1_labs_glucose_filtered.csv")
output_summary_file = os.path.join("vitaldb_data","1_labs_glucose_filtered_summary.csv")

os.makedirs("vitaldb_data", exist_ok=True)
#output_summary_file = os.makedirs(output_summary_file_path, exist_ok=True)

output_file_exists = os.path.exists(output_data_file)


#If output file exists clear contents
if (output_file_exists):
    with open(output_data_file, 'r+', newline='') as f:
        # Read the first line to get the header
        f.readline()
        # Truncate the file at the current position (after the header)
        f.truncate(f.tell())


# Save all raw glucose labs before filtering
raw_gluc_samples = df_labs[df_labs['name'] == 'gluc'].copy()
raw_gluc_samples.to_csv(os.path.join("vitaldb_data","0_all_labs_glucose.csv"), index=False)
print(f"Saved all raw glucose lab samples to 0_all_labs_glucose.csv (total {len(raw_gluc_samples)} rows)")

print("Starting data processing...")


# Counters

total_bg_samples = len(labs_gluc_cases)
bg_samples_kept = 0
bg_samples_thrown = 0

samples_before_casestart = 0
samples_after_caseend = 0

cases_kept = 0
cases_thrown = 0


# Main loop
for caseid in tqdm(range(1, 6389), desc="Processing cases"):
    #print(f"caseid: {caseid}")

    # ---- Case exclusion ----
    if caseid not in valid_caseids:
        cases_thrown += 1
        continue

    gluc_at_caseid = labs_gluc_cases[labs_gluc_cases['caseid'] == caseid]

    if gluc_at_caseid.empty:
        cases_thrown += 1
        continue

    # Get case timing
    case_meta = bg_data_cases[bg_data_cases['caseid'] == caseid].iloc[0]
    caseend_seconds = case_meta['caseend']

    # Sample timing filters
    before_start = gluc_at_caseid[gluc_at_caseid['dt'] < 0]
    after_end = gluc_at_caseid[gluc_at_caseid['dt'] > caseend_seconds]

    samples_before_casestart += len(before_start)
    samples_after_caseend += len(after_end)

    valid_samples = gluc_at_caseid[
        (gluc_at_caseid['dt'] >= 0) &
        (gluc_at_caseid['dt'] <= caseend_seconds)
    ]

    if valid_samples.empty:
        cases_thrown += 1
        continue

    # Keep case
    cases_kept += 1
    bg_samples_kept += len(valid_samples)
    bg_samples_thrown += len(before_start) + len(after_end)

    # Write kept samples
    header = not output_file_exists
    valid_samples.to_csv(output_data_file, mode="a", header=header, index=False)
    output_file_exists = True

# Build summary table
summary_rows = [
    ("BG Samples", "Total Number of BG Samples", total_bg_samples),
    ("BG Samples", "Total Number of BG Samples Thrown Out", bg_samples_thrown),
    ("BG Samples", "Total Number of BG Samples Kept", bg_samples_kept),
    ("Cases", "Total Number Cases Kept", cases_kept),
    ("Cases", "Total Number Cases Thrown Out", cases_thrown),
    ("Reason for Throwing out", "No. of Samples Thrown because taken before casestart", samples_before_casestart),
    ("Reason for Throwing out", "No. of Samples Thrown because taken after caseend", samples_after_caseend),
]

summary_df = pd.DataFrame(summary_rows, columns=["section", "metric", "value"])
summary_df.to_csv(output_summary_file, index=False)

print("CSV files saved successfully!")
print("\nSummary:")
print(summary_df)