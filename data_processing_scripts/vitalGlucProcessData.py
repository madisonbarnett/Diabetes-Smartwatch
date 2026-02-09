# This script filters VitalDB glucose lab measurements and matches them to a +/- 2.5 min window of PPG data for time-series glucose values with PPG feature extraction
import pandas as pd
import vitaldb
import os
import numpy as np
from scipy.signal import find_peaks, butter, lfilter, filtfilt
from scipy.stats import entropy, skew

# Download VitalDB data
VITALDB_LABS_URL = "https://api.vitaldb.net/labs"
df_labs = pd.read_csv(VITALDB_LABS_URL)

VITALDB_DATA_URL = "https://api.vitaldb.net/cases"
df_cases = pd.read_csv(VITALDB_DATA_URL)

# Output files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "new_vitaldb_ppg_extracted_features_30s_5minwin.csv")

# Processing parameters
PPG_SIGNAL_NAME = 'SNUADC/PLETH'
SAMPLE_RATE_HZ = 500
WINDOW_DURATION_SECONDS = 30
SAMPLES_PER_WINDOW = SAMPLE_RATE_HZ * WINDOW_DURATION_SECONDS
VALID_WINDOW_MINUTES = 2.5  # +/- window around each BG value for stability & matching to physioNet structure
BATCH_SIZE = 50 # Number of cases to write to csv at a time (improves speed by reducing file I/O)

# --- UTILITY FUNCTIONS ---
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def teager_kaiser_energy(x):
    """Computes average Teager-Kaiser Energy for a 1D signal."""
    if len(x) < 3:
        return 0
    x = np.asarray(x)
    energy = x[1:-1]**2 - x[:-2] * x[2:]
    return np.mean(np.abs(energy))

def log_energy(x):
    """Computes log-energy profile."""
    energy = np.sum(x**2)
    return np.log(energy + 1e-8)  # avoid log(0)

def spectral_entropy(x, fs):
    """Computes normalized spectral entropy."""
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    psd = np.abs(np.fft.rfft(x))**2
    psd_norm = psd / np.sum(psd)
    return entropy(psd_norm)

def extract_target_bp(bp_series):
    """
    Averages blood pressure across entire window to output a single target BP value
    """
    # Clean the data: remove NaN values
    bp_series_clean = bp_series[np.isfinite(bp_series)]

    if bp_series_clean.size == 0:
        return 0
    
    return np.mean(bp_series_clean)

def extract_ppg_features(ppg_series, fs):
    """
    Extracts a set of robust, hand-engineered features from a single PPG signal window.
    """
    features = {}

    # Clean the data: remove NaN values
    ppg_series_clean = ppg_series[np.isfinite(ppg_series)]

    if ppg_series_clean.size == 0:
        return {
            'ppg_mean': 0, 'ppg_std': 0, 'ppg_mean_pp_interval_s': 0,
            'ppg_std_pp_interval_s': 0, 'ppg_freq': 0, 'ppg_auc': 0,
            'ppg_first_deriv_max': 0, 'ppg_first_deriv_min': 0, 'ppg_entropy': 0,
            'ppg_teager_energy': 0, 'ppg_log_energy': 0, 'ppg_skew': 0, 
            'ppg_iqr': 0, 'ppg_spectral_entropy': 0
        }

    features['ppg_mean'] = np.mean(ppg_series_clean)
    features['ppg_std'] = np.std(ppg_series_clean)

    peaks, _ = find_peaks(ppg_series_clean, distance=int(fs*0.4), height=0)
    if len(peaks) > 1:
        pp_intervals = np.diff(peaks)
        features['ppg_mean_pp_interval_s'] = np.mean(pp_intervals) / fs
        features['ppg_std_pp_interval_s'] = np.std(pp_intervals) / fs
        features['ppg_freq'] = fs / np.mean(pp_intervals)
    else:
        features['ppg_mean_pp_interval_s'] = 0
        features['ppg_std_pp_interval_s'] = 0
        features['ppg_freq'] = 0

    # Normalize amplitude before calculating cumulative metrics
    ppg_series_clean = (ppg_series_clean - np.mean(ppg_series_clean)) / np.std(ppg_series_clean)

    features['ppg_auc'] = np.trapezoid(ppg_series_clean)

    derivative = np.diff(ppg_series_clean)
    features['ppg_first_deriv_max'] = np.max(derivative)
    features['ppg_first_deriv_min'] = np.min(derivative)

    hist, _ = np.histogram(ppg_series_clean, bins='auto')
    features['ppg_entropy'] = entropy(hist)

    # --- Nonlinear and higher-order features ---
    features['ppg_teager_energy'] = teager_kaiser_energy(ppg_series_clean)
    features['ppg_log_energy'] = log_energy(ppg_series_clean)
    features['ppg_skew'] = skew(ppg_series_clean)
    features['ppg_iqr'] = np.percentile(ppg_series_clean, 75) - np.percentile(ppg_series_clean, 25)
    features['ppg_spectral_entropy'] = spectral_entropy(ppg_series_clean, fs)

    return features


# MAIN DATA PROCESSING SECTION

# Filter glucose labs
labs_gluc_cases = df_labs[df_labs['name'] == 'gluc'].copy()

# Filter for only cases with demographic data
bg_data_cases = df_cases[df_cases['age'].notna()].copy()
valid_caseids = set(bg_data_cases['caseid'])

# Save all raw glucose labs before filtering
# raw_gluc_samples = df_labs[df_labs['name'] == 'gluc'].copy()
# raw_gluc_samples.to_csv(os.path.join(PROJECT_ROOT,"0_all_labs_glucose.csv"), index=False)
# print(f"Saved all raw glucose lab samples to 0_all_labs_glucose.csv (total {len(raw_gluc_samples)} rows)")

# # Save all tracks data
# df_cases.to_csv(os.path.join(PROJECT_ROOT,"0_all_case_data.csv"), index=False)
# print(f"Saved all case data to 0_all_case_data.csv (total {len(df_cases)} rows)")

print("Starting data processing...")

# Filter for cases with demographic data
bg_data_cases = df_cases[df_cases['age'].notna()].copy()
caseids_to_process = list(bg_data_cases['caseid'].unique())

print(f"Found {len(caseids_to_process)} cases with relevant data to process.")

all_case_dfs = []

# Check if the output file already exists to decide whether to write the header
output_file_exists = os.path.exists(OUTPUT_FILE)

for caseid in caseids_to_process:
    print(f"Processing Case ID: {caseid}...")

    gluc_at_caseid = labs_gluc_cases[labs_gluc_cases['caseid'] == caseid]

    if gluc_at_caseid.empty:
        print(f"  No glucose values for Case ID: {caseid}. Skipping.")
        continue

    # Get case timing
    case_meta = bg_data_cases[bg_data_cases['caseid'] == caseid].iloc[0]
    caseend_seconds = case_meta['caseend']

    # Filter glucose readings to within casestart -> caseend
    valid_samples = gluc_at_caseid[
        (gluc_at_caseid['dt'] >= 0) &
        (gluc_at_caseid['dt'] <= caseend_seconds)
    ]

    if valid_samples.empty:
        print(f"  No valid glucose values for Case ID: {caseid}. Skipping.")
        continue

    # Load raw PPG signal for the case
    ppg_vals = vitaldb.load_case(caseid, [PPG_SIGNAL_NAME], 1/SAMPLE_RATE_HZ)

    if ppg_vals is None or ppg_vals.size == 0:
        print(f"  No PPG data found for Case ID {caseid}. Skipping.")
        continue

    ppg_signal = ppg_vals[:, 0]

    for index, row in valid_samples.iterrows():
        winstart_seconds = row['dt']
        gluc_val = row['result']

        # Start index is the greater of the two: 0 or index corresponding to VALID_WINDOW_MINUTES before glucose window start time
        start_idx = max(0, int((winstart_seconds - VALID_WINDOW_MINUTES * 60) * SAMPLE_RATE_HZ))
        # End index is the lesser of the two: the last index in the waveform or index corresponding to VALID_WINDOW_MINUTES after glucose window start time
        end_idx = min(len(ppg_signal), int((winstart_seconds + VALID_WINDOW_MINUTES * 60) * SAMPLE_RATE_HZ))

        windowed_ppg_signal = ppg_signal[start_idx:end_idx]

        # Skip readings that do not contain at least one full window of PPG data
        if len(windowed_ppg_signal) < SAMPLES_PER_WINDOW:
            print(f"  PPG signal for Case ID {caseid} is too short. Skipping.")
            continue

        # Apply bandpass filter to the entire windowed signal to remove noise
        filtered_ppg_signal = butter_bandpass_filter(windowed_ppg_signal, lowcut=0.5, highcut=8, fs=SAMPLE_RATE_HZ)

        sampleLen = len(filtered_ppg_signal)

        # Process and save windows for the current case
        case_data = []
        for i in range(0, sampleLen - SAMPLES_PER_WINDOW + 1, SAMPLES_PER_WINDOW):
            ppg_window = filtered_ppg_signal[i:i + SAMPLES_PER_WINDOW]

            # Extract hand-engineered features
            ppg_features = extract_ppg_features(ppg_window, SAMPLE_RATE_HZ)

            # Skip samples where PPG is null or infinite (as returned by extract_ppg_features function)
            if all(v == 0 for v in ppg_features.values()):
                continue

            row = {
                'caseid': caseid,
                'gluc': gluc_val,
                'age': case_meta['age'],
                'sex': 1 if case_meta['sex'] == 'F' else 0,
                'preop_dm': case_meta['preop_dm'],
                'weight': case_meta['weight'],
                'height': case_meta['height'],
                'bmi': case_meta['bmi'],
                **ppg_features, # Unpack the extracted PPG features
            }
            case_data.append(row)

        # Convert list of dicts to a DataFrame and append to the CSV
        case_df = pd.DataFrame(case_data)

        if not case_df.empty:
            all_case_dfs.append(case_df)
        else:
            print(f" No valid samples found for Case ID {caseid}. Skipping write.")

        # Write batch of cases to output once all_case_dfs reaches assigned BATCH_SIZE
        if len(all_case_dfs) >= BATCH_SIZE:
            batch_df = pd.concat(all_case_dfs, ignore_index=True)
            # Write the header only once at the beginning if the file is new
            header = not output_file_exists
            batch_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
            # Set the flag to True after the first write
            output_file_exists = True
            # Clear all_case_dfs for next batch use
            all_case_dfs.clear()
            print(f"Wrote batch of {BATCH_SIZE} cases to {OUTPUT_FILE}")

if all_case_dfs:
    batch_df = pd.concat(all_case_dfs, ignore_index=True)
    header = not output_file_exists
    batch_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
    print(f" Final write: saved remaining {len(all_case_dfs)} cases to {OUTPUT_FILE}")

print(f"\nProcessing complete. All data saved to {OUTPUT_FILE}.")
