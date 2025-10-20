import pandas as pd
import vitaldb
import numpy as np
import os
from scipy.signal import find_peaks, butter, lfilter
from scipy.stats import entropy

# --- CONFIGURATION ---
OUTPUT_DIR = "./processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vitaldb_ppg_extracted_features.csv")

VITALDB_DATA_URL = "https://api.vitaldb.net/cases"
VITALDB_TRACKS_URL = "https://api.vitaldb.net/trks"

PPG_SIGNAL_NAME = 'SNUADC/PLETH'
SAMPLE_RATE_HZ = 100
WINDOW_DURATION_SECONDS = 30
SAMPLES_PER_WINDOW = SAMPLE_RATE_HZ * WINDOW_DURATION_SECONDS
VALID_WINDOW_MINUTES = 8  # +/- window around 'opstart' for BG stability

# --- UTILITY FUNCTIONS ---
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_ppg_features(ppg_series, fs):
    """
    Extracts a set of robust, hand-engineered features from a single PPG signal window.
    """
    features = {}

    # Clean the data: remove NaN values
    ppg_series_clean = ppg_series[np.isfinite(ppg_series)]

    if ppg_series_clean.size == 0:
        return {
            'ppg_mean': 0, 'ppg_std': 0, 'mean_pp_interval_s': 0,
            'std_pp_interval_s': 0, 'ppg_freq': 0, 'auc': 0,
            'first_deriv_max': 0, 'first_deriv_min': 0, 'entropy': 0
        }

    features['ppg_mean'] = np.mean(ppg_series_clean)
    features['ppg_std'] = np.std(ppg_series_clean)

    peaks, _ = find_peaks(ppg_series_clean, distance=50, height=0)
    if len(peaks) > 1:
        pp_intervals = np.diff(peaks)
        features['mean_pp_interval_s'] = np.mean(pp_intervals) / fs
        features['std_pp_interval_s'] = np.std(pp_intervals) / fs
        features['ppg_freq'] = fs / np.mean(pp_intervals)
    else:
        features['mean_pp_interval_s'] = 0
        features['std_pp_interval_s'] = 0
        features['ppg_freq'] = 0

    features['auc'] = np.trapezoid(ppg_series_clean)

    derivative = np.diff(ppg_series_clean)
    features['first_deriv_max'] = np.max(derivative)
    features['first_deriv_min'] = np.min(derivative)

    hist, _ = np.histogram(ppg_series_clean, bins='auto')
    features['entropy'] = entropy(hist)

    return features

# --- MAIN DATA PROCESSING LOOP ---
print("Starting data processing...")

# Load clinical information and track list
df_cases = pd.read_csv(VITALDB_DATA_URL)
df_trks = pd.read_csv(VITALDB_TRACKS_URL)

# Filter for cases with demographic and BG data
bg_data_cases = df_cases[df_cases['preop_gluc'].notna() & df_cases['age'].notna()].copy()
caseids_to_process = list(bg_data_cases['caseid'].unique())

print(f"Found {len(caseids_to_process)} cases with relevant data to process.")

# Check if the output file already exists to decide whether to write the header
output_file_exists = os.path.exists(OUTPUT_FILE)

for caseid in caseids_to_process:
    print(f"Processing Case ID: {caseid}...")

    # Load raw PPG signal for the case
    vals = vitaldb.load_case(caseid, [PPG_SIGNAL_NAME], 1/SAMPLE_RATE_HZ)

    if vals is None or vals.size == 0:
        print(f"  No PPG data found for Case ID {caseid}. Skipping.")
        continue

    ppg_signal = vals[:, 0]

    # Get metadata for the current case
    case_meta = bg_data_cases[bg_data_cases['caseid'] == caseid].iloc[0]

    # Filter signal within the valid time window around 'opstart' for BG stability
    opstart_seconds = case_meta['opstart']
    start_idx = max(0, int((opstart_seconds - VALID_WINDOW_MINUTES * 60) * SAMPLE_RATE_HZ))
    end_idx = min(len(ppg_signal), int((opstart_seconds + VALID_WINDOW_MINUTES * 60) * SAMPLE_RATE_HZ))

    windowed_signal = ppg_signal[start_idx:end_idx]

    if len(windowed_signal) < SAMPLES_PER_WINDOW:
        print(f"  Signal for Case ID {caseid} is too short. Skipping.")
        continue

    # Apply bandpass filter to the entire windowed signal to remove noise
    filtered_signal = butter_bandpass_filter(windowed_signal, lowcut=0.5, highcut=8, fs=SAMPLE_RATE_HZ)

    # Process and save windows for the current case
    case_data = []
    for i in range(0, len(filtered_signal) - SAMPLES_PER_WINDOW + 1, SAMPLES_PER_WINDOW):
        ppg_window = filtered_signal[i:i + SAMPLES_PER_WINDOW]

        # Extract hand-engineered features
        features = extract_ppg_features(ppg_window, SAMPLE_RATE_HZ)

        row = {
            'caseid': caseid,
            'preop_gluc': case_meta['preop_gluc'],
            'age': case_meta['age'],
            'sex': 1 if case_meta['sex'] == 'F' else 0,
            'preop_dm': case_meta['preop_dm'],
            'weight': case_meta['weight'],
            'height': case_meta['height'],
            **features # Unpack the extracted features
        }
        case_data.append(row)

    # Convert list of dicts to a DataFrame and append to the CSV
    case_df = pd.DataFrame(case_data)

    # Write the header only once at the beginning if the file is new
    header = not output_file_exists
    case_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
    # Set the flag to False after the first write
    output_file_exists = True

print(f"\nProcessing complete. All data saved to {OUTPUT_FILE}.")
