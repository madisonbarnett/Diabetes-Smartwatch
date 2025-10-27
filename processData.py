import pandas as pd
import vitaldb
import numpy as np
import os
from scipy.signal import find_peaks, butter, lfilter, filtfilt
from scipy.stats import entropy

# --- CONFIGURATION ---
OUTPUT_DIR = "./processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vitaldb_bp_extracted_features.csv")

VITALDB_DATA_URL = "https://api.vitaldb.net/cases"
VITALDB_TRACKS_URL = "https://api.vitaldb.net/trks"

PPG_SIGNAL_NAME = 'SNUADC/PLETH'
ECG_SIGNAL_NAME = 'SNUADC/ECG_II'
MEAN_BP_NAME = 'Solar8000/ART_MBP'
SYS_BP_NAME = 'Solar8000/ART_SBP'
DYS_BP_NAME = 'Solar8000/ART_DBP'
SAMPLE_RATE_HZ = 500
BP_SAMPLE_RATE_HZ = 0.5
WINDOW_DURATION_SECONDS = 30
SAMPLES_PER_WINDOW = SAMPLE_RATE_HZ * WINDOW_DURATION_SECONDS
BP_SAMPLES_PER_WINDOW = int(BP_SAMPLE_RATE_HZ * WINDOW_DURATION_SECONDS)
VALID_WINDOW_MINUTES = 8  # +/- window around 'opstart' for BG stability
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
            'ppg_mean': 0, 'ppg_std': 0, 'mean_pp_interval_s': 0,
            'std_pp_interval_s': 0, 'ppg_freq': 0, 'auc': 0,
            'first_deriv_max': 0, 'first_deriv_min': 0, 'entropy': 0
        }

    features['ppg_mean'] = np.mean(ppg_series_clean)
    features['ppg_std'] = np.std(ppg_series_clean)

    peaks, _ = find_peaks(ppg_series_clean, distance=int(fs*0.4), height=0)
    if len(peaks) > 1:
        pp_intervals = np.diff(peaks)
        features['mean_pp_interval_s'] = np.mean(pp_intervals) / fs
        features['std_pp_interval_s'] = np.std(pp_intervals) / fs
        features['ppg_freq'] = fs / np.mean(pp_intervals)
    else:
        features['mean_pp_interval_s'] = 0
        features['std_pp_interval_s'] = 0
        features['ppg_freq'] = 0

    # Normalize amplitude before calculating cumulative metrics
    ppg_series_clean = (ppg_series_clean - np.mean(ppg_series_clean)) / np.std(ppg_series_clean)

    features['auc'] = np.trapezoid(ppg_series_clean)

    derivative = np.diff(ppg_series_clean)
    features['first_deriv_max'] = np.max(derivative)
    features['first_deriv_min'] = np.min(derivative)

    hist, _ = np.histogram(ppg_series_clean, bins='auto')
    features['entropy'] = entropy(hist)

    return features

def extract_ecg_features(ecg_series, fs):
    """
    Extracts a set of robust, hand-engineered features from a single ECG signal window.
    """
    features = {}

    # Clean the data: remove NaN values
    ecg_series_clean = ecg_series[np.isfinite(ecg_series)]

    if ecg_series_clean.size == 0:
        return {
            'ecg_mean': 0, 'ecg_std': 0, 'ecg_mean_pp_interval_s': 0,
            'ecg_std_pp_interval_s': 0, 'ecg_freq': 0, 'ecg_auc': 0,
            'ecg_first_deriv_max': 0, 'ecg_first_deriv_min': 0, 'ecg_entropy': 0
        }

    features['ecg_mean'] = np.mean(ecg_series_clean)
    features['ecg_std'] = np.std(ecg_series_clean)

    peaks, _ = find_peaks(ecg_series_clean, distance=50, height=0)
    if len(peaks) > 1:
        pp_intervals = np.diff(peaks)
        features['ecg_mean_pp_interval_s'] = np.mean(pp_intervals) / fs
        features['ecg_std_pp_interval_s'] = np.std(pp_intervals) / fs
        features['ecg_freq'] = fs / np.mean(pp_intervals)
    else:
        features['ecg_mean_pp_interval_s'] = 0
        features['ecg_std_pp_interval_s'] = 0
        features['ecg_freq'] = 0

    features['ecg_auc'] = np.trapezoid(ecg_series_clean)

    derivative = np.diff(ecg_series_clean)
    features['ecg_first_deriv_max'] = np.max(derivative)
    features['ecg_first_deriv_min'] = np.min(derivative)

    hist, _ = np.histogram(ecg_series_clean, bins='auto')
    features['ecg_entropy'] = entropy(hist)

    return features

# --- MAIN DATA PROCESSING LOOP ---
print("Starting data processing...")

# Load clinical information and track list
df_cases = pd.read_csv(VITALDB_DATA_URL)
df_trks = pd.read_csv(VITALDB_TRACKS_URL)

# Filter for cases with demographic data
all_data_cases = df_cases[df_cases['age'].notna()].copy()
caseids_to_process = list(all_data_cases['caseid'].unique())

print(f"Found {len(caseids_to_process)} cases with relevant data to process.")

all_case_dfs = []

# Check if the output file already exists to decide whether to write the header
output_file_exists = os.path.exists(OUTPUT_FILE)

for caseid in caseids_to_process:
    print(f"Processing Case ID: {caseid}...")

    # Load raw mean blood pressure values for the case
    mean_bp_vals = vitaldb.load_case(caseid, [MEAN_BP_NAME], 1/BP_SAMPLE_RATE_HZ)

    if mean_bp_vals is None or mean_bp_vals.size == 0:
        print(f"  No BP data found for Case ID {caseid}. Skipping.")
        continue

    mean_bp_signal = mean_bp_vals[:, 0]

    # Load raw systolic blood pressure values for the case
    sys_bp_vals = vitaldb.load_case(caseid, [SYS_BP_NAME], 1/BP_SAMPLE_RATE_HZ)

    if sys_bp_vals is None or sys_bp_vals.size == 0:
        print(f"  No Systolic BP data found for Case ID {caseid}. Skipping.")
        continue

    sys_bp_signal = sys_bp_vals[:, 0]

    # Load raw diastolic blood pressure values for the case
    dys_bp_vals = vitaldb.load_case(caseid, [DYS_BP_NAME], 1/BP_SAMPLE_RATE_HZ)

    if dys_bp_vals is None or dys_bp_vals.size == 0:
        print(f"  No Diastolic BP data found for Case ID {caseid}. Skipping.")
        continue

    dys_bp_signal = dys_bp_vals[:, 0]
    
    # Load raw PPG signal for the case
    ppg_vals = vitaldb.load_case(caseid, [PPG_SIGNAL_NAME], 1/SAMPLE_RATE_HZ)

    if ppg_vals is None or ppg_vals.size == 0:
        print(f"  No PPG data found for Case ID {caseid}. Skipping.")
        continue

    ppg_signal = ppg_vals[:, 0]

    # Load raw ECG signal for the case
    ecg_vals = vitaldb.load_case(caseid, [ECG_SIGNAL_NAME], 1/SAMPLE_RATE_HZ)

    if ecg_vals is None or ecg_vals.size == 0:
        print(f"  No ECG data found for Case ID {caseid}. Skipping.")
        continue

    ecg_signal = ecg_vals[:, 0]

    # Get metadata for the current case
    case_meta = all_data_cases[all_data_cases['caseid'] == caseid].iloc[0]

    # Filter both signals within the valid time window around 'opstart' for BG stability
    opstart_seconds = case_meta['opstart']

    # Start index is the greater of the two: 0 or index corresponding to VALID_WINDOW_MINUTES before opstart time
    start_idx = max(0, int((opstart_seconds - VALID_WINDOW_MINUTES * 60) * BP_SAMPLE_RATE_HZ))
    # End index is the lesser of the two: the last index in the sample or index corresponding to VALID_WINDOW_MINUTES after opstart time
    end_idx = min(len(mean_bp_signal), int((opstart_seconds + VALID_WINDOW_MINUTES * 60) * BP_SAMPLE_RATE_HZ))
    
    windowed_mean_bp_signal = mean_bp_signal[start_idx:end_idx]

    if len(windowed_mean_bp_signal) < BP_SAMPLES_PER_WINDOW:
        print(f"  Time series of Mean BP values for Case ID {caseid} is too short. Skipping.")
        continue

    # Resample end index for systolic blood pressure series
    end_idx = min(len(sys_bp_signal), int((opstart_seconds + VALID_WINDOW_MINUTES * 60) * BP_SAMPLE_RATE_HZ))
    
    windowed_sys_bp_signal = sys_bp_signal[start_idx:end_idx]

    if len(windowed_sys_bp_signal) < BP_SAMPLES_PER_WINDOW:
        print(f"  Time series of Systolic BP values for Case ID {caseid} is too short. Skipping.")
        continue

    # Resample end index for systolic blood pressure series
    end_idx = min(len(dys_bp_signal), int((opstart_seconds + VALID_WINDOW_MINUTES * 60) * BP_SAMPLE_RATE_HZ))
    
    windowed_dys_bp_signal = dys_bp_signal[start_idx:end_idx]

    if len(windowed_dys_bp_signal) < BP_SAMPLES_PER_WINDOW:
        print(f"  Time series of Diastolic BP values for Case ID {caseid} is too short. Skipping.")
        continue

    # Start index is the greater of the two: 0 or index corresponding to VALID_WINDOW_MINUTES before opstart time
    start_idx = max(0, int((opstart_seconds - VALID_WINDOW_MINUTES * 60) * SAMPLE_RATE_HZ))
    # End index is the lesser of the two: the last index in the waveform or index corresponding to VALID_WINDOW_MINUTES after opstart time
    end_idx = min(len(ppg_signal), int((opstart_seconds + VALID_WINDOW_MINUTES * 60) * SAMPLE_RATE_HZ))

    windowed_ppg_signal = ppg_signal[start_idx:end_idx]

    if len(windowed_ppg_signal) < SAMPLES_PER_WINDOW:
        print(f"  PPG signal for Case ID {caseid} is too short. Skipping.")
        continue

    # Apply bandpass filter to the entire windowed signal to remove noise
    filtered_ppg_signal = butter_bandpass_filter(windowed_ppg_signal, lowcut=0.5, highcut=8, fs=SAMPLE_RATE_HZ)

    # Resample end index for ecg signal
    end_idx = min(len(ecg_signal), int((opstart_seconds + VALID_WINDOW_MINUTES * 60) * SAMPLE_RATE_HZ))

    windowed_ecg_signal = ecg_signal[start_idx:end_idx]

    if len(windowed_ecg_signal) < SAMPLES_PER_WINDOW:
        print(f"  ECG signal for Case ID {caseid} is too short. Skipping.")
        continue

    # Apply bandpass filter to the entire windowed signal to remove noise
    filtered_ecg_signal = butter_bandpass_filter(windowed_ecg_signal, lowcut=0.5, highcut=8, fs=SAMPLE_RATE_HZ)

    sampleLen = min(len(filtered_ppg_signal), len(filtered_ecg_signal))

    # Process and save windows for the current case
    case_data = []
    for i in range(0, sampleLen - SAMPLES_PER_WINDOW + 1, SAMPLES_PER_WINDOW):
        ppg_window = filtered_ppg_signal[i:i + SAMPLES_PER_WINDOW]

        # Extract hand-engineered features
        ppg_features = extract_ppg_features(ppg_window, SAMPLE_RATE_HZ)

        # Skip samples where PPG is null or infinite (as returned by extract_ppg_features function)
        if all(v == 0 for v in ppg_features.values()):
            continue

        ecg_window = filtered_ecg_signal[i:i + SAMPLES_PER_WINDOW]

        # Extract hand-engineered features
        ecg_features = extract_ecg_features(ecg_window, SAMPLE_RATE_HZ)

        # Skip samples where ECG is null or infinite (as returned by extract_ecg_features function)
        if all(v == 0 for v in ecg_features.values()):
            continue
        
        # Restructure indexing for BP samples
        j = int(i / (SAMPLES_PER_WINDOW / BP_SAMPLES_PER_WINDOW))

        mean_bp_window = windowed_mean_bp_signal[j:j + BP_SAMPLES_PER_WINDOW]

        # Extract target mean blood pressure across the window
        mean_bp = extract_target_bp(mean_bp_window)

        sys_bp_window = windowed_sys_bp_signal[j:j + BP_SAMPLES_PER_WINDOW]

        # Extract target systolic blood pressure across the window
        sys_bp = extract_target_bp(sys_bp_window)

        dys_bp_window = windowed_dys_bp_signal[j:j + BP_SAMPLES_PER_WINDOW]

        # Extract target diastolic blood pressure across the window
        dys_bp = extract_target_bp(dys_bp_window)

        # Skip samples where all BP values are null or infinite (as returned by extract_target_bp function)
        if(mean_bp == 0 or sys_bp == 0 or dys_bp == 0):
            continue

        row = {
            'caseid': caseid,
            'preop_gluc': case_meta['preop_gluc'],
            'mean_bp': mean_bp,
            'sys_bp': sys_bp,
            'dys_bp': dys_bp,
            'age': case_meta['age'],
            'sex': 1 if case_meta['sex'] == 'F' else 0,
            'preop_dm': case_meta['preop_dm'],
            'weight': case_meta['weight'],
            'height': case_meta['height'],
            **ppg_features, # Unpack the extracted PPG features
            **ecg_features # Unpack the extracted ECG features
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
