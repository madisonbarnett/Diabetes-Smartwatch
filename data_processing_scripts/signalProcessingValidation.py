import pandas as pd
import vitaldb
import os
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import entropy, skew

# ---------------- CONFIG ----------------
PPG_SIGNAL_NAME = 'SNUADC/PLETH'
SAMPLE_RATE_HZ = 100
WINDOW_DURATION_SECONDS = 15
SAMPLES_PER_WINDOW = SAMPLE_RATE_HZ * WINDOW_DURATION_SECONDS

NUM_PATIENTS = 3
WINDOWS_PER_PATIENT = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_OUTPUT = os.path.join(PROJECT_ROOT, "sigVal_raw_ppg.csv")
PROCESSED_OUTPUT = os.path.join(PROJECT_ROOT, "sigVal_processed_ppg.csv")

# ---------------- FILTER ----------------
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    return butter(order, [lowcut/nyq, highcut/nyq], btype='band')

def bandpass_filter(data):
    b, a = butter_bandpass(0.5, 8, SAMPLE_RATE_HZ)
    return filtfilt(b, a, data)

def teager_energy(x):
    return np.mean(np.abs(x[1:-1]**2 - x[:-2]*x[2:])) if len(x) > 2 else 0

# ---------------- FEATURES ----------------
def extract_features(ppg, fs):
    ppg = ppg[np.isfinite(ppg)]
    if len(ppg) < 3:
        return None

    features = {}

    features['ppg_std'] = np.std(ppg)

    peaks, _ = find_peaks(ppg, distance=int(fs*0.4))
    if len(peaks) > 1:
        intervals = np.diff(peaks)
        features['ppg_mean_pp_interval_s'] = np.mean(intervals)/fs
        features['ppg_std_pp_interval_s'] = np.std(intervals)/fs
    else:
        return None  # ❗ reject bad windows

    ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)

    features['ppg_teager_energy'] = teager_energy(ppg)
    features['ppg_skew'] = skew(ppg)
    features['ppg_iqr'] = np.percentile(ppg, 75) - np.percentile(ppg, 25)

    hist, _ = np.histogram(ppg, bins='auto')
    features['ppg_entropy'] = entropy(hist)

    deriv = np.diff(ppg)
    features['ppg_first_deriv_max'] = np.max(deriv)

    return features

# ---------------- LOAD ----------------
labs = pd.read_csv("https://api.vitaldb.net/labs")
cases = pd.read_csv("https://api.vitaldb.net/cases")

labs = labs[labs['name'] == 'gluc']
cases = cases[cases['age'].notna()]

all_caseids = cases['caseid'].unique()

processed_rows = []
raw_rows = []

valid_patients = 0

print("Searching for valid patients...")

# ---------------- MAIN LOOP ----------------
for caseid in all_caseids:

    if valid_patients >= NUM_PATIENTS:
        break

    print(f"\nChecking case {caseid}")

    case_meta = cases[cases['caseid'] == caseid].iloc[0]
    case_labs = labs[labs['caseid'] == caseid].sort_values("dt")

    if len(case_labs) < WINDOWS_PER_PATIENT:
        print("  ❌ Not enough glucose values")
        continue

    # Load PPG
    ppg_data = vitaldb.load_case(caseid, [PPG_SIGNAL_NAME], 1/SAMPLE_RATE_HZ)
    if ppg_data is None or len(ppg_data) == 0:
        print("  ❌ No PPG data")
        continue

    ppg_signal = ppg_data[:, 0]

    valid_windows = []
    raw_windows = []

    # Try to extract up to 3 valid windows
    for _, lab_row in case_labs.iterrows():

        if len(valid_windows) >= WINDOWS_PER_PATIENT:
            break

        dt = lab_row['dt']
        gluc = lab_row['result']

        start = int(dt * SAMPLE_RATE_HZ)
        end = start + SAMPLES_PER_WINDOW

        if end > len(ppg_signal):
            continue

        raw_window = ppg_signal[start:end]

        if len(raw_window) < SAMPLES_PER_WINDOW:
            continue

        filtered = bandpass_filter(raw_window)

        feats = extract_features(filtered, SAMPLE_RATE_HZ)

        if feats is None:
            continue  # reject bad signal

        valid_windows.append((feats, gluc, raw_window))

    # Ensure EXACTLY 3 valid windows
    if len(valid_windows) < WINDOWS_PER_PATIENT:
        print("  ❌ Could not find 3 valid windows")
        continue

    print("  ✅ Valid patient accepted")

    # Save data
    for win_idx, (feats, gluc, raw_window) in enumerate(valid_windows):

        # RAW
        for i, val in enumerate(raw_window):
            raw_rows.append({
                "caseid": caseid,
                "window_id": win_idx,
                "sample_index": i,
                "ppg": val
            })

        # FEATURES
        processed_rows.append({
            "caseid": caseid,
            "window_id": win_idx,
            "gluc": gluc,
            "age": case_meta['age'],
            "weight": case_meta['weight'],
            "height": case_meta['height'],
            "preop_dm": case_meta['preop_dm'],
            **feats
        })

    valid_patients += 1

# ---------------- FINAL CHECK ----------------
if len(processed_rows) != NUM_PATIENTS * WINDOWS_PER_PATIENT:
    raise ValueError(f"Expected {NUM_PATIENTS * WINDOWS_PER_PATIENT} rows but got {len(processed_rows)}")

# ---------------- SAVE ----------------
pd.DataFrame(processed_rows).to_csv(PROCESSED_OUTPUT, index=False)
pd.DataFrame(raw_rows).to_csv(RAW_OUTPUT, index=False)

print("\nDone.")
print(f"Saved {len(processed_rows)} feature rows (should be 9)")