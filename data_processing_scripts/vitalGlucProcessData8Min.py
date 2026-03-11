# This script processes VitalDB glucose lab measurements and matches them to +/- 8 min windows of PPG data,
# following the Zeynali et al. (2025) methodology: downsampling, filtering, 10-s coarse segmentation,
# peak-centered 1-s fine segmentation with stricter quality filtering, normalization, and flattening for MLP input.
import pandas as pd
import vitaldb
import os
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.spatial.distance import cosine

# Download VitalDB metadata
VITALDB_LABS_URL = "https://api.vitaldb.net/labs"
df_labs = pd.read_csv(VITALDB_LABS_URL)

VITALDB_DATA_URL = "https://api.vitaldb.net/cases"
df_cases = pd.read_csv(VITALDB_DATA_URL)

# Output files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join("./processed_data", "vitaldb_1s_ppg_segments_flattened_clean.csv")

# Processing parameters
PPG_SIGNAL_NAME = 'SNUADC/PLETH'
ORIGINAL_SAMPLE_RATE_HZ = 500
TARGET_SAMPLE_RATE_HZ = 100          # as in the paper
DOWNSAMPLE_FACTOR = ORIGINAL_SAMPLE_RATE_HZ // TARGET_SAMPLE_RATE_HZ

WINDOW_DURATION_MIN = 8              # +/- 8 min around each glucose timestamp
WINDOW_DURATION_SEC = WINDOW_DURATION_MIN * 60 * 2  # 16 min total
COARSE_SEGMENT_SEC = 10              # 10-second chunks
FINE_WINDOW_SEC = 1                  # 1-second final windows
SAMPLES_PER_FINE_WINDOW = TARGET_SAMPLE_RATE_HZ * FINE_WINDOW_SEC  # 100 samples

COSINE_SIMILARITY_THRESHOLD = 0.92   # stricter than 0.85 — better quality, fewer noisy/redundant pulses
MAX_GOOD_WINDOWS_PER_COARSE = 30     # hard cap to prevent explosion when many peaks pass
BATCH_SIZE = 50                      # cases to accumulate before writing to CSV

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

def downsample_signal(signal, original_fs, target_fs):
    """Simple decimation (integer factor only)"""
    if original_fs % target_fs != 0:
        raise ValueError("Downsample factor must be integer")
    factor = original_fs // target_fs
    return signal[::factor]

def normalize_window(window):
    """Min-max normalize to [0,1] per window"""
    min_val = np.min(window)
    max_val = np.max(window)
    if max_val == min_val:
        return np.zeros_like(window)
    return (window - min_val) / (max_val - min_val)

def cosine_similarity(a, b):
    """Cosine similarity between two vectors (1 - cosine distance)"""
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return 1 - cosine(a, b)

# MAIN DATA PROCESSING SECTION

# Filter glucose labs
labs_gluc_cases = df_labs[df_labs['name'] == 'gluc'].copy()

# Filter for cases with demographic data
bg_data_cases = df_cases[df_cases['age'].notna()].copy()
caseids_to_process = list(bg_data_cases['caseid'].unique())

print(f"Found {len(caseids_to_process)} cases with relevant data to process.")

all_segments = []  # list of dicts: one per 1-s segment

print("Starting data processing...")

for caseid in caseids_to_process:
    print(f"Processing Case ID: {caseid}...")

    gluc_at_caseid = labs_gluc_cases[labs_gluc_cases['caseid'] == caseid]

    if gluc_at_caseid.empty:
        print(f"  No glucose values for Case ID: {caseid}. Skipping.")
        continue

    # Get case timing
    case_meta = bg_data_cases[bg_data_cases['caseid'] == caseid].iloc[0]
    caseend_seconds = case_meta['caseend']

    # Filter glucose readings to within case duration
    valid_samples = gluc_at_caseid[
        (gluc_at_caseid['dt'] >= 0) &
        (gluc_at_caseid['dt'] <= caseend_seconds)
    ]

    if valid_samples.empty:
        print(f"  No valid glucose values for Case ID: {caseid}. Skipping.")
        continue

    # Load raw PPG signal for the entire case
    ppg_vals = vitaldb.load_case(caseid, [PPG_SIGNAL_NAME], 1/ORIGINAL_SAMPLE_RATE_HZ)

    if ppg_vals is None or ppg_vals.size == 0:
        print(f"  No PPG data found for Case ID {caseid}. Skipping.")
        continue

    ppg_signal = ppg_vals[:, 0]  # shape: (n_samples,)

    # Process each glucose measurement
    for _, row in valid_samples.iterrows():
        tm_seconds = row['dt']           # timestamp of glucose measurement
        gluc_val = row['result']

        # Crop ±8 minutes around tm
        start_sec = max(0, tm_seconds - WINDOW_DURATION_MIN * 60)
        end_sec = min(caseend_seconds, tm_seconds + WINDOW_DURATION_MIN * 60)

        start_idx = int(start_sec * ORIGINAL_SAMPLE_RATE_HZ)
        end_idx = int(end_sec * ORIGINAL_SAMPLE_RATE_HZ)

        cropped_ppg = ppg_signal[start_idx:end_idx]

        if len(cropped_ppg) < (WINDOW_DURATION_SEC * ORIGINAL_SAMPLE_RATE_HZ * 0.5):
            print(f"  Cropped PPG too short for Case {caseid} at t={tm_seconds:.1f}s. Skipping.")
            continue

        # Downsample to 100 Hz
        downsampled_ppg = downsample_signal(cropped_ppg, ORIGINAL_SAMPLE_RATE_HZ, TARGET_SAMPLE_RATE_HZ)

        # Forward fill any rare NaNs (though VitalDB waveforms are usually clean)
        downsampled_ppg = pd.Series(downsampled_ppg).ffill().bfill().values

        # Apply Butterworth bandpass filter (0.5-8 Hz, 3rd order)
        filtered_ppg = butter_bandpass_filter(
            downsampled_ppg, lowcut=0.5, highcut=8, fs=TARGET_SAMPLE_RATE_HZ, order=3
        )

        # Coarse segmentation: split into 10-second non-overlapping chunks
        samples_per_coarse = COARSE_SEGMENT_SEC * TARGET_SAMPLE_RATE_HZ  # 1000 samples
        n_coarse = len(filtered_ppg) // samples_per_coarse

        for i in range(n_coarse):
            coarse_start = i * samples_per_coarse
            coarse_end = coarse_start + samples_per_coarse
            coarse_segment = filtered_ppg[coarse_start:coarse_end]

            # Peak detection — stricter thresholds to reduce redundancy
            height_thresh = np.percentile(np.abs(coarse_segment), 40)   # was 20 — filters more noise
            peaks, _ = find_peaks(
                coarse_segment,
                height=height_thresh,
                distance=TARGET_SAMPLE_RATE_HZ * 0.6   # was 0.4 — min ~0.6 s between peaks
            )

            if len(peaks) == 0:
                continue

            # Extract 1-s windows centered on each peak
            half_win_samples = SAMPLES_PER_FINE_WINDOW // 2  # 50 samples
            candidate_windows = []

            for peak_idx in peaks:
                win_start = max(0, peak_idx - half_win_samples)
                win_end = min(len(coarse_segment), peak_idx + half_win_samples + (SAMPLES_PER_FINE_WINDOW % 2))
                if win_end - win_start == SAMPLES_PER_FINE_WINDOW:
                    candidate_windows.append(coarse_segment[win_start:win_end])

            if not candidate_windows:
                continue

            # Compute template = mean of all candidate windows in this coarse segment
            template = np.mean(candidate_windows, axis=0)

            # Quality filter: stricter threshold
            good_windows = []
            for w in candidate_windows:
                sim = cosine_similarity(w, template)
                if sim >= COSINE_SIMILARITY_THRESHOLD:
                    good_windows.append(w)

            # Cap number of good windows per 10-s segment to avoid redundancy explosion
            if len(good_windows) > MAX_GOOD_WINDOWS_PER_COARSE:
                # Randomly subsample to cap (could also sort by similarity and take top N)
                np.random.shuffle(good_windows)
                good_windows = good_windows[:MAX_GOOD_WINDOWS_PER_COARSE]

            # Normalize and flatten good windows
            for w in good_windows:
                norm_w = normalize_window(w)  # [0,1]
                flat_vector = norm_w.flatten()  # 100 values

                row = {
                    'caseid': caseid,
                    'gluc': gluc_val,
                    'age': case_meta['age'],
                    'sex': 1 if case_meta['sex'] == 'F' else 0,
                    'preop_dm': case_meta['preop_dm'],
                    'weight': case_meta['weight'],
                    'height': case_meta['height'],
                    'bmi': case_meta['bmi'],
                    # flattened 1-s PPG vector (100 columns named ppg_0 to ppg_99)
                    **{f'ppg_{j}': val for j, val in enumerate(flat_vector)}
                }
                all_segments.append(row)

    # Batch write to CSV
    if len(all_segments) >= BATCH_SIZE * 50:  # adjusted heuristic (fewer segments expected now)
        batch_df = pd.DataFrame(all_segments)
        output_file_exists = os.path.exists(OUTPUT_FILE)
        header = not output_file_exists
        batch_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
        print(f"Wrote {len(batch_df)} 1-s segments to {OUTPUT_FILE}")
        all_segments.clear()

# Final write
if all_segments:
    batch_df = pd.DataFrame(all_segments)
    output_file_exists = os.path.exists(OUTPUT_FILE)
    header = not output_file_exists
    batch_df.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
    print(f"Final write: saved {len(batch_df)} remaining 1-s segments to {OUTPUT_FILE}")

# Optional: final dataset-level deduplication (safety net)
print("Performing final dataset-level deduplication...")
full_df = pd.read_csv(OUTPUT_FILE)
ppg_cols = [f'ppg_{i}' for i in range(100)]
full_df = full_df.drop_duplicates(subset=ppg_cols, keep='first')
full_df.to_csv(OUTPUT_FILE, index=False)
print(f"After deduplication: {len(full_df)} rows remain.")

print(f"\nProcessing complete. Cleaned 1-second flattened PPG segments saved to {OUTPUT_FILE}.")