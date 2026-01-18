"""
plot_ppg_waveform_15s_opstart_dual.py
Author: Tyler Bish | CWID: 12204514
Description:
    Loads a VitalDB case, extracts a 15-second PPG window starting
    1 minute after operation start time ('opstart'), and plots both
    the raw and filtered PPG waveforms for comparison.
"""

import pandas as pd
import vitaldb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- CONFIGURATION ---
CASE_ID = 1                     # Change this to desired VitalDB case ID
PPG_SIGNAL_NAME = 'SNUADC/PLETH'
SAMPLE_RATE_HZ = 500
WINDOW_DURATION_SECONDS = 15
LOWCUT = 0.5                    # Bandpass filter lower cutoff (Hz)
HIGHCUT = 8                     # Bandpass filter upper cutoff (Hz)
VITALDB_DATA_URL = "https://api.vitaldb.net/cases"

# --- HELPER FUNCTIONS ---
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

# --- LOAD METADATA ---
print("Loading case metadata...")
df_cases = pd.read_csv(VITALDB_DATA_URL)

# Verify case exists
if CASE_ID not in df_cases['caseid'].values:
    raise ValueError(f"Case ID {CASE_ID} not found in VitalDB case list.")

case_meta = df_cases[df_cases['caseid'] == CASE_ID].iloc[0]
opstart_seconds = case_meta['opstart']

print(f"Case ID: {CASE_ID}")
print(f"Operation start time: {opstart_seconds:.2f} seconds")

# --- LOAD PPG SIGNAL ---
print(f"Loading PPG signal for Case ID {CASE_ID}...")
ppg_data = vitaldb.load_case(CASE_ID, [PPG_SIGNAL_NAME], 1 / SAMPLE_RATE_HZ)

if ppg_data is None or ppg_data.size == 0:
    raise ValueError(f"No PPG data found for Case ID {CASE_ID}.")

ppg_signal = ppg_data[:, 0]
print(f"Loaded {len(ppg_signal)} samples of PPG data.")

# --- EXTRACT 15-SECOND WINDOW (1 MIN AFTER OPSTART) ---
start_time = opstart_seconds + 60  # 1 minute after opstart
start_idx = int(start_time * SAMPLE_RATE_HZ)
end_idx = start_idx + int(WINDOW_DURATION_SECONDS * SAMPLE_RATE_HZ)

if end_idx > len(ppg_signal):
    raise ValueError(f"Case {CASE_ID} shorter than requested window.")

ppg_window_raw = ppg_signal[start_idx:end_idx]

# --- FILTER THE SIGNAL ---
ppg_window_filtered = butter_bandpass_filter(ppg_window_raw, LOWCUT, HIGHCUT, SAMPLE_RATE_HZ)

# --- TIME AXIS ---
time_axis = np.arange(len(ppg_window_raw)) / SAMPLE_RATE_HZ

# --- PLOT BOTH WAVEFORMS ---
plt.figure(figsize=(10, 6))

# Raw PPG
plt.subplot(2, 1, 1)
plt.plot(time_axis, ppg_window_raw, color='red', linewidth=1)
plt.title(f"Raw PPG Waveform (15s Window) — Case {CASE_ID}\nStart = opstart + 60s")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid(True)

# Filtered PPG
plt.subplot(2, 1, 2)
plt.plot(time_axis, ppg_window_filtered, color='blue', linewidth=1)
plt.title("Filtered PPG Waveform (0.5–8 Hz Bandpass)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid(True)

plt.tight_layout()
plt.show()
