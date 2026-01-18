import pandas as pd
import vitaldb
import numpy as np
import os
from scipy.signal import find_peaks, butter, lfilter, filtfilt
from scipy.stats import entropy, skew

# This script extracts ECG, PPG, BG, and BP

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "PhysionetData")

# UPDATE THE OUTPUT FILE ACCORDING TO SAMPLE WINDOW SIZE (e.g. _30s for 30 second window)

BG_ALL_PATH = os.path.join(DATA_DIR, "bg_data_all_patients.csv")
PPG_ALL_PATH = os.path.join(DATA_DIR, "ppg_data_all_patients.csv")

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
OUTPUT_FILENAME = "physioNet_ppg_extracted_features_30s.csv"
OUTPUT_FILE = os.path.join(DATA_DIR, OUTPUT_FILENAME)

SAMPLE_RATE_HZ = 64
WINDOW_DURATION_SECONDS = 30
SAMPLES_PER_WINDOW = SAMPLE_RATE_HZ * WINDOW_DURATION_SECONDS
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

def extract_bg_data(bg_df):
    """
    Extracts only timestamp and bg value from Dexcom csv files
    """
    ts_col = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    g_col  = "Glucose Value (mg/dL)"    

    # Keep only true glucose readings (EGV rows)
    egv = bg_df.loc[bg_df["Event Type"].eq("EGV"), [ts_col, g_col]].copy()

    # Parse / clean
    egv[ts_col] = pd.to_datetime(egv[ts_col], errors="coerce")
    egv[g_col]  = pd.to_numeric(egv[g_col], errors="coerce")

    # Drop anything still bad, sort, reset index
    egv = egv.dropna(subset=[ts_col, g_col]).sort_values(ts_col).reset_index(drop=True)

    # Optional: rename to simpler column names
    egv = egv.rename(columns={ts_col: "timestamp", g_col: "glucose_mg_dl"})
    return egv

def get_patient_start_ts(bg_all_df: pd.DataFrame, patient_id: str) -> pd.Timestamp | None:
    """Return first (min) timestamp for this patient_id as a pd.Timestamp, or None if missing."""
    sub = bg_all_df.loc[bg_all_df["patient_id"].eq(patient_id), "timestamp"]
    if sub.empty:
        return None
    return sub.min()  # already datetime if parsed

def filter_ppg_from_timestamp(ppg_df: pd.DataFrame, start_ts: pd.Timestamp, ts_col: str = "timestamp") -> pd.DataFrame:
    ppg_df = ppg_df.copy()
    ppg_df = ppg_df.rename(columns={"datetime": ts_col})
    ppg_df[ts_col] = pd.to_datetime(ppg_df[ts_col], errors="coerce")
    ppg_df = ppg_df.dropna()
    return ppg_df.loc[ppg_df[ts_col] >= start_ts].reset_index(drop=True)


# -------------------------------------------------------------------
# PASTABLE BLOCK (per-patient PPG files: BVP_001.csv ... BVP_016.csv)
# - Reads BG_ALL once
# - For each patient:
#     * reads that patient's PPG file only (much smaller than all-patients)
#     * builds 10 non-overlapping 30s windows spanning +/-2.5 minutes around each BG timestamp
#     * bandpass filters each window then calls extract_ppg_features()
#     * writes every 50 BG events so you can inspect while running
# - Writes header once
# - Drops timestamps before writing final database
#
# REQUIREMENTS:
# - butter_bandpass_filter(data, lowcut, highcut, fs, order=3) exists
# - extract_ppg_features(ppg_series, fs) exists
# - BG file has columns: patient_id, sex, timestamp, glucose_mg_dl
# - PPG per-patient file has a timestamp column and a BVP column (auto-detected)
# -------------------------------------------------------------------

def _detect_time_and_bvp_cols(df_cols):
    cols = list(df_cols)

    # timestamp candidates
    ts_candidates = ["timestamp", "datetime", "time", "Timestamp", "DateTime", "date_time"]
    # bvp candidates
    bvp_candidates = ["bvp", "BVP", "pleth", "PPG", "ppg", "signal", "value", "BVP (a.u.)"]

    norm = {c.strip().lower(): c for c in cols}

    ts_col = None
    for cand in ts_candidates:
        key = cand.strip().lower()
        if key in norm:
            ts_col = norm[key]
            break
    if ts_col is None:
        # substring fallback
        for k, orig in norm.items():
            if "time" in k or "date" in k:
                ts_col = orig
                break

    bvp_col = None
    for cand in bvp_candidates:
        key = cand.strip().lower()
        if key in norm:
            bvp_col = norm[key]
            break
    if bvp_col is None:
        for k, orig in norm.items():
            if "bvp" in k or "ppg" in k or "pleth" in k:
                bvp_col = orig
                break

    if ts_col is None or bvp_col is None:
        raise ValueError(f"Could not detect timestamp/bvp columns. Columns were: {cols}")

    return ts_col, bvp_col


def write_features_from_per_patient_bvp_files(
    bg_csv_path: str,
    bvp_dir: str,
    out_csv_path: str,
    patient_ids=range(1, 17),          # 1..16
    fs: int = 64,
    win_seconds: int = 30,
    half_window_minutes: float = 2.5,
    min_coverage: float = 0.90,
    lowcut: float = 0.5,
    highcut: float = 8.0,
    order: int = 3,
    batch_events: int = 50
):
    # ---- output prep ----
    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    header = not os.path.exists(out_csv_path)

    # Drop timestamps before writing final DB
    DROP_COLS = ["glucose_timestamp", "window_start", "window_end"]

    # ---- load BG (small) ----
    bg = pd.read_csv(bg_csv_path, dtype={"sex": "string"})
    bg["patient_id"] = pd.to_numeric(bg["patient_id"], errors="coerce").astype("Int64")
    bg["timestamp"] = pd.to_datetime(bg["timestamp"], errors="coerce")
    bg["glucose_mg_dl"] = pd.to_numeric(bg["glucose_mg_dl"], errors="coerce")

    # sex patient-level lookup
    if "sex" in bg.columns:
        bg["sex_num"] = bg["sex"].astype(str).str.lower().map({"m": 1, "f": 0})
        sex_lookup = (
            bg.dropna(subset=["patient_id", "sex_num"])
              .groupby("patient_id")["sex_num"]
              .first()
              .to_dict()
        )
    else:
        sex_lookup = {}

    bg = bg.dropna(subset=["patient_id", "timestamp", "glucose_mg_dl"]).sort_values(["patient_id", "timestamp"])
    bg_events_by_patient = {int(pid): grp.copy() for pid, grp in bg.groupby("patient_id")}

    # ---- constants ----
    half_ns = int(pd.Timedelta(minutes=half_window_minutes).value)  # ns
    win_ns  = int(pd.Timedelta(seconds=win_seconds).value)          # ns
    n_windows = int((2 * half_ns) // win_ns)                        # 10

    expected_samples = fs * win_seconds
    min_samples = int(expected_samples * min_coverage)

    print(f"[CONFIG] windows/event={n_windows} (expect 10) | min_samples/window={min_samples}/{expected_samples}")
    print(f"[FILES ] BG={bg_csv_path}")
    print(f"[FILES ] BVP_DIR={bvp_dir}")
    print(f"[OUT   ] {out_csv_path} header_first_write={header}")

    # ---- per patient ----
    for pid in patient_ids:
        pid = int(pid)

        if pid not in bg_events_by_patient:
            print(f"[SKIP] patient {pid}: no BG events")
            continue

        # per-patient BVP file path
        bvp_path = os.path.join(bvp_dir, f"BVP_{pid:03d}.csv")
        if not os.path.exists(bvp_path):
            print(f"[SKIP] patient {pid}: missing {bvp_path}")
            continue

        bg_pid = bg_events_by_patient[pid].reset_index(drop=True)
        sex_val = sex_lookup.get(pid, pd.NA)

        print(f"\n[PATIENT {pid}] BG events={len(bg_pid):,} sex={sex_val} | reading {os.path.basename(bvp_path)}")

        # ---- read that patient's BVP file ----
        bvp_raw = pd.read_csv(bvp_path)

        # detect timestamp + bvp columns robustly
        ts_col, bvp_col = _detect_time_and_bvp_cols(bvp_raw.columns)
        bvp_raw = bvp_raw.rename(columns={ts_col: "timestamp", bvp_col: "bvp"})

        bvp_raw["timestamp"] = pd.to_datetime(bvp_raw["timestamp"], errors="coerce")
        bvp_raw["bvp"] = pd.to_numeric(bvp_raw["bvp"], errors="coerce")
        bvp_raw = bvp_raw.dropna(subset=["timestamp", "bvp"]).sort_values("timestamp").reset_index(drop=True)

        if bvp_raw.empty:
            print(f"[SKIP] patient {pid}: BVP file has no valid data after parsing")
            continue

        # arrays for fast slicing
        ts = bvp_raw["timestamp"].to_numpy(dtype="datetime64[ns]")
        bvp = bvp_raw["bvp"].to_numpy(dtype=np.float32)
        ts_ns = ts.astype("int64")

        ts_min, ts_max = ts_ns[0], ts_ns[-1]
        print(f"[PATIENT {pid}] BVP rows={len(bvp_raw):,} range={pd.to_datetime(ts_min)} -> {pd.to_datetime(ts_max)}")
        print(f"[PATIENT {pid}] detected cols: timestamp='{ts_col}', bvp='{bvp_col}'")

        buffer_rows = []
        buffer_events = 0

        for _, ev in bg_pid.iterrows():
            g_t_ns = np.datetime64(ev["timestamp"], "ns").astype("int64")
            g_val = float(ev["glucose_mg_dl"])

            span_start = g_t_ns - half_ns
            span_end   = g_t_ns + half_ns

            # Require full +/- span
            if span_start < ts_min or span_end > ts_max:
                continue

            event_rows_before = len(buffer_rows)

            # 10 windows of 30s across [t-2.5min, t+2.5min]
            for w_idx in range(n_windows):
                w_start = span_start + w_idx * win_ns
                w_end   = w_start + win_ns

                left  = np.searchsorted(ts_ns, w_start, side="left")
                right = np.searchsorted(ts_ns, w_end,   side="left")
                if right <= left:
                    continue

                seg = bvp[left:right]
                seg = seg[np.isfinite(seg)]
                if seg.size < min_samples:
                    continue

                try:
                    seg_f = butter_bandpass_filter(seg.astype(float), lowcut, highcut, fs, order=order)
                except Exception:
                    continue

                feats = extract_ppg_features(seg_f, fs)
                if all(v == 0 for v in feats.values()):
                    continue

                buffer_rows.append({
                    "patient_id": pid,
                    "sex": int(sex_val) if pd.notna(sex_val) else pd.NA,
                    "glucose_mg_dl": g_val,
                    **feats
                })

            if len(buffer_rows) > event_rows_before:
                buffer_events += 1

            if buffer_events >= batch_events:
                batch_df = pd.DataFrame(buffer_rows)
                batch_df = batch_df.drop(columns=[c for c in DROP_COLS if c in batch_df.columns], errors="ignore")
                batch_df.to_csv(out_csv_path, mode="a", header=header, index=False)
                header = False

                buffer_rows.clear()
                buffer_events = 0

        if buffer_rows:
            batch_df = pd.DataFrame(buffer_rows)
            batch_df = batch_df.drop(columns=[c for c in DROP_COLS if c in batch_df.columns], errors="ignore")
            batch_df.to_csv(out_csv_path, mode="a", header=header, index=False)
            header = False
            print(f"[WRITE FINAL] patient={pid} wrote_rows={len(batch_df):,}")

    print("\n[DONE]")


# ------------------------- EXAMPLE USAGE -------------------------

# BVP_DIR = os.path.join(PROJECT_ROOT, "PhysionetData")

# write_features_from_per_patient_bvp_files(
#     bg_csv_path=BG_ALL_PATH,
#     bvp_dir=BVP_DIR,
#     out_csv_path=OUTPUT_FILE,
#     patient_ids=range(1, 17),
#     fs=SAMPLE_RATE_HZ,
#     win_seconds=30,
#     half_window_minutes=2.5,
#     min_coverage=0.90,
#     lowcut=0.5,
#     highcut=8.0,
#     order=3,
#     batch_events=50
# )

