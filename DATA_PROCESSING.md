# Our Data Preprocessing

A more detailed look into our data preprocessing pipeline, how raw VitalDB records are filtered and matched, and how final model-ready features are generated.

## An important note

Any machine learning pipeline is only as reliable as the quality of the input data. Our preprocessing script is designed to isolate glucose measurements with valid timing, match them to photoplethysmography (PPG) waveforms, and convert those raw signals into a structured feature table for model training.

The process described here is performed by `data_processing_scripts/vitalGlucProcessData.py`.

## Data Sources

The preprocessing pipeline uses two VitalDB tables:
- `https://api.vitaldb.net/labs`: contains laboratory measurements, including continuous-time blood glucose values
- `https://api.vitaldb.net/cases`: contains case-level metadata such as age, sex, case duration, weight, height, BMI, and diabetes history

In addition to the tabular metadata, the script loads raw PPG waveform data directly from each case using the VitalDB API. The waveform channel used is `SNUADC/PLETH`.

## Initial Filtering

The script first narrows the dataset to the records that are useful for blood glucose estimation.

Glucose records are selected by filtering the labs table to rows where `name == 'gluc'`. Cases are then filtered to only those with demographic information present, specifically cases where `age` is not missing. This ensures each final sample includes both physiological features and basic patient metadata.

For each case, glucose measurements are also filtered by time so that only readings occurring during the recorded procedure are retained. Any glucose value with a timestamp outside the interval from case start through `caseend` is removed.

## Signal Matching

Each valid glucose reading is paired with a surrounding section of PPG signal. The script uses:
- A sampling rate of 500 Hz
- A 10-second feature window
- A valid matching range of +/- 5 minutes around each glucose measurement

For a given glucose timestamp, the script extracts the available PPG signal within that 10-minute span. If there is not enough signal to contain at least one full 10-second window, that glucose sample is skipped.

This approach allows each glucose measurement to be associated with multiple short PPG segments from the surrounding period rather than relying on a single raw snapshot.

## Signal Filtering

Before features are extracted, each matched PPG segment is passed through a Butterworth band-pass filter with cutoffs at 0.5 Hz and 8 Hz.

This filtering step removes low-frequency drift and high-frequency noise while preserving the pulsatile content of the waveform that is most relevant to cardiovascular dynamics.

## Windowing

After filtering, the matched PPG segment is divided into consecutive non-overlapping 10-second windows.

At 500 Hz, each window contains 5000 samples. Every full window is processed independently. Windows that are too short or contain unusable signal values are discarded.

## Feature Extraction

Each 10-second PPG window is transformed into a hand-engineered feature vector. The script extracts both general waveform statistics and pulse wave morphology features.

General PPG features include:
- Mean and standard deviation
- Mean and standard deviation of peak-to-peak interval
- Estimated pulse frequency
- Area under the curve
- Maximum and minimum first derivative
- Histogram entropy
- Teager-Kaiser energy
- Log energy
- Skewness
- Interquartile range
- Spectral entropy

Pulse wave morphology features include:
- Rise time
- Decay time
- Pulse width
- Pulse amplitude
- Maximum slope
- Minimum slope

Peak-based features are computed using detected waveform peaks. If a window does not contain enough valid peaks or the signal is otherwise unusable, the script returns zero-valued placeholders and later excludes all-zero feature rows from the final dataset.

## Demographic Features

Each retained PPG window is augmented with case-level metadata from VitalDB. These values include:
- `caseid`
- `gluc`
- `age`
- `sex`
- `preop_dm`
- `weight`
- `height`
- `bmi`

The `sex` field is encoded numerically, with female represented as `1` and all other values represented as `0`.

## Delta Feature Engineering

After all windows tied to a given glucose reading are assembled into a case-level DataFrame, the script creates within-patient delta features.

Delta features are computed for the main statistical PPG features by taking the difference between each row and the previous row in time order. This captures short-term changes in waveform behavior rather than only absolute signal properties.

The script creates delta versions of:
- `ppg_mean`
- `ppg_std`
- `ppg_mean_pp_interval_s`
- `ppg_std_pp_interval_s`
- `ppg_freq`
- `ppg_auc`
- `ppg_first_deriv_max`
- `ppg_first_deriv_min`
- `ppg_entropy`
- `ppg_teager_energy`
- `ppg_log_energy`
- `ppg_skew`
- `ppg_iqr`
- `ppg_spectral_entropy`

Because the first row in a sequence has no prior row for comparison, its delta values are filled with `0`.

## Output Dataset

The final processed rows are written to:

```text
delta_vitaldb_ppg_extracted_features_15s_5minwin.csv
```

To improve write speed, processed data is buffered and saved in batches of 50 cases at a time. If the output file already exists, new results are appended without rewriting the header.

Each row in the final CSV represents one valid 10-second PPG window associated with a glucose measurement, enriched with demographic features and within-patient delta features.

## Why This Pipeline Matters

Raw clinical waveform data is noisy, irregular, and difficult to use directly in a predictive model. This preprocessing pipeline converts that raw information into a structured tabular dataset that:
- Aligns glucose values with nearby physiological signal
- Removes invalid or poorly matched samples
- Standardizes the signal through filtering and fixed-length windows
- Encodes waveform behavior through interpretable engineered features
- Preserves patient context through demographic variables
- Captures short-term physiological change through delta features

This processed dataset is the direct input to the model training pipeline described in `MODEL.md`.
