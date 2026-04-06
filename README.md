# Diabetes-Smartwatch

Machine learning pipeline for non-invasive blood glucose estimation from photoplethysmography (PPG), focused on deployment-ready models for a smartwatch capstone project under the University of Alabama ECE Department.

## Current best implementation

The current best-performing setup in this repository is:

- Model script: `mlp.py`
- Dataset: `processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv`
- Data definition: VitalDB continuous glucose values + timestamps matched with continuous PPG context from a **5-minute window** around each glucose value (`+/-2.5 min`), then converted into extracted PPG features from **15-second windows** inside that 5-minute segment.

In `mlp.py`, this is configured by default:

- `DATASET = 'vitaldb'`
- `SUFFIX = '15s'`
- `DATAFILE = 'processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv'`

## Repository structure

- `mlp.py`: Primary MLP training/evaluation script (group-aware split by patient/case, CEGA evaluation, optional quantization/export).
- `MLP_Average.py`: Runs multiple MLP trials and reports average metrics.
- `complexMLP.py`: Additional dense neural-network experiment.
- `data_processing_scripts/`: Dataset creation and feature-engineering scripts for VitalDB/PhysioNet workflows.
- `helper_scripts/`: Utilities for CEGA plotting, feature analysis, model quantization, and C conversion.
- `processed_data/`: Generated feature datasets used for model training.
- `raw_data/`: Source datasets and raw files (VitalDB/PhysioNet/Mendeley).
- `model_weights/`: Saved model artifacts (`.keras`, `.tflite`, `.pkl`, `.h`, `.c`).
- `results/`: Stored evaluation outputs (including LOPO and CEGA plots from RFR baselines).
- `old_models/`: Archived model experiments (RF, XGBoost, DNN/LSTM variants, validation scripts, notebooks).

Important note: When attempting to run some old models import directories may need to be updated to correctly access datasets and custom helper scripts.

## End-to-end data pipeline (high level)

1. Pull or parse raw physiological data (`raw_data/` or VitalDB API).
2. Align glucose targets to signal windows.
3. Filter PPG and extract engineered features per window.
4. Write feature table CSV to `processed_data/`.
5. Train/evaluate models (`mlp.py`, `MLP_Average.py`, or older baselines).
6. Optionally quantize/export models for embedded deployment.

## Key data-processing scripts

- `data_processing_scripts/vitalGlucProcessData.py`:
  - Builds VitalDB glucose regression dataset using glucose lab values (`gluc`) and PPG features.
  - Uses 500 Hz PPG, `+/-2.5 min` around each glucose timestamp.
  - Splits each 5-minute region into 15-second feature windows.
  - Outputs `new_vitaldb_ppg_extracted_features_15s_5minwin.csv`.

- `data_processing_scripts/physioNetDataProcess.py`:
  - Builds PhysioNet PPG feature dataset.
  - Uses patient BG timestamps and per-patient BVP files.
  - Produces `processed_data/physioNet_ppg_extracted_features_30s.csv`.

- `data_processing_scripts/processData.py` / `processDataPPG.py` / `OldPPGProcess.py`:
  - Earlier VitalDB feature extraction pipelines (with and without ECG/BP).

- `data_processing_scripts/vitalDB_script1.py`:
  - Filters VitalDB glucose labs by valid case timing and writes summary CSVs.

- `data_processing_scripts/aggregateData.py`:
  - Aggregates window-level features into case-level summary features.

- `data_processing_scripts/ppgWave.py`:
  - Diagnostic plotting of raw vs filtered VitalDB PPG waveform windows.

## Key helper scripts

- `helper_scripts/cega.py`: Clarke Error Grid Analysis (zone stats + plot/save utilities).
- `helper_scripts/featureAnalysis.py`: Feature correlation and glucose distribution analysis on VitalDB feature tables.
- `helper_scripts/physioFeatureAnalysis.py`: Similar feature analysis for PhysioNet table.
- `helper_scripts/compile_example_rows.py`: Builds a 20-row example CSV from `processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv` using random input rows from 20 unique `caseid` patients.
- `helper_scripts/modelExampleEntriesTest.py`: Runs float32, int8 TFLite, and C-header inference on `vitaldb_20_example_rows.csv` and compares predictions to the row-level `gluc` values.
- `helper_scripts/averageErrorExtractor.py`: Adds percent-error columns to a model comparison CSV and prints the mean absolute and mean percent errors for each inference path.
- `helper_scripts/quantizeModel.py`: Converts a saved Keras model to quantized TFLite.
- `helper_scripts/convertToC.py`: Converts a trained sklearn model (`.pkl`) to embedded C/header via `emlearn`.

## Training the best model (`mlp.py`)

From repository root:

```bash
python mlp.py
```

What `mlp.py` currently does:

- Loads `processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv`.
- Uses group-aware splitting by `caseid` (prevents patient leakage).
- Trains a dense MLP with batch normalization + dropout on 12 most important features.
- Predicts glucose target in mg/dL.
- Reports `R^2`, MAE, MAPE on held-out test patients.
- Generates Clarke Error Grid plot via `helper_scripts/cega.py`.
- Saves `model_weights/mlp.keras` and `model_weights/mlp_scalers.pkl`.
- Quantizes and saves `model_weights/mlp_int8.tflite` (ready for conversion to C).

## Model evaluation scripts

### Full held-out test set comparison

Run from repository root:

```bash
python modelTestScript.py
```

This script loads:

- `model_weights/mlp.keras`
- `model_weights/mlp_scalers.pkl`
- `model_weights/mlp_int8.tflite`
- `model_weights/mlp.h`
- `test_set.csv`

It evaluates all three inference paths:

- Keras float32
- TFLite int8
- C-header / embedded-equivalent int8

It writes `test_predictions_comparison.csv`, which includes:

- `caseid`
- `actual_gluc`
- all three predictions
- `error_keras`
- `error_tflite`
- `error_c_header`

### Build 20 example entries

Run from repository root:

```bash
python helper_scripts/compile_example_rows.py
```

This creates `vitaldb_20_example_rows.csv` by:

- reading `processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv`
- selecting the first available row for each unique `caseid`
- keeping the first 20 unique-case rows
- exporting only:
  - `gluc`
  - `age`
  - `weight`
  - `height`
  - `preop_dm`
  - `ppg_mean_pp_interval_s`
  - `ppg_std`
  - `ppg_teager_energy`
  - `ppg_skew`
  - `ppg_iqr`
  - `ppg_entropy`
  - `ppg_first_deriv_max`
  - `ppg_std_pp_interval_s`

### Test on the 20 example entries

Run from repository root:

```bash
python helper_scripts/modelExampleEntriesTest.py
```

This script compares float32, int8 TFLite, and C-header predictions against the `gluc` value in each row of `vitaldb_20_example_rows.csv`.

It writes `example_entries_predictions_comparison.csv`, which includes:

- `example_row`
- `actual_gluc`
- all three predictions
- `error_keras`
- `error_tflite`
- `error_c_header`

### Summarize average error and percent error

Run from repository root:

```bash
python helper_scripts/averageErrorExtractor.py
```

By default this script reads `test_predictions_comparison.csv`, adds:

- `percent_error_keras`
- `percent_error_tflite`
- `percent_error_c_header`

and then prints the mean of both the absolute-error and percent-error columns. Percent error is computed as:

```text
abs(predicted - actual_gluc) / actual_gluc * 100
```

If you want to summarize `example_entries_predictions_comparison.csv` instead, update `TESTPATH` in `helper_scripts/averageErrorExtractor.py`.

## Dependencies

Install all project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes core dependencies for the main pipeline (`mlp.py`) plus legacy/experimental dependencies used in `old_models/`.

## Git LFS

This repository uses Git Large File Storage (Git LFS) for large artifacts, including processed datasets and model files that are too large for normal Git tracking.

After cloning, make sure Git LFS is installed and pull LFS-managed files:

```bash
git lfs install
git lfs pull
```
## Notes and caveats

- Several scripts are research-stage and may require path/column updates before running on a fresh machine.
- `data_processing_scripts/mendeleyDataProcess.py` is marked unfinished.
- `MLP_Average.py` imports `cega` as `from cega import cega`; if needed, change to `from helper_scripts.cega import cega` to match this repo layout.
- Some model artifacts in `model_weights/` were generated by older experiments in `old_models/`.

## Recommended workflow for this repo

1. Use or regenerate `processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv`.
2. Train/evaluate with `mlp.py`.
3. Run `python modelTestScript.py` to compare the float32, int8, and C-header models on the held-out test set.
4. Optionally run `python helper_scripts/compile_example_rows.py` and `python helper_scripts/modelExampleEntriesTest.py` for a small hand-checkable example set.
5. Run `python helper_scripts/averageErrorExtractor.py` to append percent errors and summarize mean error values.
6. Inspect CEGA and standard regression metrics.
7. Convert .tflite model to C header file using:
```bash
xxd -i {TFLITE_MODEL} > mlp.h
```


