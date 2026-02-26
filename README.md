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
- Trains a dense MLP with batch normalization + dropout.
- Predicts log-transformed glucose target and inverts to mg/dL.
- Reports `R^2`, MAE, MAPE on held-out test patients.
- Generates Clarke Error Grid plot via `helper_scripts/cega.py`.

Important note: `mlp.py` currently contains an explicit `exit()` right after CEGA plotting, so the quantization/export block below that line will not run unless you remove/comment out the `exit()` call.

## Dependencies

Install all project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes core dependencies for the main pipeline (`mlp.py`) plus legacy/experimental dependencies used in `old_models/`.

## Notes and caveats

- Several scripts are research-stage and may require path/column updates before running on a fresh machine.
- `data_processing_scripts/mendeleyDataProcess.py` is marked unfinished.
- `MLP_Average.py` imports `cega` as `from cega import cega`; if needed, change to `from helper_scripts.cega import cega` to match this repo layout.
- Some model artifacts in `model_weights/` were generated by older experiments in `old_models/`.

## Recommended workflow for this repo

1. Use or regenerate `processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv`.
2. Train/evaluate with `mlp.py`.
3. Inspect CEGA and standard regression metrics.
4. If deployment is needed, remove `exit()` in `mlp.py` and run quantization/export.
