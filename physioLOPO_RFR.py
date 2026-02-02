# lopo_eval.py
# Compare Random Split vs LOPO (Leave-One-Patient-Out) for RF regression
# Prints clear terminal labels + sets plot titles before each CEGA display.
#
# Outputs saved to: ./lopo_outputs/
#   - random_split_predictions.csv
#   - lopo_global_predictions.csv
#   - lopo_per_patient_metrics.csv
#   - comparison_summary.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from cega import cega, cega_download

# =========================
# CONFIG
# =========================
CSV_PATH = "processed_data/physioNet_ppg_extracted_features_30s.csv"
OUT_DIR = "lopo_outputs_top_6_features"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL = "glucose_mg_dl"
GROUP_COL = "patient_id"
DROP_COLS = []  # dropped if present, used for redundant features (i.e. "ppg_freq" / "ppg_mean_pp_interval_s"). Empty for testing.

ALL_FEATURES = [
    'sex', 'ppg_mean', 'ppg_std', 'ppg_mean_pp_interval_s', 'ppg_std_pp_interval_s',
    'ppg_auc', 'ppg_first_deriv_max', 'ppg_first_deriv_min', 'ppg_entropy',
    'ppg_teager_energy', 'ppg_log_energy', 'ppg_skew', 'ppg_iqr', 'ppg_spectral_entropy'
]
IMPORTANT_FEATURES = ['sex', 'ppg_iqr', 'ppg_skew', 'ppg_teager_energy', 'ppg_mean_pp_interval_s', 'ppg_std']
RIDGE_REGRESSION_IMPORTANT_FEATURES = ['sex', 'ppg_iqr', 'ppg_first_deriv_max', 'ppg_spectral_entropy', 'ppg_freq', 'ppg_std_pp_interval_s']

USE_IMPORTANT_FEATURES = True  # True -> IMPORTANT_FEATURES, False -> ALL_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42

# LOPO controls
MIN_SAMPLES_PER_PATIENT = 1     # raise (e.g., 50) to skip tiny held-out sets
RUN_CEGA_PER_PATIENT = True     # show a CEGA plot for each held-out patient
MAX_PATIENT_CEGA_PLOTS = None   # e.g., 5 to cap plots; None for all


# =========================
# Helpers
# =========================
def make_rf(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        warm_start=False
    )


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae_mg_dl": float(mean_absolute_error(y_true, y_pred)),
        "mape_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
    }


# =========================
# Random Split Evaluation
# =========================
def eval_random_split(df: pd.DataFrame, features: list[str]) -> dict:
    X = df[features]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    model = make_rf(RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    print("\n===========================================")
    print("RANDOM SPLIT (shuffle=True) EVALUATION")
    print("===========================================")
    print(f"n_train={len(y_train)} | n_test={len(y_test)}")
    print(f"R2={metrics['r2']:.3f} | MAE={metrics['mae_mg_dl']:.2f} mg/dL | MAPE={metrics['mape_%']:.2f}%")

    # Save predictions
    pred_df = pd.DataFrame({"actual": y_test.to_numpy(), "pred": y_pred})
    pred_path = os.path.join(OUT_DIR, "random_split_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved: {pred_path}")
    cega_path = os.path.join(OUT_DIR, "RandomSplit_CEGA.png")

    # CEGA plot (download png)
    cega_download(y_test, y_pred, file_path=cega_path, show_plot=True)

    return {
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
        "metrics": metrics
    }


# =========================
# LOPO Evaluation
# =========================
def eval_lopo(df: pd.DataFrame, features: list[str]) -> dict:
    patient_ids = sorted(df[GROUP_COL].unique())

    per_patient_rows = []
    all_y_true = []
    all_y_pred = []

    print("\n===========================================")
    print("LOPO (LEAVE-ONE-PATIENT-OUT) EVALUATION")
    print("===========================================")
    print(f"Patients found: {len(patient_ids)}")

    plotted = 0

    for pid in patient_ids:
        test_mask = (df[GROUP_COL] == pid)
        n_test = int(test_mask.sum())
        if n_test < MIN_SAMPLES_PER_PATIENT:
            print(f"[SKIP] patient_id={pid} (n={n_test} < {MIN_SAMPLES_PER_PATIENT})")
            continue

        train_df = df.loc[~test_mask]
        test_df = df.loc[test_mask]

        X_train = train_df[features]
        y_train = train_df[TARGET_COL]
        X_test = test_df[features]
        y_test = test_df[TARGET_COL]

        model = make_rf(RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = compute_metrics(y_test, y_pred)
        per_patient_rows.append({
            "patient_id": pid,
            "n_test": n_test,
            **m
        })

        print(f"[LOPO] patient={pid} | n={n_test} | R2={m['r2']:.3f} | MAE={m['mae_mg_dl']:.2f} | MAPE={m['mape_%']:.2f}%")

        file_name = f"Patient_{pid}_CEGA.png"
        file_path = os.path.join(OUT_DIR, file_name)

        # Per-patient CEGA
        if RUN_CEGA_PER_PATIENT:
            if MAX_PATIENT_CEGA_PLOTS is None or plotted < MAX_PATIENT_CEGA_PLOTS:
                cega_download(y_test, y_pred, file_path=file_path, show_plot=True)
                plotted += 1

        # Global aggregation
        all_y_true.append(y_test.to_numpy())
        all_y_pred.append(y_pred)

    results_df = pd.DataFrame(per_patient_rows).sort_values("patient_id").reset_index(drop=True)
    metrics_path = os.path.join(OUT_DIR, "lopo_per_patient_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved: {metrics_path}")

    # Aggregate predictions across all held-out patients
    if not all_y_true:
        raise RuntimeError("LOPO produced no predictions. Check GROUP_COL/patient_id and MIN_SAMPLES_PER_PATIENT.")

    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    # Global (micro-average) metrics
    global_metrics = compute_metrics(y_true_all, y_pred_all)

    # Macro-average (patient-balanced)
    macro = {
        "macro_r2_mean": float(results_df["r2"].mean()) if len(results_df) else float("nan"),
        "macro_r2_std": float(results_df["r2"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "macro_mae_mean": float(results_df["mae_mg_dl"].mean()) if len(results_df) else float("nan"),
        "macro_mae_std": float(results_df["mae_mg_dl"].std(ddof=1)) if len(results_df) > 1 else 0.0,
        "macro_mape_mean": float(results_df["mape_%"].mean()) if len(results_df) else float("nan"),
        "macro_mape_std": float(results_df["mape_%"].std(ddof=1)) if len(results_df) > 1 else 0.0,
    }

    print("\n--- LOPO Global (micro-average across all held-out samples) ---")
    print(f"n_total_test={len(y_true_all)}")
    print(f"R2={global_metrics['r2']:.3f} | MAE={global_metrics['mae_mg_dl']:.2f} mg/dL | MAPE={global_metrics['mape_%']:.2f}%")

    print("\n--- LOPO Macro-average (each patient equal weight) ---")
    print(f"R2={macro['macro_r2_mean']:.3f} ± {macro['macro_r2_std']:.3f}")
    print(f"MAE={macro['macro_mae_mean']:.2f} ± {macro['macro_mae_std']:.2f} mg/dL")
    print(f"MAPE={macro['macro_mape_mean']:.2f} ± {macro['macro_mape_std']:.2f}%")

    # Save global predictions
    pred_df = pd.DataFrame({"actual": y_true_all, "pred": y_pred_all})
    pred_path = os.path.join(OUT_DIR, "lopo_global_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved: {pred_path}")

    cega_path = os.path.join(OUT_DIR, "All_Patients_CEGA.png")

    # Global CEGA plot
    cega_download(y_true_all, y_pred_all, file_path=cega_path, show_plot=True)

    return {
        "per_patient_df": results_df,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "global_metrics": global_metrics,
        "macro_metrics": macro
    }


# =========================
# Comparison Summary
# =========================
def save_comparison_summary(random_eval: dict, lopo_eval: dict):
    summary = {
        "random_r2": random_eval["metrics"]["r2"],
        "random_mae_mg_dl": random_eval["metrics"]["mae_mg_dl"],
        "random_mape_%": random_eval["metrics"]["mape_%"],
        "lopo_global_r2": lopo_eval["global_metrics"]["r2"],
        "lopo_global_mae_mg_dl": lopo_eval["global_metrics"]["mae_mg_dl"],
        "lopo_global_mape_%": lopo_eval["global_metrics"]["mape_%"],
        "lopo_macro_r2_mean": lopo_eval["macro_metrics"]["macro_r2_mean"],
        "lopo_macro_r2_std": lopo_eval["macro_metrics"]["macro_r2_std"],
        "lopo_macro_mae_mean": lopo_eval["macro_metrics"]["macro_mae_mean"],
        "lopo_macro_mae_std": lopo_eval["macro_metrics"]["macro_mae_std"],
        "lopo_macro_mape_mean": lopo_eval["macro_metrics"]["macro_mape_mean"],
        "lopo_macro_mape_std": lopo_eval["macro_metrics"]["macro_mape_std"],
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(OUT_DIR, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    print("\n===========================================")
    print("COMPARISON SUMMARY (Random Split vs LOPO)")
    print("===========================================")
    print(f"Random Split  | R2={summary['random_r2']:.3f} | MAE={summary['random_mae_mg_dl']:.2f} | MAPE={summary['random_mape_%']:.2f}%")
    print(f"LOPO (global) | R2={summary['lopo_global_r2']:.3f} | MAE={summary['lopo_global_mae_mg_dl']:.2f} | MAPE={summary['lopo_global_mape_%']:.2f}%")
    print(f"LOPO (macro)  | R2={summary['lopo_macro_r2_mean']:.3f} ± {summary['lopo_macro_r2_std']:.3f}")


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(CSV_PATH)

    # Drop cols safely
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Choose feature set
    features = IMPORTANT_FEATURES if USE_IMPORTANT_FEATURES else ALL_FEATURES

    # Validate required columns
    required = set(features + [TARGET_COL, GROUP_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Clear any old figures
    plt.close("all")

    # 1) Random split eval + CEGA
    random_eval = eval_random_split(df, features)

    # 2) LOPO eval:
    #    - per-patient metrics + per-patient CEGA (optional)
    #    - global LOPO CEGA across all held-out predictions
    lopo_eval = eval_lopo(df, features)

    # 3) Comparison summary CSV + terminal print
    save_comparison_summary(random_eval, lopo_eval)


if __name__ == "__main__":
    main()
