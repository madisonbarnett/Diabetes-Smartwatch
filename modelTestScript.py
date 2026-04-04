import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from pathlib import Path

def main():
    MODELPATH = './model_weights/mlp.keras'
    SCALERPATH = './model_weights/mlp_scalers.pkl'
    TESTPATH = 'test_set.csv'
    OUTPUT_CSV = 'test_predictions.csv'   # <-- New: where results will be saved

    # Load model and scaler
    print("Loading model and scaler...")
    model = tf.keras.models.load_model(MODELPATH)
    scaler = joblib.load(SCALERPATH)

    # Load test set efficiently
    print("Loading test set...")
    start_time = time.time()
    
    # Read only needed columns with proper dtypes for speed
    df_test = pd.read_csv(TESTPATH, dtype='float32')
    
    load_time = time.time() - start_time
    print(f"Test set loaded: {len(df_test)} rows in {load_time:.2f} seconds")

    # Define columns
    id_col = 'caseid'
    target_col = 'gluc'
    features = [col for col in df_test.columns if col not in [id_col, target_col]]

    # Prepare data
    print("Preparing data...")
    X_test = df_test[features].values.astype(np.float32)
    y_actual = df_test[target_col].values.astype(np.float32)
    case_ids = df_test[id_col].values   # Keep original case IDs

    # Apply scaling
    if scaler is not None:
        print("Applying scaler to test set...")
        X_test = scaler.transform(X_test)

    # Make predictions on the entire set at once
    print("Running predictions...")
    start_pred = time.time()
    y_pred = model.predict(X_test, batch_size=2048, verbose=1).flatten()
    pred_time = time.time() - start_pred
    print(f"Predictions done in {pred_time:.2f} seconds")

    # Calculate overall metrics
    r2_test = r2_score(y_actual, y_pred)
    mae_test = np.mean(np.abs(y_actual - y_pred))
    mape_test = mean_absolute_percentage_error(y_actual, y_pred) * 100

    # Create results DataFrame and save to CSV
    print(f"Saving predictions to '{OUTPUT_CSV}'...")
    results_df = pd.DataFrame({
        'caseid': case_ids,
        'actual_gluc': y_actual,
        'predicted_gluc': y_pred,
        'absolute_error': np.abs(y_actual - y_pred)
    })
    
    # Optional: round for nicer reading in Excel
    results_df['predicted_gluc'] = results_df['predicted_gluc'].round(4)
    results_df['absolute_error'] = results_df['absolute_error'].round(4)
    
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved successfully! ({len(results_df)} rows)")

    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"MLP results on TEST set ({len(df_test)} samples):")
    print(f"    R²          : {r2_test:.3f}")
    print(f"    MAE         : {mae_test:.2f} mg/dL")
    print(f"    MAPE        : {mape_test:.2f}%")
    print(f"    Total time  : {total_time:.1f} seconds")
    print("="*70)
    print(f"Output file: {OUTPUT_CSV}")
    print("You can now open this CSV to compare actual vs predicted values row by row.")


if __name__ == "__main__":
    main()