import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from pathlib import Path

def run_keras_inference(model, X_test, scaler=None):
    """Run inference with the original Keras model (float32)."""
    if scaler is not None:
        X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test, batch_size=2048, verbose=0).flatten()
    return y_pred


def run_tflite_inference(tflite_path, X_test, scaler=None):
    """Run inference with an int8 quantized .tflite model.
    Handles input/output quantization automatically."""
    print(f"Loading TFLite model: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Get quantization parameters
    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    is_input_int8 = input_details['dtype'] == np.int8
    is_output_int8 = output_details['dtype'] == np.int8

    predictions = []

    print("Running TFLite inference (int8)...")
    start = time.time()

    for i in range(len(X_test)):
        x = X_test[i:i+1]  # keep 2D shape: (1, num_features)

        if scaler is not None:
            x = scaler.transform(x)

        # Quantize input if the model expects int8
        if is_input_int8:
            x = (x / input_scale + input_zero_point).astype(np.int8)

        # Set input tensor
        interpreter.set_tensor(input_details['index'], x)
        interpreter.invoke()

        # Get output and dequantize if needed
        y = interpreter.get_tensor(output_details['index'])[0]  # scalar output

        if is_output_int8:
            y = (y.astype(np.float32) - output_zero_point) * output_scale

        predictions.append(y)

    tflite_time = time.time() - start
    print(f"TFLite inference finished in {tflite_time:.2f} seconds")

    return np.array(predictions).flatten()


def main():
    # ================== CONFIG ==================
    MODELPATH = './model_weights/mlp.keras'
    SCALERPATH = './model_weights/mlp_scalers.pkl'
    TESTPATH = 'test_set.csv'
    
    TFLITE_PATH = './model_weights/mlp_int8.tflite'   # <-- Change this to your int8 file path

    # OUTPUT_KERAS = 'test_predictions_keras.csv'
    # OUTPUT_TFLITE = 'test_predictions_tflite_int8.csv'
    # ===========================================

    # Load scaler and test data (same as before)
    print("Loading scaler and test set...")
    scaler = joblib.load(SCALERPATH)
    
    df_test = pd.read_csv(TESTPATH, dtype='float32')
    
    id_col = 'caseid'
    target_col = 'gluc'
    features = [col for col in df_test.columns if col not in [id_col, target_col]]

    X_test = df_test[features].values.astype(np.float32)
    y_actual = df_test[target_col].values.astype(np.float32)
    case_ids = df_test[id_col].values

    # === 1. Test with original Keras model ===
    print("\n=== Testing Keras (float32) model ===")
    model = tf.keras.models.load_model(MODELPATH)
    y_pred_keras = run_keras_inference(model, X_test, scaler)

    # === 2. Test with int8 TFLite model ===
    print("\n=== Testing int8 TFLite model ===")
    y_pred_tflite = run_tflite_inference(TFLITE_PATH, X_test, scaler)

    # === Save both results for easy comparison ===
    results = pd.DataFrame({
        'caseid': case_ids,
        'actual_gluc': y_actual,
        'predicted_keras': np.round(y_pred_keras, 4),
        'predicted_tflite_int8': np.round(y_pred_tflite, 4),
        'error_keras': np.abs(y_actual - y_pred_keras),
        'error_tflite': np.abs(y_actual - y_pred_tflite)
    })

    results.to_csv('test_predictions_comparison.csv', index=False)
    print(f"\n✅ Comparison saved to 'test_predictions_comparison.csv'")

    # === Metrics ===
    def print_metrics(name, y_pred):
        r2 = r2_score(y_actual, y_pred)
        mae = np.mean(np.abs(y_actual - y_pred))
        mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
        print(f"\n{name} Results:")
        print(f"    R²   : {r2:.3f}")
        print(f"    MAE  : {mae:.2f} mg/dL")
        print(f"    MAPE : {mape:.2f}%")

    print_metrics("Keras (float32)", y_pred_keras)
    print_metrics("TFLite (int8)", y_pred_tflite)

    print("\nYou can now open 'test_predictions_comparison.csv' to see side-by-side actual vs both predictions.")


if __name__ == "__main__":
    main()