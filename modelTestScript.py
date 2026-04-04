import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import re
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from pathlib import Path
from helper_scripts.cega import cega

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


def run_c_header_inference(header_path, X_test, scaler=None):
    """
    Test the final C header file (mlp.h) - exactly what will run on your microcontroller.
    Robust parsing that handles xxd -i output properly.
    """
    print(f"Loading model from C header: {header_path}")
    
    # Read the entire .h file
    with open(header_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Robust extraction of the byte array using regex (handles newlines, spaces, and comments)
    # Looks for the pattern: = { ... };
    match = re.search(r'=\s*\{(.*?)\}\s*;', content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find the byte array in {header_path}. "
                        "Make sure the file contains 'unsigned char ...[] = {'")

    byte_str = match.group(1)

    # Clean and extract all hex values (0xNN)
    hex_values = re.findall(r'0x[0-9a-fA-F]+', byte_str)
    
    if not hex_values:
        raise ValueError("No hex values (0x..) found in the C header file.")

    print(f"Successfully extracted {len(hex_values):,} bytes from mlp.h")

    # Convert hex strings to uint8 array
    model_bytes = np.array([int(h, 16) for h in hex_values], dtype=np.uint8)

    print(f"Model byte array created: {len(model_bytes):,} bytes")

    # Now create the TFLite interpreter from the bytes (same as .tflite)
    interpreter = tf.lite.Interpreter(model_content=model_bytes.tobytes())
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    is_input_int8 = input_details['dtype'] == np.int8
    is_output_int8 = output_details['dtype'] == np.int8

    predictions = []
    print("Running inference from C header (int8)...")
    start = time.time()

    for i in range(len(X_test)):
        x = X_test[i:i+1].copy()   # shape (1, num_features)

        if scaler is not None:
            x = scaler.transform(x)

        # Quantize input
        if is_input_int8:
            x = (x / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details['index'], x)
        interpreter.invoke()

        # Get raw output
        y = interpreter.get_tensor(output_details['index'])[0]   # should be scalar

        # Dequantize output to float
        if is_output_int8:
            y = (y.astype(np.float32) - output_zero_point) * output_scale
        else:
            y = y.astype(np.float32)

        predictions.append(y[0] if isinstance(y, np.ndarray) and y.ndim > 0 else float(y))

    c_time = time.time() - start
    print(f"C header inference finished in {c_time:.2f} seconds ({c_time/len(X_test)*1000:.2f} ms per sample)")

    return np.array(predictions).flatten()


def main():
    # ================== CONFIG ==================
    MODELPATH = './model_weights/mlp.keras'
    SCALERPATH = './model_weights/mlp_scalers.pkl'
    TESTPATH = 'test_set.csv'
    
    TFLITE_PATH = './model_weights/mlp_int8.tflite'
    C_HEADER_PATH = './model_weights/mlp.h'          # <-- Your C header file

    # ===========================================

    # Load scaler and test data
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

    # === 3. Test with final C header (mlp.h) ===
    print("\n=== Testing final C header (mlp.h) ===")
    y_pred_c = run_c_header_inference(C_HEADER_PATH, X_test, scaler)

    # === Save all three results for easy comparison ===
    results = pd.DataFrame({
        'caseid': case_ids,
        'actual_gluc': y_actual,
        'predicted_keras': np.round(y_pred_keras, 4),
        'predicted_tflite_int8': np.round(y_pred_tflite, 4),
        'predicted_c_header': np.round(y_pred_c, 4),
        'error_keras': np.abs(y_actual - y_pred_keras),
        'error_tflite': np.abs(y_actual - y_pred_tflite),
        'error_c_header': np.abs(y_actual - y_pred_c)
    })

    results.to_csv('test_predictions_comparison.csv', index=False)
    print(f"\n✅ Full comparison saved to 'test_predictions_comparison.csv'")

    # === Metrics for all three models ===
    def print_metrics(name, y_pred):
        r2 = r2_score(y_actual, y_pred)
        mae = np.mean(np.abs(y_actual - y_pred))
        mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
        print(f"\n{name} Results:")
        print(f"    R²   : {r2:.3f}")
        print(f"    MAE  : {mae:.2f} mg/dL")
        print(f"    MAPE : {mape:.2f}%")
        cega(y_actual, y_pred)

    print_metrics("Keras (float32)", y_pred_keras)
    print_metrics("TFLite (int8)", y_pred_tflite)
    print_metrics("C Header (mlp.h)", y_pred_c)

    print("\n" + "="*70)
    print("You can now open 'test_predictions_comparison.csv'")
    print("to compare actual vs predicted values across all three versions.")
    print("="*70)


if __name__ == "__main__":
    main()