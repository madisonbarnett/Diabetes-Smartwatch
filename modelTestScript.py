import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def test_model(model, df_test, row_index, features, scaler, id_col, target_col):                 
    row_df = df_test.iloc[[row_index]]
    print(f"Testing row index: {row_index}")
    print(f"Case ID: {row_df[id_col].iloc[0]}")
    print(f"Actual {target_col}: {row_df[target_col].iloc[0]}")

    # Select only the features (in correct order!)
    exclude = [id_col, target_col]
    features = [col for col in df_test.columns if col not in exclude]
    
    X_row = row_df[features].values.astype(np.float32)
    actual = row_df[target_col].iloc[0]

    # Apply scaling if you used a scaler during training
    if scaler is not None:
        X_row = scaler.transform(X_row)  
        print("Applied scaling to the row")

    y_pred = model.predict(X_row, verbose=0).flatten()[0]
    error = abs(y_pred - actual)
    
    print(f"Predicted {target_col.upper():<6}: {y_pred:.4f}")
    print(f"Abs Error         : {error:.4f}")
    print(f"{'='*65}")
    
    return y_pred, actual, error


def main():
    MODELPATH = './model_weights/mlp.keras'
    SCALERPATH = './model_weights/mlp_scalers.pkl'
    TESTPATH = 'test_set.csv'

    # Load model
    model = tf.keras.models.load_model(MODELPATH)
    scaler = joblib.load(SCALERPATH)
    
    # Load the full test set
    df_test = pd.read_csv(TESTPATH)

    # 29322
    for i in range(1, 3):
        y_pred, actual, error = test_model(model, df_test, i, df_test.columns, scaler, 'caseid', 'gluc')

if __name__ == "__main__":
    main()
