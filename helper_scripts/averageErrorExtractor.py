from pathlib import Path

import numpy as np
import pandas as pd

TESTPATH = Path("test_predictions_comparison.csv")
ACTUAL_COLUMN = "actual_gluc"
PERCENT_ERROR_SPECS = {
    "percent_error_keras": "predicted_keras",
    "percent_error_tflite": "predicted_tflite_int8",
    "percent_error_c_header": "predicted_c_header",
}
MEAN_COLUMNS = [
    "error_keras",
    "error_tflite",
    "error_c_header",
    "percent_error_keras",
    "percent_error_tflite",
    "percent_error_c_header",
]


def main() -> None:
    df_test = pd.read_csv(TESTPATH)
    print("Test data loaded")

    actual_values = df_test[ACTUAL_COLUMN]
    safe_denominator = actual_values.replace(0, np.nan)

    for percent_error_column, prediction_column in PERCENT_ERROR_SPECS.items():
        df_test[percent_error_column] = round((
            (df_test[prediction_column] - actual_values).abs() / safe_denominator
        ) * 100 , 5)

    df_test.to_csv(TESTPATH, index=False)
    print(f"Updated {TESTPATH} with percent error columns")

    for column in MEAN_COLUMNS:
        print(f"Average {column}: {df_test[column].mean()}")


if __name__ == "__main__":
    main()
