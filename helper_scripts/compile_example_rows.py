from pathlib import Path

import pandas as pd

SOURCE_CSV = Path("processed_data/new_vitaldb_ppg_extracted_features_15s_5minwin.csv")
OUTPUT_CSV = Path("vitaldb_20_example_rows.csv")
CASE_ID_COLUMN = "caseid"
FEATURE_COLUMNS = [
    "gluc",
    "age",
    "weight",
    "height",
    "preop_dm",
    "ppg_mean_pp_interval_s",
    "ppg_std",
    "ppg_teager_energy",
    "ppg_skew",
    "ppg_iqr",
    "ppg_entropy",
    "ppg_first_deriv_max",
    "ppg_std_pp_interval_s",
]
NUM_EXAMPLES = 20


def main() -> None:
    required_columns = [CASE_ID_COLUMN, *FEATURE_COLUMNS]
    df = pd.read_csv(SOURCE_CSV, usecols=required_columns)

    unique_case_rows = df.drop_duplicates(subset=CASE_ID_COLUMN, keep="first")

    if len(unique_case_rows) < NUM_EXAMPLES:
        raise ValueError(
            f"Only found {len(unique_case_rows)} unique caseids in {SOURCE_CSV}, "
            f"but {NUM_EXAMPLES} examples were requested."
        )

    selected_rows = unique_case_rows.head(NUM_EXAMPLES).copy()
    selected_caseids = selected_rows[CASE_ID_COLUMN].tolist()

    output_df = selected_rows[FEATURE_COLUMNS]
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(output_df)} example rows to {OUTPUT_CSV}")
    print(f"Selected caseids: {selected_caseids}")
    print(f"Output columns: {FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
