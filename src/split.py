"""
split.py — Data Splitting Module

Loads the raw UCI Cervical Cancer Risk Factors dataset, validates its schema,
separates features and target, performs a stratified train/test split, and
saves the results.

CRITICAL GUARDRAILS:
    - This script does NOT fit any StandardScaler, imputer, or encoder.
    - This script does NOT perform feature selection.
    - This script does NOT train any model.
    - This script does NOT duplicate or fabricate rows.
    - This script does NOT perform any oversampling.

The split is stratified on the target variable (Biopsy) to preserve the
class distribution ratio in both train and test sets. This is especially
important because the positive class represents only ~6.4% of records.

Usage:
    python src/split.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Add project root to path so we can import config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    RAW_DATASET_PATH,
    RANDOM_STATE,
    SPLITS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    DATASET_INFO,
    ensure_directories,
    validate_file_exists,
)


def load_raw_dataset() -> pd.DataFrame:
    """
    Load the raw UCI Cervical Cancer Risk Factors CSV.

    Validates:
        1. The file exists at the expected path.
        2. The dataset contains the expected number of columns.
        3. All expected column names are present.

    Returns:
        pd.DataFrame: The raw dataset with all original values (including '?').
    """
    # --- Validate file exists ---
    validate_file_exists(RAW_DATASET_PATH, "Raw dataset CSV")

    # --- Load ---
    df = pd.read_csv(RAW_DATASET_PATH)

    # --- Validate column count ---
    if df.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"Expected {len(ALL_COLUMNS)} columns, found {df.shape[1]}.\n"
            f"Expected columns: {ALL_COLUMNS}\n"
            f"Found columns: {list(df.columns)}"
        )

    # --- Validate column names ---
    missing_cols = set(ALL_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing expected columns in dataset: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # --- Validate target column ---
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset.\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def split_data(df: pd.DataFrame) -> None:
    """
    Perform a stratified train/test split and save the results.

    The split preserves the class distribution of the target variable
    in both train and test sets. This is critical because the positive
    class (Biopsy=1) represents only ~6.4% of records.

    Args:
        df: The raw dataset DataFrame (all columns including target).
    """
    # --- Separate features and target ---
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # --- Stratified split ---
    # random_state=42 for reproducibility across all runs
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --- Reconstruct full DataFrames (features + target) for saving ---
    # We save complete rows (features + target) so that downstream scripts
    # can load and separate X/y as needed.
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # --- Ensure output directory exists ---
    ensure_directories()

    # --- Save ---
    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)

    # --- Report ---
    print("=" * 65)
    print("  CERVICAL CANCER DATA — SPLIT REPORT")
    print("=" * 65)
    print()

    print(f"  Dataset source:  {DATASET_INFO['source_url']}")
    print(f"  Target variable: {TARGET_COLUMN}")
    print(f"  Random state:    {RANDOM_STATE}")
    print(f"  Test size:       {TEST_SIZE} ({TEST_SIZE * 100:.0f}%)")
    print()

    print("  DATASET DIMENSIONS")
    print("  " + "-" * 40)
    print(f"  Total rows:      {df.shape[0]}")
    print(f"  Total columns:   {df.shape[1]} ({len(FEATURE_COLUMNS)} features + 1 target)")
    print(f"  Training shape:  {train_df.shape}")
    print(f"  Test shape:      {test_df.shape}")
    print()

    # --- Class distribution ---
    print("  OVERALL CLASS DISTRIBUTION")
    print("  " + "-" * 40)
    overall_counts = y.value_counts().sort_index()
    for cls_val, count in overall_counts.items():
        pct = count / len(y) * 100
        label = "Positive (cancer confirmed)" if cls_val == 1 else "Negative (healthy)"
        print(f"  Class {cls_val} ({label}): {count:>4d} ({pct:5.2f}%)")
    print()

    print("  TRAINING SET CLASS DISTRIBUTION")
    print("  " + "-" * 40)
    train_counts = y_train.value_counts().sort_index()
    for cls_val, count in train_counts.items():
        pct = count / len(y_train) * 100
        label = "Positive" if cls_val == 1 else "Negative"
        print(f"  Class {cls_val} ({label}): {count:>4d} ({pct:5.2f}%)")
    print()

    print("  TEST SET CLASS DISTRIBUTION")
    print("  " + "-" * 40)
    test_counts = y_test.value_counts().sort_index()
    for cls_val, count in test_counts.items():
        pct = count / len(y_test) * 100
        label = "Positive" if cls_val == 1 else "Negative"
        print(f"  Class {cls_val} ({label}): {count:>4d} ({pct:5.2f}%)")
    print()

    # --- Verify stratification preserved the ratio ---
    train_pos_rate = y_train.mean()
    test_pos_rate = y_test.mean()
    overall_pos_rate = y.mean()
    print("  STRATIFICATION VERIFICATION")
    print("  " + "-" * 40)
    print(f"  Overall positive rate:  {overall_pos_rate:.4f} ({overall_pos_rate * 100:.2f}%)")
    print(f"  Training positive rate: {train_pos_rate:.4f} ({train_pos_rate * 100:.2f}%)")
    print(f"  Test positive rate:     {test_pos_rate:.4f} ({test_pos_rate * 100:.2f}%)")
    print()

    # --- Dataset limitation notice ---
    print("  [!] DATASET LIMITATION")
    print("  " + "-" * 40)
    print(f"  UCI dataset rows:          {df.shape[0]}")
    print(f"  University minimum target: {DATASET_INFO['university_min_rows']}")
    print(f"  Shortfall:                 {DATASET_INFO['university_min_rows'] - df.shape[0]} rows")
    print("  Records have NOT been fabricated or duplicated.")
    print()

    # --- Files saved ---
    print("  FILES SAVED")
    print("  " + "-" * 40)
    print(f"  Training set: {TRAIN_CSV_PATH}")
    print(f"  Test set:     {TEST_CSV_PATH}")
    print()
    print("=" * 65)
    print("  Split complete. Ready for preprocessing (src/preprocess.py).")
    print("=" * 65)


def main():
    """Entry point: load dataset, validate, split, and save."""
    print("\n[split.py] Loading raw dataset...")
    df = load_raw_dataset()

    print(f"[split.py] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[split.py] Performing stratified train/test split (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...\n")

    split_data(df)


if __name__ == "__main__":
    main()
