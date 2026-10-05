"""
preprocess.py -- Preprocessing Pipeline Module

Loads the train/test splits produced by split.py, handles missing values,
encodes and standardizes features, and saves a fitted preprocessing pipeline.

CRITICAL DESIGN DECISIONS:
    1. Preprocessing is FIT ONLY on training data.
    2. Test data is TRANSFORMED using the already-fitted pipeline.
    3. The saved pipeline artifact is reused by train.py, test.py, and app.py.
    4. Two columns with 91.7% missing values are DROPPED (not imputed).
    5. '?' values are treated as NaN (missing), not as a literal category.

Column categorisation:
    - Continuous features: imputed with median, then StandardScaler
    - Binary features: imputed with most-frequent value (no scaling needed)

Usage:
    python src/preprocess.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_COLUMNS,
    HIGH_MISSINGNESS_COLUMNS,
    PREPROCESSING_PIPELINE_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    ensure_directories,
    validate_file_exists,
)


# ============================================================
# COLUMN DEFINITIONS FOR PREPROCESSING
# ============================================================

# Columns to DROP before preprocessing (91.7% missing -- imputing would
# create data rather than preserve it; documented as a design decision).
COLUMNS_TO_DROP = list(HIGH_MISSINGNESS_COLUMNS)

# After dropping the target and the two high-missingness columns, the
# remaining 33 features fall into two categories:

# CONTINUOUS features: need imputation (median) + standardisation.
CONTINUOUS_FEATURES = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives (years)",
    "IUD (years)",
    "STDs (number)",
    "STDs: Number of diagnosis",
]

# BINARY features (0/1): need imputation (most-frequent) but NOT scaling.
# Scaling binary features would distort their natural 0/1 interpretation
# and provides no benefit for tree-based models. For linear/MLP models the
# effect is negligible on 0/1 features.
BINARY_FEATURES = [
    "Smokes",
    "Hormonal Contraceptives",
    "IUD",
    "STDs",
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
    "Hinselmann",
    "Schiller",
    "Citology",
]

# The canonical feature order after preprocessing (continuous first, then binary).
# This order is preserved by the ColumnTransformer.
PREPROCESSED_FEATURE_NAMES = CONTINUOUS_FEATURES + BINARY_FEATURES


# ============================================================
# DATA CLEANING (deterministic, no fitting required)
# ============================================================

def clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Clean raw data before the sklearn pipeline processes it.

    This function performs deterministic transformations that do NOT require
    fitting (no statistics are learned):
        1. Separate the target column (if present).
        2. Drop columns with extreme missingness (>80%).
        3. Replace '?' with NaN.
        4. Cast all feature columns to numeric types.

    This function is used by:
        - preprocess.py (during pipeline fitting)
        - app.py (when processing new patient input)

    Args:
        df: Raw DataFrame (may or may not include the target column).

    Returns:
        (X_cleaned, y) where y is None if the target column was not present.
    """
    df = df.copy()

    # --- Separate target if present ---
    y = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].copy()
        df = df.drop(columns=[TARGET_COLUMN])

    # --- Drop high-missingness columns ---
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # --- Replace '?' with NaN ---
    df = df.replace("?", np.nan)

    # --- Convert all columns to numeric ---
    # Many columns are stored as strings because '?' prevented pandas
    # from auto-detecting numeric types during CSV loading.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Validate expected features are present ---
    expected_features = set(CONTINUOUS_FEATURES + BINARY_FEATURES)
    actual_features = set(df.columns)
    missing = expected_features - actual_features
    if missing:
        raise ValueError(
            f"Missing expected feature columns after cleaning: {missing}\n"
            f"Available columns: {sorted(actual_features)}"
        )

    # --- Reorder columns to canonical order ---
    # This ensures consistent feature ordering across all uses.
    df = df[PREPROCESSED_FEATURE_NAMES]

    return df, y


# ============================================================
# SKLEARN PREPROCESSING PIPELINE
# ============================================================

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer for the preprocessing pipeline.

    The pipeline has two branches:
        1. Continuous features: SimpleImputer(median) -> StandardScaler
        2. Binary features:     SimpleImputer(most_frequent) -> passthrough

    Returns:
        A ColumnTransformer (unfitted).
    """
    continuous_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # No scaler: binary 0/1 features don't benefit from scaling.
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        # verbose_feature_names_out=False gives clean column names
        # without transformer prefixes (e.g. "Age" not "continuous__Age").
        verbose_feature_names_out=False,
        remainder="drop",  # safety: drop any unexpected columns
    )

    return preprocessor


# ============================================================
# MAIN PREPROCESSING WORKFLOW
# ============================================================

def main():
    """
    Full preprocessing workflow:
        1. Load train/test splits
        2. Clean raw data (? -> NaN, drop high-missingness columns)
        3. Fit preprocessing pipeline on TRAINING data ONLY
        4. Transform both train and test
        5. Save pipeline artifact and processed data
    """
    # --- Validate inputs exist ---
    validate_file_exists(TRAIN_CSV_PATH, "Training split (train.csv)")
    validate_file_exists(TEST_CSV_PATH, "Test split (test.csv)")

    # --- Load splits ---
    print("\n[preprocess.py] Loading train/test splits...")
    train_raw = pd.read_csv(TRAIN_CSV_PATH)
    test_raw = pd.read_csv(TEST_CSV_PATH)
    print(f"  Train raw shape: {train_raw.shape}")
    print(f"  Test raw shape:  {test_raw.shape}")

    # --- Clean raw data ---
    print("\n[preprocess.py] Cleaning raw data...")
    print(f"  Dropping {len(COLUMNS_TO_DROP)} high-missingness columns: {COLUMNS_TO_DROP}")
    print("  Replacing '?' with NaN and converting to numeric types...")

    X_train, y_train = clean_raw_data(train_raw)
    X_test, y_test = clean_raw_data(test_raw)

    print(f"  X_train cleaned shape: {X_train.shape}")
    print(f"  X_test cleaned shape:  {X_test.shape}")
    print(f"  Continuous features:   {len(CONTINUOUS_FEATURES)}")
    print(f"  Binary features:       {len(BINARY_FEATURES)}")
    print(f"  Total features:        {len(PREPROCESSED_FEATURE_NAMES)}")

    # --- Report missing values in cleaned training data ---
    train_missing = X_train.isnull().sum()
    train_missing = train_missing[train_missing > 0]
    if len(train_missing) > 0:
        print(f"\n  Missing values in training data ({len(train_missing)} columns):")
        for col, count in train_missing.items():
            pct = count / len(X_train) * 100
            print(f"    {col}: {count} ({pct:.1f}%)")
    else:
        print("\n  No missing values in training data.")

    # --- Build and FIT pipeline on TRAINING data ONLY ---
    print("\n[preprocess.py] Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline()

    print("[preprocess.py] Fitting pipeline on TRAINING data ONLY...")
    X_train_processed = preprocessor.fit_transform(X_train)
    print("  Pipeline fitted successfully.")

    # --- Transform test data using ALREADY FITTED pipeline ---
    print("[preprocess.py] Transforming test data using fitted pipeline (NO refitting)...")
    X_test_processed = preprocessor.transform(X_test)
    print("  Test data transformed successfully.")

    # --- Get feature names ---
    feature_names = preprocessor.get_feature_names_out().tolist()

    # --- Convert to DataFrames for saving ---
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # --- Ensure directories exist ---
    ensure_directories()

    # --- Save preprocessing pipeline ---
    joblib.dump(preprocessor, PREPROCESSING_PIPELINE_PATH)
    print(f"\n[preprocess.py] Saved preprocessing pipeline to: {PREPROCESSING_PIPELINE_PATH}")

    # --- Save processed data ---
    train_processed_path = PROCESSED_DATA_DIR / "X_train.csv"
    test_processed_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    feature_names_path = PROCESSED_DATA_DIR / "feature_names.txt"

    X_train_df.to_csv(train_processed_path, index=False)
    X_test_df.to_csv(test_processed_path, index=False)
    y_train.to_csv(y_train_path, index=False, header=True)
    y_test.to_csv(y_test_path, index=False, header=True)

    with open(feature_names_path, "w") as f:
        for name in feature_names:
            f.write(name + "\n")

    # --- Final report ---
    print()
    print("=" * 65)
    print("  PREPROCESSING REPORT")
    print("=" * 65)
    print()
    print("  DATA CLEANING")
    print("  " + "-" * 40)
    print(f"  Columns dropped (>80% missing):  {COLUMNS_TO_DROP}")
    print(f"  '?' values replaced with NaN:    Yes")
    print(f"  All columns cast to numeric:     Yes")
    print()
    print("  PIPELINE COMPONENTS")
    print("  " + "-" * 40)
    print(f"  Continuous ({len(CONTINUOUS_FEATURES)} cols): median imputation + StandardScaler")
    print(f"  Binary ({len(BINARY_FEATURES)} cols):     most-frequent imputation (no scaling)")
    print()
    print("  OUTPUT DIMENSIONS")
    print("  " + "-" * 40)
    print(f"  X_train processed: {X_train_df.shape}")
    print(f"  X_test processed:  {X_test_df.shape}")
    print(f"  Feature count:     {len(feature_names)}")
    print()
    print("  LEAKAGE PREVENTION")
    print("  " + "-" * 40)
    print("  Scaler fitted on:     TRAINING data ONLY")
    print("  Imputer fitted on:    TRAINING data ONLY")
    print("  Test data:            TRANSFORMED only (no fitting)")
    print()

    # --- Verify no NaN in processed data ---
    train_nan = np.isnan(X_train_processed).sum()
    test_nan = np.isnan(X_test_processed).sum()
    print("  DATA QUALITY CHECK")
    print("  " + "-" * 40)
    print(f"  NaN in processed train: {train_nan}")
    print(f"  NaN in processed test:  {test_nan}")
    if train_nan > 0 or test_nan > 0:
        print("  [WARNING] NaN values remain after preprocessing!")
    else:
        print("  All NaN values handled successfully.")
    print()

    print("  FILES SAVED")
    print("  " + "-" * 40)
    print(f"  Pipeline:       {PREPROCESSING_PIPELINE_PATH}")
    print(f"  X_train:        {train_processed_path}")
    print(f"  X_test:         {test_processed_path}")
    print(f"  y_train:        {y_train_path}")
    print(f"  y_test:         {y_test_path}")
    print(f"  Feature names:  {feature_names_path}")
    print()
    print("=" * 65)
    print("  Preprocessing complete. Ready for training (src/train.py).")
    print("=" * 65)


if __name__ == "__main__":
    main()
