"""
Segment 3A — Preprocessing Milestone Test

This test validates that the saved preprocessing pipeline can correctly
transform a single synthetic patient record, matching the schema and
dimensionality of the training data.

This is the MOST IMPORTANT integration test before model training.
If this fails, the Streamlit dashboard will also fail during live demos.

Tests:
    1. Load the saved preprocessing pipeline.
    2. Construct one synthetic patient record using the actual dataset schema.
    3. Transform the record.
    4. Confirm transformation succeeds without errors.
    5. Confirm output dimensionality matches training data.
    6. Confirm feature ordering is consistent.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PREPROCESSING_PIPELINE_PATH,
    PROCESSED_DATA_DIR,
    validate_file_exists,
)
from src.preprocess import (
    PREPROCESSED_FEATURE_NAMES,
    clean_raw_data,
)


def create_synthetic_patient() -> pd.DataFrame:
    """
    Construct one synthetic patient record using the ACTUAL dataset schema.

    Values are realistic but entirely fictional. This tests the pipeline's
    ability to handle a single-row DataFrame -- the exact use case for app.py.
    """
    patient = {
        # Continuous features
        "Age": 32,
        "Number of sexual partners": 2,
        "First sexual intercourse": 17,
        "Num of pregnancies": 1,
        "Smokes": 0,
        "Smokes (years)": 0.0,
        "Smokes (packs/year)": 0.0,
        "Hormonal Contraceptives": 1,
        "Hormonal Contraceptives (years)": 3.0,
        "IUD": 0,
        "IUD (years)": 0.0,
        "STDs": 0,
        "STDs (number)": 0,
        "STDs:condylomatosis": 0,
        "STDs:cervical condylomatosis": 0,
        "STDs:vaginal condylomatosis": 0,
        "STDs:vulvo-perineal condylomatosis": 0,
        "STDs:syphilis": 0,
        "STDs:pelvic inflammatory disease": 0,
        "STDs:genital herpes": 0,
        "STDs:molluscum contagiosum": 0,
        "STDs:AIDS": 0,
        "STDs:HIV": 0,
        "STDs:Hepatitis B": 0,
        "STDs:HPV": 0,
        "STDs: Number of diagnosis": 0,
        # These two will be dropped by clean_raw_data:
        "STDs: Time since first diagnosis": "?",
        "STDs: Time since last diagnosis": "?",
        "Dx:Cancer": 0,
        "Dx:CIN": 0,
        "Dx:HPV": 0,
        "Dx": 0,
        "Hinselmann": 0,
        "Schiller": 0,
        "Citology": 0,
        # No Biopsy column (this is a new patient, not training data)
    }
    return pd.DataFrame([patient])


def run_milestone_test():
    """Execute the preprocessing milestone test."""
    print("=" * 65)
    print("  SEGMENT 3A -- PREPROCESSING MILESTONE TEST")
    print("=" * 65)
    print()

    # --- Test 1: Load saved pipeline ---
    print("  TEST 1: Load saved preprocessing pipeline")
    validate_file_exists(PREPROCESSING_PIPELINE_PATH, "Preprocessing pipeline")
    preprocessor = joblib.load(PREPROCESSING_PIPELINE_PATH)
    print("    [PASS] Pipeline loaded successfully.")
    print()

    # --- Test 2: Construct synthetic patient ---
    print("  TEST 2: Construct synthetic patient record")
    patient_raw = create_synthetic_patient()
    print(f"    Raw patient shape: {patient_raw.shape}")
    print(f"    Raw patient columns ({len(patient_raw.columns)}): {list(patient_raw.columns)[:5]}... (truncated)")
    print("    [PASS] Synthetic patient constructed.")
    print()

    # --- Test 3: Clean the record ---
    print("  TEST 3: Clean raw patient data")
    patient_cleaned, _ = clean_raw_data(patient_raw)
    print(f"    Cleaned patient shape: {patient_cleaned.shape}")
    print(f"    Expected features: {len(PREPROCESSED_FEATURE_NAMES)}")
    assert patient_cleaned.shape[1] == len(PREPROCESSED_FEATURE_NAMES), (
        f"Feature count mismatch: got {patient_cleaned.shape[1]}, "
        f"expected {len(PREPROCESSED_FEATURE_NAMES)}"
    )
    print("    [PASS] Cleaning succeeded, feature count correct.")
    print()

    # --- Test 4: Transform using saved pipeline ---
    print("  TEST 4: Transform using saved preprocessing pipeline")
    patient_transformed = preprocessor.transform(patient_cleaned)
    print(f"    Transformed shape: {patient_transformed.shape}")
    print("    [PASS] Transformation succeeded without errors.")
    print()

    # --- Test 5: Verify output dimensionality ---
    print("  TEST 5: Verify output dimensionality matches training data")
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    expected_shape_cols = X_train.shape[1]
    actual_shape_cols = patient_transformed.shape[1]
    print(f"    Training data features: {expected_shape_cols}")
    print(f"    Patient record features: {actual_shape_cols}")
    assert actual_shape_cols == expected_shape_cols, (
        f"Dimensionality mismatch: patient has {actual_shape_cols} features, "
        f"training has {expected_shape_cols}"
    )
    print("    [PASS] Dimensionality matches.")
    print()

    # --- Test 6: Verify feature ordering ---
    print("  TEST 6: Verify feature ordering is consistent")
    pipeline_feature_names = preprocessor.get_feature_names_out().tolist()
    training_feature_names = X_train.columns.tolist()
    assert pipeline_feature_names == training_feature_names, (
        f"Feature ordering mismatch.\n"
        f"Pipeline: {pipeline_feature_names[:5]}...\n"
        f"Training: {training_feature_names[:5]}..."
    )
    print(f"    Pipeline features: {pipeline_feature_names[:5]}... ({len(pipeline_feature_names)} total)")
    print(f"    Training features: {training_feature_names[:5]}... ({len(training_feature_names)} total)")
    print("    [PASS] Feature ordering is consistent.")
    print()

    # --- Test 7: Verify no NaN in output ---
    print("  TEST 7: Verify no NaN in transformed output")
    nan_count = np.isnan(patient_transformed).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in transformed output"
    print(f"    NaN count: {nan_count}")
    print("    [PASS] No NaN values in output.")
    print()

    # --- Test 8: Test with missing values (simulating incomplete patient input) ---
    print("  TEST 8: Transform patient with missing values")
    patient_with_missing = create_synthetic_patient()
    # Simulate a patient who didn't answer some questions.
    # Use np.nan (not "?") because the DataFrame already has numeric dtypes.
    # In real CSV loading, "?" would be present as strings, but clean_raw_data
    # converts them to NaN anyway -- so this tests the imputation path directly.
    patient_with_missing["Number of sexual partners"] = np.nan
    patient_with_missing["Smokes (years)"] = np.nan
    patient_cleaned_missing, _ = clean_raw_data(patient_with_missing)
    patient_transformed_missing = preprocessor.transform(patient_cleaned_missing)
    nan_count_missing = np.isnan(patient_transformed_missing).sum()
    assert nan_count_missing == 0, f"Found {nan_count_missing} NaN values after imputation"
    print(f"    Transformed shape: {patient_transformed_missing.shape}")
    print(f"    NaN count after imputation: {nan_count_missing}")
    print("    [PASS] Missing values handled correctly by pipeline imputer.")
    print()

    # --- Summary ---
    print("=" * 65)
    print("  ALL 8 TESTS PASSED")
    print("=" * 65)
    print()
    print("  Input schema:      35 raw features (+ 2 dropped = 33 used)")
    print(f"  Transformed shape: {patient_transformed.shape}")
    print(f"  Expected shape:    (1, {expected_shape_cols})")
    print(f"  Feature ordering:  Consistent with training data")
    print()
    print("  The preprocessing pipeline is ready for model training")
    print("  and for live patient predictions in app.py.")
    print("=" * 65)


if __name__ == "__main__":
    run_milestone_test()
