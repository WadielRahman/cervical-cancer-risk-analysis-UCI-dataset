"""
End-to-End Integration Test

Cleans all generated artifacts, runs the entire pipeline from scratch,
and validates that every output is produced correctly.

This proves the project is fully reproducible.

Usage:
    python src/integration_test.py
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXPLAINABILITY_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    PREPROCESSING_PIPELINE_PATH,
)

# ============================================================
# EXPECTED OUTPUTS
# ============================================================

EXPECTED_SPLITS = ["train.csv", "test.csv"]

EXPECTED_PROCESSED = [
    "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "feature_names.txt",
]

EXPECTED_MODELS = [
    "LogisticRegression.joblib", "RandomForest.joblib", "MLPClassifier.joblib",
    "XGBoost.joblib", "LightGBM.joblib", "CatBoost.joblib",
]

EXPECTED_METRICS = [
    "cv_results.json", "cv_summary.csv", "test_results.json", "test_summary.csv",
]

EXPECTED_FIGURES = [
    "confusion_matrices.png", "roc_curves.png", "pr_curves.png", "metric_comparison.png",
]

EXPECTED_EXPLAINABILITY = [
    "shap_values_RandomForest.npy", "shap_values_LogisticRegression.npy",
    "shap_values_XGBoost.npy", "shap_values_LightGBM.npy", "shap_values_CatBoost.npy",
    "shap_bar_RandomForest.png", "shap_beeswarm_RandomForest.png",
    "shap_importance_comparison.png", "shap_background.npy", "feature_names.npy",
]


# ============================================================
# CLEAN
# ============================================================

def clean_artifacts():
    """Remove all generated artifacts to test from clean state."""
    dirs_to_clean = [SPLITS_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR, EXPLAINABILITY_DIR]

    print("  CLEANING ARTIFACTS")
    print("  " + "-" * 40)
    for d in dirs_to_clean:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
            print(f"    Cleaned: {d}")

    if PREPROCESSING_PIPELINE_PATH.exists():
        PREPROCESSING_PIPELINE_PATH.unlink()
        print(f"    Removed: {PREPROCESSING_PIPELINE_PATH}")

    print()


# ============================================================
# RUN PIPELINE
# ============================================================

def run_step(script_name: str, description: str) -> tuple[bool, float]:
    """Run a pipeline step and return (success, elapsed_seconds)."""
    print(f"  RUNNING: {description}")
    print(f"    Script: {script_name}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"    Status: PASSED (exit code 0, {elapsed:.1f}s)")
    else:
        print(f"    Status: FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
        print(f"    STDERR: {result.stderr[:500]}")

    return result.returncode == 0, elapsed


# ============================================================
# VALIDATE OUTPUTS
# ============================================================

def validate_outputs() -> tuple[int, int]:
    """Check that all expected output files exist. Return (passed, failed)."""
    passed = 0
    failed = 0

    checks = [
        ("Splits", SPLITS_DIR, EXPECTED_SPLITS),
        ("Processed Data", PROCESSED_DATA_DIR, EXPECTED_PROCESSED),
        ("Models", MODELS_DIR, EXPECTED_MODELS),
        ("Metrics", METRICS_DIR, EXPECTED_METRICS),
        ("Figures", FIGURES_DIR, EXPECTED_FIGURES),
        ("Explainability", EXPLAINABILITY_DIR, EXPECTED_EXPLAINABILITY),
        ("Pipeline", PREPROCESSING_PIPELINE_PATH.parent, [PREPROCESSING_PIPELINE_PATH.name]),
    ]

    for category, directory, expected_files in checks:
        print(f"  {category}:")
        for filename in expected_files:
            filepath = directory / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"    [PASS] {filename} ({size:,} bytes)")
                passed += 1
            else:
                print(f"    [FAIL] {filename} — NOT FOUND")
                failed += 1

    return passed, failed


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 65)
    print("  END-TO-END INTEGRATION TEST")
    print("  Testing full pipeline reproducibility from clean state")
    print("=" * 65)
    print()

    # --- Step 1: Clean ---
    clean_artifacts()

    # --- Step 2: Run full pipeline ---
    pipeline_steps = [
        ("src/split.py", "Step 1: Data Splitting"),
        ("src/preprocess.py", "Step 2: Preprocessing"),
        ("src/train.py", "Step 3: Model Training"),
        ("src/test.py", "Step 4: Test Evaluation"),
        ("src/explain.py", "Step 5: SHAP Explainability"),
        ("src/test_preprocessing.py", "Step 6: Preprocessing Milestone Test"),
    ]

    results = []
    total_time = 0

    print("  PIPELINE EXECUTION")
    print("  " + "-" * 40)

    for script, desc in pipeline_steps:
        success, elapsed = run_step(script, desc)
        results.append((desc, success, elapsed))
        total_time += elapsed

        if not success:
            print(f"\n  [ABORT] Pipeline failed at: {desc}")
            print("  Remaining steps skipped.")
            break

        print()

    # --- Step 3: Validate outputs ---
    print()
    print("  OUTPUT VALIDATION")
    print("  " + "-" * 40)
    passed, failed = validate_outputs()

    # --- Summary ---
    print()
    print("=" * 65)
    print("  INTEGRATION TEST SUMMARY")
    print("=" * 65)
    print()

    print("  PIPELINE STEPS:")
    for desc, success, elapsed in results:
        status = "PASSED" if success else "FAILED"
        print(f"    [{status}] {desc} ({elapsed:.1f}s)")

    print()
    print(f"  Total pipeline time: {total_time:.1f}s")
    print()
    print(f"  OUTPUT FILES:")
    print(f"    Passed: {passed}")
    print(f"    Failed: {failed}")
    print()

    all_steps_passed = all(s for _, s, _ in results)

    if all_steps_passed and failed == 0:
        print("  *** ALL TESTS PASSED — PROJECT IS FULLY REPRODUCIBLE ***")
    elif all_steps_passed:
        print(f"  Pipeline passed but {failed} output file(s) missing.")
    else:
        print("  *** PIPELINE FAILED — SEE ERRORS ABOVE ***")

    print()
    print("=" * 65)

    return 0 if (all_steps_passed and failed == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
