"""
Centralized configuration for the Cervical Cancer Patient Data Analysis project.

All paths, constants, random seeds, and column definitions are defined here.
Every other script imports from this module instead of hard-coding values.
"""

import os
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================
# Resolve project root relative to this file's location (src/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# RANDOM SEED — used everywhere for reproducibility
# ============================================================
RANDOM_STATE = 42

# ============================================================
# DATA PATHS
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Raw dataset filename
RAW_DATASET_FILENAME = "risk_factors_cervical_cancer.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / RAW_DATASET_FILENAME

# Split files
TRAIN_CSV_PATH = SPLITS_DIR / "train.csv"
TEST_CSV_PATH = SPLITS_DIR / "test.csv"

# ============================================================
# MODEL PATHS
# ============================================================
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# ARTIFACT PATHS
# ============================================================
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
EXPLAINABILITY_DIR = ARTIFACTS_DIR / "explainability"

# Preprocessing pipeline artifact
PREPROCESSING_PIPELINE_PATH = ARTIFACTS_DIR / "preprocessing_pipeline.joblib"

# Metrics files
CV_RESULTS_JSON_PATH = METRICS_DIR / "cv_results.json"
TEST_RESULTS_JSON_PATH = METRICS_DIR / "test_results.json"

# ============================================================
# TARGET VARIABLE
# ============================================================
TARGET_COLUMN = "Biopsy"

# ============================================================
# DATASET COLUMNS
# ============================================================
# The full list of 36 columns in the UCI Cervical Cancer Risk Factors dataset.
# 35 features + 1 target (Biopsy).
ALL_COLUMNS = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
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
    "STDs: Number of diagnosis",
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV",
    "Dx",
    "Hinselmann",
    "Schiller",
    "Citology",
    "Biopsy",
]

# Feature columns (all columns except the target)
FEATURE_COLUMNS = [col for col in ALL_COLUMNS if col != TARGET_COLUMN]

# Columns that use "?" to represent missing values in the raw CSV.
# These are stored as strings by pandas and need ? → NaN conversion + numeric cast.
COLUMNS_WITH_QUESTION_MARKS = [
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD",
    "IUD (years)",
    "STDs",
    "STDs (number)",
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
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
]

# Columns with extreme missingness (>80%) — flagged for possible exclusion
HIGH_MISSINGNESS_COLUMNS = [
    "STDs: Time since first diagnosis",   # 91.7% missing
    "STDs: Time since last diagnosis",    # 91.7% missing
]

# ============================================================
# TRAIN/TEST SPLIT SETTINGS
# ============================================================
TEST_SIZE = 0.2

# ============================================================
# CROSS-VALIDATION SETTINGS
# ============================================================
CV_N_SPLITS = 5

# ============================================================
# MODEL NAMES — canonical identifiers used across all scripts
# ============================================================
MODEL_NAMES = [
    "LogisticRegression",
    "RandomForest",
    "MLPClassifier",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

# ============================================================
# DATASET METADATA — for documentation and dashboard display
# ============================================================
DATASET_INFO = {
    "name": "UCI Cervical Cancer Risk Factors Dataset",
    "source_url": "https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors",
    "n_records": 858,
    "expected_rows": 858,  # alias used by app.py
    "n_features": 35,
    "target": TARGET_COLUMN,
    "university_min_rows": 2000,
    "origin": "Hospital Universitario de Caracas, Venezuela",
    "citation": "Fernandes, Cardoso & Fernandes (2017). DOI: 10.24432/C5Z310",
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories():
    """Create all required project directories if they don't exist."""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        EXPLAINABILITY_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def validate_file_exists(filepath: Path, description: str = "File"):
    """Raise FileNotFoundError with a clear message if a required file is missing."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"{description} not found at: {filepath}\n"
            f"Please ensure the required pipeline step has been executed."
        )
