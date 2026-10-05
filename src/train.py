"""
train.py -- Model Training Module

Trains and tunes 6 classification models using stratified 5-fold
cross-validation on the training data ONLY. The test set is never
used during this stage.

Models trained:
    1. Logistic Regression   (class_weight='balanced')
    2. Random Forest          (class_weight='balanced')
    3. MLPClassifier          (no native class-weight -- documented limitation)
    4. XGBoost                (scale_pos_weight)
    5. LightGBM               (is_unbalance=True)
    6. CatBoost               (auto_class_weights='Balanced')

Model selection is based on cross-validation PR-AUC (primary metric).
The test set remains untouched until test.py.

Usage:
    python src/train.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_validate,
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CV_N_SPLITS,
    CV_RESULTS_JSON_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    ensure_directories,
    validate_file_exists,
)

# ---------------------------------------------------------------------------
# Suppress non-critical warnings for cleaner output
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# SCORING DEFINITIONS
# ============================================================
# All metrics used for model evaluation. PR-AUC is the primary
# model-selection metric because it focuses on the minority class.

SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": make_scorer(
        average_precision_score, response_method="predict_proba"
    ),
    # Brier Score is a loss (lower=better). make_scorer negates it
    # internally when greater_is_better=False, so cross_validate
    # returns negative values. We negate back when reporting.
    "neg_brier": make_scorer(
        brier_score_loss,
        greater_is_better=False,
        response_method="predict_proba",
    ),
}


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_model_definitions(scale_pos_weight: float) -> dict:
    """
    Define all 6 required models with their fixed parameters and
    small, defensible hyperparameter search spaces.

    Args:
        scale_pos_weight: Ratio of negative to positive samples
                          (n_neg / n_pos), used by XGBoost.

    Returns:
        Dictionary of model definitions with class, fixed params,
        and parameter grid for each model.
    """
    return {
        # ----------------------------------------------------------
        # 1. LOGISTIC REGRESSION — interpretable baseline
        # ----------------------------------------------------------
        "LogisticRegression": {
            "class": LogisticRegression,
            "fixed_params": {
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 2000,
                "random_state": RANDOM_STATE,
            },
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
            },
            "description": "Regularisation strength C tuned; class_weight='balanced'",
        },
        # ----------------------------------------------------------
        # 2. RANDOM FOREST — non-linear ensemble baseline
        # ----------------------------------------------------------
        "RandomForest": {
            "class": RandomForestClassifier,
            "fixed_params": {
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
            },
            "description": "n_estimators and max_depth tuned; class_weight='balanced'",
        },
        # ----------------------------------------------------------
        # 3. MLPCLASSIFIER — required neural network model
        # IMPORTANT: MLPClassifier does NOT support class_weight.
        # This is documented as a known limitation (Segment 4).
        # ----------------------------------------------------------
        "MLPClassifier": {
            "class": MLPClassifier,
            "fixed_params": {
                "activation": "relu",
                "solver": "adam",
                "max_iter": 1500,
                "random_state": RANDOM_STATE,
                "early_stopping": True,
                "validation_fraction": 0.15,
            },
            "param_grid": {
                "hidden_layer_sizes": [(64, 32), (100, 50)],
                "alpha": [0.001, 0.01, 0.1],
            },
            "description": "Architecture and regularisation tuned; no class_weight (limitation)",
        },
        # ----------------------------------------------------------
        # 4. XGBOOST — gradient boosting (depth-wise)
        # ----------------------------------------------------------
        "XGBoost": {
            "class": XGBClassifier,
            "fixed_params": {
                "scale_pos_weight": scale_pos_weight,
                "random_state": RANDOM_STATE,
                "verbosity": 0,
            },
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
            "description": "Trees, depth, LR tuned; scale_pos_weight for imbalance",
        },
        # ----------------------------------------------------------
        # 5. LIGHTGBM — gradient boosting (leaf-wise)
        # ----------------------------------------------------------
        "LightGBM": {
            "class": LGBMClassifier,
            "fixed_params": {
                "is_unbalance": True,
                "random_state": RANDOM_STATE,
                "verbose": -1,
                "force_col_wise": True,
            },
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
            "description": "Trees, depth, LR tuned; is_unbalance=True for imbalance",
        },
        # ----------------------------------------------------------
        # 6. CATBOOST — gradient boosting (categorical-aware)
        # ----------------------------------------------------------
        "CatBoost": {
            "class": CatBoostClassifier,
            "fixed_params": {
                "auto_class_weights": "Balanced",
                "random_seed": RANDOM_STATE,
                "verbose": 0,
            },
            "param_grid": {
                "iterations": [100, 200],
                "depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
            "description": "Iterations, depth, LR tuned; auto_class_weights='Balanced'",
        },
    }


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

def search_best_params(
    model_name: str,
    model_def: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[dict | None, dict | None]:
    """
    Search over the parameter grid using cross-validation.

    For each parameter combination, runs 5-fold stratified CV and
    records all metrics. Selects the best configuration by mean
    CV PR-AUC (the primary model-selection metric).

    Args:
        model_name: Human-readable model name.
        model_def: Dictionary containing 'class', 'fixed_params', 'param_grid'.
        X: Training features (numpy array).
        y: Training labels (numpy array).
        cv: StratifiedKFold cross-validator.

    Returns:
        (best_all_params, best_cv_metrics) or (None, None) if all configs failed.
    """
    param_combinations = list(ParameterGrid(model_def["param_grid"]))
    n_combos = len(param_combinations)

    best_pr_auc = -1.0
    best_all_params = None
    best_cv_results = None

    for i, tuning_params in enumerate(param_combinations, 1):
        # Merge fixed + tuning parameters
        all_params = {**model_def["fixed_params"], **tuning_params}
        model = model_def["class"](**all_params)

        try:
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=SCORING,
                error_score="raise",
                return_train_score=False,
            )

            mean_pr_auc = cv_results["test_pr_auc"].mean()
            marker = " *** BEST ***" if mean_pr_auc > best_pr_auc else ""
            print(f"    [{i}/{n_combos}] {tuning_params} -> PR-AUC: {mean_pr_auc:.4f}{marker}")

            if mean_pr_auc > best_pr_auc:
                best_pr_auc = mean_pr_auc
                best_all_params = all_params
                best_cv_results = cv_results

        except Exception as e:
            print(f"    [{i}/{n_combos}] {tuning_params} -> FAILED: {e}")
            continue

    if best_all_params is None:
        return None, None

    # Summarise best CV results into a clean dictionary
    brier_mean = -best_cv_results["test_neg_brier"].mean()  # negate back
    brier_std = best_cv_results["test_neg_brier"].std()

    best_metrics = {
        "precision": {
            "mean": float(best_cv_results["test_precision"].mean()),
            "std": float(best_cv_results["test_precision"].std()),
        },
        "recall": {
            "mean": float(best_cv_results["test_recall"].mean()),
            "std": float(best_cv_results["test_recall"].std()),
        },
        "f1": {
            "mean": float(best_cv_results["test_f1"].mean()),
            "std": float(best_cv_results["test_f1"].std()),
        },
        "roc_auc": {
            "mean": float(best_cv_results["test_roc_auc"].mean()),
            "std": float(best_cv_results["test_roc_auc"].std()),
        },
        "pr_auc": {
            "mean": float(best_cv_results["test_pr_auc"].mean()),
            "std": float(best_cv_results["test_pr_auc"].std()),
        },
        "brier_score": {
            "mean": float(brier_mean),
            "std": float(brier_std),
        },
    }

    return best_all_params, best_metrics


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def params_to_serializable(params: dict, fixed_keys: set) -> dict:
    """
    Extract tuning-only parameters and make JSON-serializable.

    Converts tuples (e.g. hidden_layer_sizes) to lists for JSON.
    """
    tuning = {}
    for k, v in params.items():
        if k not in fixed_keys:
            if isinstance(v, tuple):
                v = list(v)
            tuning[k] = v
    return tuning


# ============================================================
# MAIN TRAINING WORKFLOW
# ============================================================

def main():
    """
    Full training workflow:
        1. Load processed training data
        2. Compute class imbalance weight
        3. For each model: hyperparameter search via 5-fold stratified CV
        4. Select best config by PR-AUC
        5. Refit best config on full training set
        6. Save model and CV results
    """
    # --- Validate inputs ---
    x_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    validate_file_exists(x_train_path, "Processed training features")
    validate_file_exists(y_train_path, "Processed training labels")

    # --- Load data ---
    print("\n[train.py] Loading processed training data...")
    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Convert to numpy for consistency across all models
    X_train_np = X_train.values

    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # --- Compute class imbalance ratio ---
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    print(f"  Class balance: {n_neg} negative, {n_pos} positive (ratio {scale_pos_weight:.2f}:1)")

    # --- Define CV strategy ---
    skf = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    print(f"  CV strategy: StratifiedKFold(n_splits={CV_N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})")
    print(f"  Primary selection metric: PR-AUC")

    # --- Get model definitions ---
    model_defs = get_model_definitions(scale_pos_weight)

    # --- Ensure output directories exist ---
    ensure_directories()

    # --- Train all models ---
    all_results = {}

    for model_name, model_def in model_defs.items():
        print(f"\n{'=' * 65}")
        print(f"  TRAINING: {model_name}")
        print(f"  {model_def['description']}")
        print(f"  Search space: {len(list(ParameterGrid(model_def['param_grid'])))} configurations")
        print(f"{'=' * 65}")

        start_time = time.time()

        best_params, best_metrics = search_best_params(
            model_name, model_def, X_train_np, y_train, skf,
        )
        elapsed = time.time() - start_time

        if best_params is None:
            print(f"\n  [ERROR] No valid configuration found for {model_name}. Skipping.")
            continue

        # --- Report best CV results ---
        m = best_metrics
        fixed_keys = set(model_def["fixed_params"].keys())
        tuning_only = params_to_serializable(best_params, fixed_keys)

        print(f"\n  BEST CONFIGURATION:")
        print(f"    Tuned params: {tuning_only}")
        print(f"    PR-AUC:       {m['pr_auc']['mean']:.4f} +/- {m['pr_auc']['std']:.4f}")
        print(f"    ROC-AUC:      {m['roc_auc']['mean']:.4f} +/- {m['roc_auc']['std']:.4f}")
        print(f"    F1:           {m['f1']['mean']:.4f} +/- {m['f1']['std']:.4f}")
        print(f"    Precision:    {m['precision']['mean']:.4f} +/- {m['precision']['std']:.4f}")
        print(f"    Recall:       {m['recall']['mean']:.4f} +/- {m['recall']['std']:.4f}")
        print(f"    Brier Score:  {m['brier_score']['mean']:.4f} +/- {m['brier_score']['std']:.4f}")
        print(f"    Time:         {elapsed:.1f}s")

        # --- Refit on full training set ---
        print(f"\n  Refitting {model_name} on full training set ({len(y_train)} samples)...")
        final_model = model_def["class"](**best_params)
        final_model.fit(X_train_np, y_train)

        # --- Save model ---
        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(final_model, model_path)
        print(f"  Model saved: {model_path}")

        # --- Store results ---
        all_results[model_name] = {
            "best_tuning_params": tuning_only,
            "cv_metrics": best_metrics,
            "training_time_seconds": round(elapsed, 1),
        }

    # ============================================================
    # SUMMARY COMPARISON TABLE
    # ============================================================
    print(f"\n{'=' * 80}")
    print("  MODEL COMPARISON SUMMARY (Cross-Validation on Training Data)")
    print(f"{'=' * 80}")
    print()

    header = (
        f"  {'Model':<20s} {'PR-AUC':>8s} {'ROC-AUC':>8s} "
        f"{'F1':>8s} {'Recall':>8s} {'Precision':>10s} {'Brier':>8s}"
    )
    print(header)
    print("  " + "-" * 72)

    best_model_name = None
    best_model_pr_auc = -1.0

    for name, res in all_results.items():
        m = res["cv_metrics"]
        pr = m["pr_auc"]["mean"]
        line = (
            f"  {name:<20s} {pr:>8.4f} {m['roc_auc']['mean']:>8.4f} "
            f"{m['f1']['mean']:>8.4f} {m['recall']['mean']:>8.4f} "
            f"{m['precision']['mean']:>10.4f} {m['brier_score']['mean']:>8.4f}"
        )

        if pr > best_model_pr_auc:
            best_model_pr_auc = pr
            best_model_name = name

        print(line)

    print()
    print(f"  >>> Best model by CV PR-AUC: {best_model_name} ({best_model_pr_auc:.4f})")
    print()

    # Mark the best model
    for name in all_results:
        all_results[name]["is_best_cv_model"] = (name == best_model_name)

    # ============================================================
    # SAVE CV RESULTS
    # ============================================================
    with open(CV_RESULTS_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  CV results saved to: {CV_RESULTS_JSON_PATH}")

    # Also save as CSV for easy viewing
    rows = []
    for name, res in all_results.items():
        m = res["cv_metrics"]
        rows.append({
            "Model": name,
            "PR-AUC": round(m["pr_auc"]["mean"], 4),
            "ROC-AUC": round(m["roc_auc"]["mean"], 4),
            "F1": round(m["f1"]["mean"], 4),
            "Recall": round(m["recall"]["mean"], 4),
            "Precision": round(m["precision"]["mean"], 4),
            "Brier Score": round(m["brier_score"]["mean"], 4),
            "Best": name == best_model_name,
        })
    cv_csv_path = PROCESSED_DATA_DIR.parent.parent / "artifacts" / "metrics" / "cv_summary.csv"
    pd.DataFrame(rows).to_csv(cv_csv_path, index=False)
    print(f"  CV summary CSV saved to: {cv_csv_path}")

    print()
    print("=" * 80)
    print("  Training complete. Models saved to models/.")
    print("  DO NOT alter models based on test results.")
    print("  Ready for final test evaluation (src/test.py).")
    print("=" * 80)


if __name__ == "__main__":
    main()
