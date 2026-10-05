"""
explain.py -- SHAP Explainability Module

Generates SHAP explanations for all models (except MLPClassifier, which
performed too poorly to warrant explanation). Produces:
    1. SHAP summary bar plots (top features by mean |SHAP|) per model
    2. SHAP beeswarm plot for the best model (Random Forest)
    3. Combined feature importance comparison across all models
    4. Waterfall plot for an individual positive-class patient
    5. Saved SHAP values and background data for the dashboard

SHAP explainer types:
    - LogisticRegression: shap.LinearExplainer
    - RandomForest, XGBoost, LightGBM, CatBoost: shap.TreeExplainer
    - MLPClassifier: SKIPPED (poor performance; KernelExplainer too slow)

Usage:
    python src/explain.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EXPLAINABILITY_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    ensure_directories,
    validate_file_exists,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Models to explain (skip MLP — it detected zero cancer cases)
MODELS_TO_EXPLAIN = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


# ============================================================
# SHAP COMPUTATION
# ============================================================

def compute_shap_for_model(
    model_name: str,
    model,
    X_test: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
) -> np.ndarray | None:
    """
    Compute SHAP values for a single model.

    Uses the appropriate explainer type based on the model:
        - LinearExplainer for Logistic Regression
        - TreeExplainer for tree-based models

    Args:
        model_name: Model identifier.
        model: Fitted model object.
        X_test: Test features for explanation.
        X_train: Training features (background for LinearExplainer).
        feature_names: List of feature names.

    Returns:
        numpy array of SHAP values (n_samples, n_features) or None on failure.
    """
    try:
        if model_name == "LogisticRegression":
            masker = shap.maskers.Independent(X_train)
            explainer = shap.LinearExplainer(model, masker)
        else:
            # TreeExplainer for RF, XGBoost, LightGBM, CatBoost
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test)

        # Handle binary classification output format
        # Some explainers return a list [class_0_values, class_1_values]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # positive class

        return shap_values

    except Exception as e:
        print(f"    [WARNING] SHAP failed for {model_name}: {e}")
        return None


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_shap_summary_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    model_name: str,
    save_dir: Path,
) -> None:
    """Generate and save a SHAP bar plot (mean |SHAP| per feature)."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:15]  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    features = [feature_names[i] for i in sorted_idx]
    values = mean_abs_shap[sorted_idx]

    ax.barh(range(len(features)), values[::-1], color="#1f77b4")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"SHAP Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = save_dir / f"shap_bar_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
    save_dir: Path,
) -> None:
    """Generate and save a SHAP beeswarm plot."""
    explanation = shap.Explanation(
        values=shap_values,
        data=X_test,
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title(f"SHAP Beeswarm — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_dir / f"shap_beeswarm_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
    patient_idx: int,
    save_dir: Path,
) -> None:
    """Generate and save a SHAP waterfall plot for one patient."""
    explanation = shap.Explanation(
        values=shap_values[patient_idx],
        base_values=0,  # approximate; actual base value varies by explainer
        data=X_test[patient_idx],
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — {model_name} (Patient #{patient_idx})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = save_dir / f"shap_waterfall_{model_name}_patient{patient_idx}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def plot_combined_importance(
    all_shap: dict[str, np.ndarray],
    feature_names: list[str],
    save_dir: Path,
) -> None:
    """Plot a combined heatmap of feature importance across all models."""
    # Compute mean |SHAP| for each model
    importance_data = {}
    for model_name, shap_values in all_shap.items():
        importance_data[model_name] = np.abs(shap_values).mean(axis=0)

    df = pd.DataFrame(importance_data, index=feature_names)

    # Select top 15 features by average importance across models
    df["avg"] = df.mean(axis=1)
    top_features = df.nlargest(15, "avg").drop(columns=["avg"])

    fig, ax = plt.subplots(figsize=(12, 8))
    import seaborn as sns
    sns.heatmap(
        top_features,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        "SHAP Feature Importance Comparison (Top 15 Features)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Feature")
    ax.set_xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = save_dir / "shap_importance_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")

    return top_features


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    """
    Full SHAP explainability workflow:
        1. Load models and test data
        2. Compute SHAP values for each model
        3. Generate summary plots
        4. Save SHAP values for dashboard
    """
    # --- Load data ---
    print("\n[explain.py] Loading data...")
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv").values
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv").values
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()

    with open(PROCESSED_DATA_DIR / "feature_names.txt") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")

    ensure_directories()

    # --- Compute SHAP for each model ---
    all_shap = {}

    for model_name in MODELS_TO_EXPLAIN:
        print(f"\n  Computing SHAP: {model_name}")
        model_path = MODELS_DIR / f"{model_name}.joblib"
        validate_file_exists(model_path, f"Model {model_name}")

        model = joblib.load(model_path)
        shap_values = compute_shap_for_model(
            model_name, model, X_test, X_train, feature_names,
        )

        if shap_values is not None:
            all_shap[model_name] = shap_values
            print(f"    SHAP values shape: {shap_values.shape}")

            # Save SHAP values
            np.save(
                EXPLAINABILITY_DIR / f"shap_values_{model_name}.npy",
                shap_values,
            )

            # Generate bar plot for each model
            plot_shap_summary_bar(
                shap_values, feature_names, model_name, EXPLAINABILITY_DIR,
            )

    # --- Beeswarm plot for best model (RandomForest) ---
    if "RandomForest" in all_shap:
        print("\n  Generating beeswarm plot for RandomForest (best CV model)...")
        plot_shap_beeswarm(
            all_shap["RandomForest"], X_test, feature_names,
            "RandomForest", EXPLAINABILITY_DIR,
        )

    # --- Waterfall plot for an individual positive-class patient ---
    positive_indices = np.where(y_test == 1)[0]
    if len(positive_indices) > 0 and "RandomForest" in all_shap:
        patient_idx = positive_indices[0]  # first positive case
        print(f"\n  Generating waterfall plot (Patient #{patient_idx}, Biopsy=1)...")
        plot_shap_waterfall(
            all_shap["RandomForest"], X_test, feature_names,
            "RandomForest", patient_idx, EXPLAINABILITY_DIR,
        )

    # --- Combined importance comparison ---
    if len(all_shap) > 1:
        print("\n  Generating combined importance comparison...")
        importance_df = plot_combined_importance(
            all_shap, feature_names, EXPLAINABILITY_DIR,
        )

    # --- Save background data for dashboard's on-the-fly SHAP ---
    # Use 100 random training samples as background for SHAP explainers
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    background = X_train[bg_idx]
    np.save(EXPLAINABILITY_DIR / "shap_background.npy", background)
    print(f"\n  Saved background data: {background.shape}")

    # --- Save feature names for dashboard ---
    np.save(EXPLAINABILITY_DIR / "feature_names.npy", np.array(feature_names))

    # --- Summary report ---
    print()
    print("=" * 65)
    print("  SHAP EXPLAINABILITY REPORT")
    print("=" * 65)
    print()
    print(f"  Models explained: {len(all_shap)}")
    print(f"  Models skipped:   MLPClassifier (poor performance)")
    print()

    # Top features per model
    for model_name, sv in all_shap.items():
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:5]
        top_features = [(feature_names[i], mean_abs[i]) for i in top_idx]
        print(f"  {model_name} — Top 5 features:")
        for feat, val in top_features:
            print(f"    {feat}: {val:.4f}")
        print()

    print("  FILES SAVED")
    print("  " + "-" * 40)
    print(f"  SHAP values:       {EXPLAINABILITY_DIR}/shap_values_*.npy")
    print(f"  Bar plots:         {EXPLAINABILITY_DIR}/shap_bar_*.png")
    print(f"  Beeswarm plot:     {EXPLAINABILITY_DIR}/shap_beeswarm_RandomForest.png")
    print(f"  Waterfall plot:    {EXPLAINABILITY_DIR}/shap_waterfall_*.png")
    print(f"  Importance heatmap:{EXPLAINABILITY_DIR}/shap_importance_comparison.png")
    print(f"  Background data:   {EXPLAINABILITY_DIR}/shap_background.npy")
    print()
    print("=" * 65)
    print("  Explainability complete. Ready for dashboard (app.py).")
    print("=" * 65)


if __name__ == "__main__":
    main()
