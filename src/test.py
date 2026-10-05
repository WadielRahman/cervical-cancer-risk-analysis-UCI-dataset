"""
test.py -- Final Test Evaluation Module

Loads all 6 saved models and the already-processed test set, generates
predictions and probabilities, and computes all evaluation metrics.

CRITICAL GUARDRAILS:
    1. This is the ONE AND ONLY stage where the test set is used.
    2. The preprocessing pipeline is NOT refit -- the test data was
       already transformed in preprocess.py using the training-fitted pipeline.
    3. No model retraining or hyperparameter changes happen here.
    4. Results here are FINAL. If they look bad, we document it honestly.

Metrics computed:
    - Precision, Recall, F1 Score
    - ROC-AUC, PR-AUC
    - Brier Score
    - Confusion Matrix
    - Classification Report

Usage:
    python src/test.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_NAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TEST_RESULTS_JSON_PATH,
    ensure_directories,
    validate_file_exists,
)


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(
    model_name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate a single model on the test set.

    Computes all metrics and returns them as a dictionary.

    Args:
        model_name: Human-readable model name.
        model: Fitted sklearn-compatible model.
        X_test: Test features (numpy array).
        y_test: Test labels (numpy array).

    Returns:
        Dictionary with all computed metrics.
    """
    # --- Predictions ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- Classification Report ---
    cls_report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "model_name": model_name,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": cls_report,
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_confusion_matrices(all_results: list[dict], y_test: np.ndarray) -> None:
    """Plot confusion matrices for all models in a grid."""
    n_models = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, result in enumerate(all_results):
        cm = confusion_matrix(
            y_test,
            result["y_pred"],
        )
        ax = axes[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
            cbar=False,
        )
        name = result["model_name"]
        pr_auc = result["metrics"]["pr_auc"]
        ax.set_title(f"{name}\nPR-AUC: {pr_auc:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    # Hide unused subplots if any
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Confusion Matrices — Final Test Set Evaluation",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(all_results: list[dict], y_test: np.ndarray) -> None:
    """Plot ROC curves for all models on a single axis."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for result in all_results:
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        auc = result["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{result['model_name']} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Final Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pr_curves(all_results: list[dict], y_test: np.ndarray) -> None:
    """Plot Precision-Recall curves for all models on a single axis."""
    fig, ax = plt.subplots(figsize=(8, 7))

    baseline = y_test.mean()

    for result in all_results:
        prec, rec, _ = precision_recall_curve(y_test, result["y_proba"])
        auc = result["metrics"]["pr_auc"]
        ax.plot(rec, prec, label=f"{result['model_name']} (PR-AUC={auc:.3f})", linewidth=2)

    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence={baseline:.3f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Final Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    path = FIGURES_DIR / "pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_metric_comparison(all_results: list[dict]) -> None:
    """Plot a grouped bar chart comparing metrics across models."""
    metrics_to_plot = ["pr_auc", "roc_auc", "f1", "recall", "precision"]
    display_names = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "f1": "F1",
        "recall": "Recall",
        "precision": "Precision",
    }

    model_names = [r["model_name"] for r in all_results]
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)

    x = np.arange(n_models)
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, metric in enumerate(metrics_to_plot):
        values = [r["metrics"][metric] for r in all_results]
        bars = ax.bar(x + i * width, values, width, label=display_names[metric])

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Final Test Set Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    path = FIGURES_DIR / "metric_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# MAIN TEST EVALUATION WORKFLOW
# ============================================================

def main():
    """
    Full test evaluation workflow:
        1. Load processed test data
        2. Load all 6 saved models
        3. Evaluate each model on the test set
        4. Generate comparison visualizations
        5. Save results to artifacts/metrics/
    """
    # --- Validate inputs ---
    x_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    validate_file_exists(x_test_path, "Processed test features")
    validate_file_exists(y_test_path, "Processed test labels")

    # --- Load test data ---
    print("\n[test.py] Loading processed test data...")
    X_test = pd.read_csv(x_test_path).values
    y_test = pd.read_csv(y_test_path).values.ravel()
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Test positives: {int(y_test.sum())} / {len(y_test)} ({y_test.mean()*100:.1f}%)")

    # --- Load and evaluate all models ---
    print("\n[test.py] Evaluating all models on held-out test set...")
    print("  (This is the ONE AND ONLY time the test set is used for evaluation)")
    print()

    all_results = []

    for model_name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        validate_file_exists(model_path, f"Saved model {model_name}")

        model = joblib.load(model_path)
        result = evaluate_model(model_name, model, X_test, y_test)
        all_results.append(result)

        # Print per-model results
        m = result["metrics"]
        cm = result["confusion_matrix"]
        print(f"  {model_name}:")
        print(f"    PR-AUC: {m['pr_auc']:.4f}  |  ROC-AUC: {m['roc_auc']:.4f}")
        print(f"    F1: {m['f1']:.4f}  |  Precision: {m['precision']:.4f}  |  Recall: {m['recall']:.4f}")
        print(f"    Brier: {m['brier_score']:.4f}  |  Accuracy: {m['accuracy']:.4f}")
        print(f"    CM: TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']}")
        print()

    # --- Ensure output directories ---
    ensure_directories()

    # --- Generate visualizations ---
    print("[test.py] Generating visualizations...")
    plot_confusion_matrices(all_results, y_test)
    plot_roc_curves(all_results, y_test)
    plot_pr_curves(all_results, y_test)
    plot_metric_comparison(all_results)

    # --- Summary comparison table ---
    print()
    print("=" * 85)
    print("  FINAL TEST SET RESULTS — MODEL COMPARISON")
    print("=" * 85)
    print()
    header = (
        f"  {'Model':<20s} {'PR-AUC':>8s} {'ROC-AUC':>8s} "
        f"{'F1':>8s} {'Recall':>8s} {'Precision':>10s} "
        f"{'Brier':>8s} {'TP':>4s} {'FN':>4s}"
    )
    print(header)
    print("  " + "-" * 80)

    best_model_name = None
    best_pr_auc = -1.0

    for result in all_results:
        m = result["metrics"]
        cm = result["confusion_matrix"]
        pr = m["pr_auc"]
        line = (
            f"  {result['model_name']:<20s} {pr:>8.4f} {m['roc_auc']:>8.4f} "
            f"{m['f1']:>8.4f} {m['recall']:>8.4f} {m['precision']:>10.4f} "
            f"{m['brier_score']:>8.4f} {cm['tp']:>4d} {cm['fn']:>4d}"
        )
        if pr > best_pr_auc:
            best_pr_auc = pr
            best_model_name = result["model_name"]
        print(line)

    print()
    print(f"  >>> Best model by Test PR-AUC: {best_model_name} ({best_pr_auc:.4f})")
    print()

    # --- Context warning ---
    n_pos = int(y_test.sum())
    print(f"  [!] CONTEXT: Only {n_pos} positive cases in the test set.")
    print(f"      A single misclassification changes Recall by ~{100/n_pos:.0f}%.")
    print("      Cross-validation results (Segment 6) are more reliable for model comparison.")
    print()

    # --- Save results ---
    # Prepare JSON-serializable output (exclude y_pred and y_proba arrays)
    results_for_json = {}
    for result in all_results:
        results_for_json[result["model_name"]] = {
            "metrics": result["metrics"],
            "confusion_matrix": result["confusion_matrix"],
            "is_best_test_model": result["model_name"] == best_model_name,
        }

    with open(TEST_RESULTS_JSON_PATH, "w") as f:
        json.dump(results_for_json, f, indent=2)
    print(f"  Test results saved to: {TEST_RESULTS_JSON_PATH}")

    # Save summary CSV
    rows = []
    for result in all_results:
        m = result["metrics"]
        cm = result["confusion_matrix"]
        rows.append({
            "Model": result["model_name"],
            "PR-AUC": round(m["pr_auc"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "F1": round(m["f1"], 4),
            "Recall": round(m["recall"], 4),
            "Precision": round(m["precision"], 4),
            "Brier Score": round(m["brier_score"], 4),
            "TP": cm["tp"],
            "FP": cm["fp"],
            "FN": cm["fn"],
            "TN": cm["tn"],
        })
    test_csv_path = METRICS_DIR / "test_summary.csv"
    pd.DataFrame(rows).to_csv(test_csv_path, index=False)
    print(f"  Test summary CSV saved to: {test_csv_path}")

    print()
    print("=" * 85)
    print("  Test evaluation complete.")
    print("  DO NOT retrain or tune models based on these results.")
    print("  Ready for SHAP explainability (Segment 8) and dashboard (Segment 9).")
    print("=" * 85)


if __name__ == "__main__":
    main()
