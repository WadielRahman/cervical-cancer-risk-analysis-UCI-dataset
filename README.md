# Cervical Cancer Patient Data Analysis

## Project Overview

**Domain:** Healthcare / Data Analytics / Machine Learning

**Objective:** Analyze cervical cancer risk factors using the UCI Cervical Cancer
Risk Factors dataset and build an interactive, explainable machine-learning
application for risk prediction.

**Dataset:** [UCI Cervical Cancer Risk Factors](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)
— 858 patient records, 35 features, target variable: `Biopsy`

> **Dataset Limitation:** The UCI dataset contains **858 records**. The
> university course guideline specifies a minimum of **2,000 rows**. This
> discrepancy is transparently documented. Records were **NOT** fabricated or
> duplicated to meet the row requirement. Methodological honesty is preserved.

## Medical Disclaimer

> **This is a machine-learning / data-analytics educational project, NOT a
> clinical diagnostic system.** Predictions are model-based probability
> estimates derived from the UCI dataset and must not be interpreted as medical
> diagnoses. The system never states "You have cervical cancer." It uses
> language such as "The model estimates an elevated probability..." **Always
> consult a qualified healthcare professional for clinical evaluation.**

---

## Features

1. **Model Comparison Dashboard** — Compare 6 classification models with
   comprehensive evaluation metrics, visualizations, and SHAP explainability
2. **Interactive Patient Risk Assessment** — Enter patient features and
   receive model-based probability estimates with per-prediction SHAP
   waterfall explanations
3. **Full Explainability** — SHAP feature importance for every model,
   beeswarm plots, waterfall plots, and cross-model comparison heatmaps

## Models

| Model | Class Imbalance Handling | Type |
|:---|:---|:---|
| Logistic Regression | `class_weight='balanced'` | Linear baseline |
| Random Forest | `class_weight='balanced'` | Ensemble (bagging) |
| MLPClassifier | None (documented limitation) | Neural network |
| XGBoost | `scale_pos_weight` | Gradient boosting |
| LightGBM | `is_unbalance=True` | Gradient boosting |
| CatBoost | `auto_class_weights='Balanced'` | Gradient boosting |

## Results Summary

### Cross-Validation (Training Data — 686 samples)

| Model | PR-AUC | ROC-AUC | F1 | Recall | Precision |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.7575** | 0.9496 | 0.7380 | 0.8417 | 0.6581 |
| XGBoost | 0.7327 | 0.9447 | 0.6783 | 0.7944 | 0.6008 |
| CatBoost | 0.7290 | 0.9450 | 0.7258 | 0.8861 | 0.6186 |
| LightGBM | 0.6989 | 0.9437 | 0.6951 | 0.8861 | 0.5774 |
| Logistic Regression | 0.6962 | 0.9621 | 0.6971 | 0.8639 | 0.5869 |
| MLPClassifier | 0.1802 | 0.5506 | 0.0364 | 0.0222 | 0.1000 |

**Best model by CV PR-AUC: Random Forest (0.7575)**

### Final Test Set (172 samples — used exactly ONCE)

| Model | PR-AUC | ROC-AUC | Recall | TP | FN |
|:---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.6912 | 0.9136 | 0.8182 | 9/11 | 2 |
| Random Forest | 0.6897 | 0.9481 | 0.7273 | 8/11 | 3 |
| LightGBM | 0.6887 | 0.9588 | 0.8182 | 9/11 | 2 |
| CatBoost | 0.6603 | 0.9283 | 0.8182 | 9/11 | 2 |
| XGBoost | 0.6479 | 0.9548 | 0.7273 | 8/11 | 3 |
| MLPClassifier | 0.0696 | 0.5031 | 0.0000 | 0/11 | 11 |

> **Note:** Only 11 positive cases in the test set. A single misclassification
> changes Recall by ~9%. Cross-validation results are more reliable for model
> comparison.

### Key Findings

1. **Schiller test** is the dominant predictor across all models (confirmed by
   both Mutual Information analysis and SHAP)
2. **Clinical screening tests** (Schiller, Hinselmann, Citology) carry the most
   predictive signal — models are learning clinically meaningful patterns
3. **MLPClassifier failed** to detect any cancer cases — demonstrating why native
   class-weight support matters with 14.6:1 class imbalance
4. **Random Forest** achieved the best CV PR-AUC (0.7575) and is the recommended
   model for this dataset

---

## Pipeline Architecture

```
data/raw/risk_factors_cervical_cancer.csv
    |
    v
src/split.py ──> data/splits/train.csv + test.csv
    |              (80/20 stratified, random_state=42)
    v
src/preprocess.py ──> data/processed/X_train.csv, X_test.csv, ...
    |                  (pipeline fitted on train ONLY)
    |                  artifacts/preprocessing_pipeline.joblib
    v
src/train.py ──> models/*.joblib
    |             (5-fold stratified CV, PR-AUC selection)
    |             artifacts/metrics/cv_results.json
    v
src/test.py ──> artifacts/metrics/test_results.json
    |            artifacts/figures/*.png (confusion, ROC, PR, comparison)
    v
src/explain.py ──> artifacts/explainability/*.npy, *.png
    |               (SHAP values, bar, beeswarm, waterfall, heatmap)
    v
app.py ──> Streamlit Dashboard (http://localhost:8501)
```

## Leakage Prevention

The following safeguards are implemented and verified:

- **Train/test split** performed **BEFORE** any preprocessing
- **Preprocessing pipeline** fitted on training data **ONLY**
- **Test data** transformed using the already-fitted pipeline (no refitting)
- **Cross-validation** used only training data
- **Test set** used exactly **ONCE** for final evaluation
- **No model retraining** based on test results
- **Feature selection analysis** performed on training data only

## Preprocessing

1. **Missing values:** `?` markers converted to `NaN`
2. **Dropped columns:** `STDs: Time since first diagnosis` and
   `STDs: Time since last diagnosis` (91.7% missing — imputation would
   fabricate data)
3. **Continuous features (10):** Median imputation + StandardScaler
4. **Binary features (23):** Most-frequent imputation
5. **Implementation:** `sklearn.compose.ColumnTransformer` pipeline saved as
   `artifacts/preprocessing_pipeline.joblib`

## Class Imbalance

- **Training set:** 44 positive / 642 negative (6.41% prevalence, 14.59:1 ratio)
- **Strategy:** Native class weighting in 5 of 6 models
- **SMOTE rejected:** Only 44 positive samples — too few for reliable synthetic
  minority generation
- **MLPClassifier limitation:** Does not support `class_weight` parameter.
  Documented as a known limitation. Its poor performance (PR-AUC 0.18) confirms
  the importance of class-weight handling.
- **Primary metric:** PR-AUC chosen over accuracy because accuracy is misleading
  with imbalanced classes (a naive "always negative" classifier achieves 93.6%
  accuracy)

## Feature Selection

Four methods were investigated on the training data:

1. **Correlation Filtering:** 7 highly correlated pairs identified (mostly
   STD-related redundancy, r up to 0.984)
2. **Mutual Information:** Schiller (0.135), Hinselmann (0.070), Citology (0.023)
   are the top features — all clinical screening tests
3. **RFE (Logistic Regression):** Selected 15 of 33 features

**Decision:** All 33 features retained because:
- 20.8:1 sample-to-feature ratio is adequate
- Tree models handle irrelevant features internally
- Only 44 positive cases — removing features risks losing subtle signal
- SHAP provides post-hoc importance (Segment 8)

## Reproducibility

All random operations use `random_state=42`. The full pipeline can be
reproduced from a clean state:

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (order matters)
python src/split.py              # Step 1: Stratified split
python src/preprocess.py         # Step 2: Preprocess (fit on train only)
python src/train.py              # Step 3: Train 6 models (CV on train only)
python src/test.py               # Step 4: Final test evaluation
python src/explain.py            # Step 5: SHAP explainability

# Launch dashboard
streamlit run app.py             # Step 6: Interactive dashboard

# Verify pipeline (optional)
python src/test_preprocessing.py # Milestone test: 8/8 checks
python src/integration_test.py   # Full end-to-end reproducibility test
```

**Total pipeline time:** ~67 seconds on a standard machine.

## Project Structure

```
DA Lab Project/
|-- app.py                          # Streamlit dashboard
|-- requirements.txt                # Python dependencies
|-- README.md                       # This file
|-- data/
|   |-- raw/
|   |   |-- risk_factors_cervical_cancer.csv
|   |-- splits/
|   |   |-- train.csv               # 686 samples (80%)
|   |   |-- test.csv                # 172 samples (20%)
|   |-- processed/
|       |-- X_train.csv, X_test.csv
|       |-- y_train.csv, y_test.csv
|       |-- feature_names.txt
|-- src/
|   |-- __init__.py
|   |-- config.py                   # Centralised constants and paths
|   |-- split.py                    # Stratified train/test split
|   |-- preprocess.py               # ColumnTransformer pipeline
|   |-- train.py                    # 6-model training with CV
|   |-- test.py                     # Final test set evaluation
|   |-- explain.py                  # SHAP explainability
|   |-- test_preprocessing.py       # Milestone validation test
|   |-- integration_test.py         # End-to-end reproducibility test
|-- models/
|   |-- LogisticRegression.joblib
|   |-- RandomForest.joblib
|   |-- MLPClassifier.joblib
|   |-- XGBoost.joblib
|   |-- LightGBM.joblib
|   |-- CatBoost.joblib
|-- artifacts/
    |-- preprocessing_pipeline.joblib
    |-- metrics/
    |   |-- cv_results.json
    |   |-- cv_summary.csv
    |   |-- test_results.json
    |   |-- test_summary.csv
    |-- figures/
    |   |-- confusion_matrices.png
    |   |-- roc_curves.png
    |   |-- pr_curves.png
    |   |-- metric_comparison.png
    |-- explainability/
        |-- shap_values_*.npy
        |-- shap_bar_*.png
        |-- shap_beeswarm_RandomForest.png
        |-- shap_waterfall_*.png
        |-- shap_importance_comparison.png
        |-- shap_background.npy
```

## Dependencies

- Python 3.10+
- scikit-learn, pandas, numpy, matplotlib, seaborn
- xgboost, lightgbm, catboost
- shap
- streamlit
- joblib

See `requirements.txt` for exact versions.

## Limitations

1. **Small dataset:** 858 records with only 55 positive cases limits model
   generalisability
2. **High missingness:** 2 features dropped (91.7% missing), others imputed
3. **MLPClassifier:** Cannot handle class imbalance natively — performs poorly
4. **Single dataset:** Results may not generalise to other populations
5. **Not a clinical tool:** Educational project only

---

*University Data Analytics Lab Project — Cervical Cancer Risk Factor Analysis*
