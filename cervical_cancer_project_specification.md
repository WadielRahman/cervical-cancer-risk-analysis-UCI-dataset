# Cervical Cancer Patient Data Analysis — Google Antigravity Project Specification

## 1. Project Purpose

This document defines the implementation requirements for **Cervical Cancer Patient Data Analysis**. Google Antigravity should use it as the primary specification when inspecting, creating, refactoring, or validating the project.

This is a university **Data Analytics Laboratory** project in the Healthcare domain. The objective is to build an end-to-end machine-learning application that analyzes cervical-cancer-related patient risk factors, compares classification models, and provides interactive, explainable risk predictions.

The system has two major user-facing capabilities:

1. **Model Prediction and Comparison Dashboard** — a dashboard showing model performance, predictions, comparative metrics, feature importance, and explainability.
2. **Interactive Patient Risk Assessment** — an interface where a user answers questions corresponding to available patient risk factors and receives a model-based probability/risk prediction.

The application should be modular, reproducible, understandable, and easy to extend.

> **Medical positioning:** This is a machine-learning/data-analytics decision-support prototype, not a clinical diagnostic system. Predictions must be presented as model-based estimates from the dataset and must not be stated as definitive diagnoses.

---

## 2. Dataset and Project Context

The project uses the **UCI Cervical Cancer Risk Factors dataset**:

https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

The dataset contains patient-level risk-factor information and diagnostic outcomes. It supports demonstrations of data cleaning, exploratory analysis, feature selection, binary classification, probability prediction, model comparison, and explainable AI.

The course guideline requires discussion of dataset provenance, feature count, row count, data quality, preprocessing, EDA, feature selection, at least four ML/DL models, and an interactive dashboard.

There is an important dataset-compliance issue: the UCI dataset contains **858 records**, while the course guideline states a minimum requirement of **2,000 rows**. This must not be hidden, solved by duplicating observations, or artificially inflated. Continue using the UCI dataset for this project as instructed, but remain transparent about its actual size. Do not fabricate additional observations.

Preserve the original feature meanings and target definitions. Do not rename a medical outcome in a way that implies a stronger clinical interpretation than the source supports.

---

## 3. Core Architecture

The implementation must be separated into four principal Python files:

```text
split.py
preprocess.py
train.py
test.py
```

The intended workflow is:

```text
Raw Dataset
     |
     v
split.py
     |
     +--------------------+
     |                    |
 Training Set         Test Set
     |                    |
     v                    |
preprocess.py             |
     |                    |
 Fit preprocessing        |
 on TRAIN ONLY            |
     |                    |
 Transform train          |
 and test using           |
 train-fitted parameters  |
     |                    |
     v                    |
train.py                  |
     |                    |
 Train models             |
 Cross-validation         |
 Hyperparameter tuning    |
     |                    |
     v                    |
 Saved models ------------+
     |
     v
test.py
     |
     v
Final evaluation
     |
     v
Streamlit Dashboard
     |
     +--> Model comparison
     +--> Prediction
     +--> Risk assessment
     +--> SHAP explanations
```

Each file should have a clearly defined responsibility. Avoid placing the entire workflow into one notebook or one large script.

---

## 4. CRITICAL RULE: Split Before Standardization

This is the most important implementation requirement.

**The dataset MUST be split into training and testing data BEFORE standardization or any other learned preprocessing step.**

Correct order:

```text
Raw data
   ↓
Train/Test Split
   ↓
Fit preprocessing ONLY on training data
   ↓
Transform training data
   ↓
Transform test data using the SAME fitted preprocessing
   ↓
Train models
   ↓
Evaluate on untouched test data
```

Incorrect order:

```text
Raw data
   ↓
StandardScaler.fit_transform(all data)
   ↓
Train/Test Split
```

The incorrect approach causes **data leakage**, because information from the test set is used when calculating preprocessing parameters.

For numerical standardization, the scaler should learn the training-set mean and standard deviation only. The test set should then be transformed using those learned values.

Prefer scikit-learn `Pipeline` and `ColumnTransformer` where appropriate. The code should nevertheless remain understandable enough for a student to explain during a viva.

The split should be **stratified** for the binary target where appropriate because positive cases are relatively limited.

---

## 5. `split.py`

`split.py` is responsible only for creating reproducible training and testing datasets.

Responsibilities:

- Load the raw dataset.
- Identify the target variable.
- Separate features (`X`) and target (`y`).
- Perform a stratified train/test split.
- Use a fixed `random_state`.
- Save train and test datasets to clear locations.
- Print useful information such as shapes and target distributions.

Conceptually:

```text
load raw CSV
      ↓
identify target
      ↓
X = features
y = target
      ↓
train_test_split(..., stratify=y, random_state=...)
      ↓
save train.csv
save test.csv
```

Do not standardize, fit imputers, fit encoders, perform target-informed feature selection, or train a model in `split.py`.

---

## 6. `preprocess.py`

`preprocess.py` is responsible for data preparation.

Planned preprocessing includes:

- Missing-value treatment
- Data-type handling
- Categorical encoding
- Numerical feature transformation
- Outlier analysis/handling where justified
- Standardization of appropriate numerical variables
- Creation of a reusable preprocessing pipeline

Most importantly, preprocessing must be **fit on training data only**.

If imputation is required, imputation statistics must be learned from the training set. If an encoder is fitted, it must be fitted using training data. If a scaler is fitted, it must be fitted using training data.

The test set should only be transformed.

A reusable preprocessing object should be saved when appropriate so that the exact same transformations can later be applied to user inputs in the Streamlit application.

---

## 7. `train.py`

`train.py` is responsible for model development.

The project will compare:

1. **Logistic Regression** — baseline.
2. **Random Forest**
3. **XGBoost**
4. **LightGBM**
5. **CatBoost**
6. **MLPClassifier**
7. **SVM** — optional benchmark if computationally practical.

**MLPClassifier is a required model in the comparison**, not merely a possible future extension.

The Logistic Regression baseline provides an interpretable reference point. Tree ensembles provide nonlinear modeling capability. MLPClassifier provides a neural-network-based benchmark, while SVM can serve as an additional conventional classifier.

Do not assume that the most complex model is automatically the best model.

---

## 8. Feature Selection and Importance

The project will investigate:

- Correlation filtering
- Mutual Information
- Recursive Feature Elimination (RFE)
- SHAP-based feature importance

The project should compare:

```text
All/approved features
        VS
Reduced feature set
```

The purpose is to determine whether removing weak or redundant variables improves performance, interpretability, or efficiency.

Feature selection must also avoid leakage. If a method learns information from the target, it must be performed inside the appropriate training/cross-validation process rather than using the complete dataset before evaluation.

SHAP can provide model explanation and feature importance. Test-set information must not be used to select features before final evaluation.

---

## 9. Validation and Hyperparameter Optimization

Model development should use **stratified cross-validation** and **hyperparameter tuning** where appropriate.

General workflow:

```text
Training data
     ↓
Cross-validation
     ↓
Hyperparameter search
     ↓
Best model
     ↓
Final evaluation on untouched test set
```

The search space should be computationally reasonable for a relatively small dataset. Avoid unnecessarily huge searches.

The final test set must remain isolated until final evaluation.

---

## 10. Evaluation Metrics

Accuracy must not be the only metric.

The project will report:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Brier Score

A confusion matrix should also be available.

Multiple metrics are important because the target distribution is not perfectly balanced and positive outcomes are especially important in a risk-screening context.

Interpretation:

- **Precision:** Of predicted positives, how many were actually positive?
- **Recall:** Of actual positives, how many did the model identify?
- **F1:** Balance between precision and recall.
- **ROC-AUC:** Measures discrimination across thresholds.
- **PR-AUC:** Particularly informative when the positive class is uncommon.
- **Brier Score:** Measures the quality of predicted probabilities.

Where practical, evaluate probability calibration. Calibrated probabilities are particularly relevant because the application will display risk/probability estimates.

---

## 11. `test.py`

`test.py` evaluates saved trained models on the untouched test set.

Responsibilities:

- Load the fitted preprocessing object.
- Load trained models.
- Transform test data using training-fitted preprocessing.
- Generate predictions and probabilities.
- Calculate all agreed evaluation metrics.
- Generate confusion matrices and evaluation artifacts.
- Save results in machine-readable formats such as JSON/CSV.
- Produce data that the dashboard can display.

The test script must never refit the scaler, imputer, encoder, feature selector, or model using the test set.

---

## 12. Dashboard — Feature 1: Model Prediction and Comparison

Build the dashboard using **Streamlit**.

The first major feature is a model-performance and prediction interface.

Suggested sections:

### Overview
- Project title
- Dataset description
- Medical-use disclaimer
- Number of models evaluated

### Model Comparison

Display a comparison table:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Random Forest | | | | | | |
| XGBoost | | | | | | |
| LightGBM | | | | | | |
| CatBoost | | | | | | |
| MLPClassifier | | | | | | |
| SVM | | | | | | |

SVM may be omitted from final deployment if it does not add sufficient value, but MLPClassifier must remain in the model comparison.

### Prediction View

The user should be able to select a model and provide an input record. Display:

- Predicted class
- Predicted probability
- Probability visualization
- Relevant feature contributions

### Explainability

Use SHAP for:

- Global feature importance
- Local explanation of an individual prediction
- Positive and negative feature contributions

---

## 13. Dashboard — Feature 2: Interactive Patient Risk Assessment

The second major feature is a simple question-and-answer interface.

Instead of asking users to manipulate raw column names, present understandable questions corresponding to actual dataset variables.

Example structure:

```text
Age: [input]
Smoking history: [Yes/No]
Number of pregnancies: [input]
Hormonal contraceptive use: [Yes/No]
IUD use: [Yes/No]
STD history: [Yes/No]
...
```

The exact questions must be derived from the actual dataset schema. Do not invent variables that do not exist.

After submission:

```text
User answers
     ↓
Construct feature record
     ↓
Apply saved preprocessing
     ↓
Send to selected trained model(s)
     ↓
Generate prediction probability
     ↓
Display result
     ↓
Generate SHAP explanation
```

Use careful wording. Do not tell a user:

> “You have cervical cancer.”

Prefer:

> “The model estimates an elevated probability based on the information provided.”

The application should clearly encourage professional medical evaluation and must not imply that the model replaces screening or diagnosis.

---

## 14. Explainable AI

Explainability is a core requirement, not merely a visual enhancement.

SHAP should be integrated into the prediction workflow so users can understand why a model produced a particular result.

For an individual prediction, show the most influential features.

Conceptual output:

```text
Model-estimated probability: 0.72

Factors influencing this prediction:
+ Factor A
+ Factor B
- Factor C
- Factor D
```

The actual factors must be generated from the model and input record, not hard-coded.

SHAP contributions should not be described as causal effects. A feature influencing a model prediction does not prove that it causes cervical cancer.

---

## 15. Recommended Project Structure

Aim for a structure similar to:

```text
cervical-cancer-analysis/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── models/
│
├── artifacts/
│   ├── metrics/
│   ├── figures/
│   └── explainability/
│
├── src/
│   ├── split.py
│   ├── preprocess.py
│   ├── train.py
│   └── test.py
│
├── app.py
├── requirements.txt
├── README.md
└── project_specification.md
```

Adapt the exact structure to the existing repository if necessary, but preserve modular responsibilities.

---

## 16. Engineering Requirements for Google Antigravity

When modifying or generating the project:

1. Inspect the existing repository before making changes.
2. Do not unnecessarily overwrite working code.
3. Preserve reproducibility.
4. Use deterministic random seeds where appropriate.
5. Keep data paths configurable rather than hard-coded to one computer.
6. Do not duplicate observations to satisfy the course row requirement.
7. Do not introduce data leakage.
8. **Never standardize before train/test splitting.**
9. Ensure the same preprocessing pipeline is used for training, testing, and user-entered dashboard data.
10. Save trained models and preprocessing artifacts in a reproducible format.
11. Keep model-specific code organized.
12. Include clear error handling for missing files and invalid input.
13. Validate that user input matches the expected feature schema.
14. Do not silently drop features.
15. Document important assumptions.
16. Make the Streamlit interface understandable to a non-programmer.
17. Clearly distinguish prediction from clinical diagnosis.
18. Include MLPClassifier in training, evaluation, and model comparison.
19. Ensure probability-producing models expose probabilities consistently; use an appropriate probability strategy for models that do not natively provide them.
20. Keep the final test set untouched until the final evaluation stage.

---

## 17. Final End-to-End Workflow

```text
                 RAW UCI DATASET
                       |
                       v
                  split.py
                       |
              +--------+--------+
              |                 |
          TRAIN SET          TEST SET
              |                 |
              v                 |
        preprocess.py           |
              |                 |
       TRAIN-FITTED             |
       preprocessing            |
              |                 |
              v                 |
           train.py             |
              |                 |
       +------+------+------+---+
       |      |      |      |
      LR     RF     XGB    LGBM
       |      |      |      |
     CatBoost MLPClassifier SVM
              |
              v
       Cross-validation
       + Hyperparameter tuning
              |
              v
          Best models
              |
              v
            test.py
              |
              v
       Final test metrics
              |
              v
       Streamlit Dashboard
          /                    /             Model comparison    Patient Q&A
       |                 |
       v                 v
Predictions        Risk probability
       |                 |
       +--------+--------+
                |
                v
          SHAP Explanation
```

The final application should demonstrate a complete data-science lifecycle:

**data splitting → leakage-safe preprocessing → feature selection → model training → cross-validation → hyperparameter tuning → unbiased testing → probability prediction → explainability → interactive deployment.**

The central technical principle is non-negotiable:

> **Split first. Fit preprocessing only on training data. Transform the test data afterward using the training-fitted preprocessing.**

This principle must also be preserved when building the Streamlit patient-risk interface.

## 18. Success Criteria

Google Antigravity should consider the implementation successful when:

- `split.py`, `preprocess.py`, `train.py`, and `test.py` have clear responsibilities.
- The dataset is split before standardization.
- No preprocessing leakage occurs.
- Training and testing transformations are reproducible.
- Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and MLPClassifier are trained and compared.
- SVM is included when practical.
- Cross-validation and hyperparameter tuning are implemented appropriately.
- The final test set remains unseen during training and model-selection decisions.
- Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier Score are available.
- SHAP explanations can be generated for relevant predictions.
- The Streamlit dashboard compares model performance.
- The dashboard accepts patient-style question/answer input.
- The same saved preprocessing pipeline is applied to dashboard input.
- The application displays model-based risk/probability rather than claiming a definitive diagnosis.
- The codebase is reproducible, documented, and suitable for university demonstration and viva examination.
