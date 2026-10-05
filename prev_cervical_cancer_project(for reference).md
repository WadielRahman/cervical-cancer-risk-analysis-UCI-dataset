# Integrated Cervical Cancer Data Platform
### Group 8 — Research Lab Project | United International University
**Dataset:** UCI Cervical Cancer Risk Factors (858 patients, Hospital Universitario de Caracas)  
**Program:** B.Sc. in Data Science (BSDS) | UIU  
**Supervised by:** RBDCSC Research Group

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Clinical Background](#2-clinical-background)
3. [Team Roles and Responsibilities](#3-team-roles-and-responsibilities)
4. [Dataset Description](#4-dataset-description)
5. [System Architecture](#5-system-architecture)
6. [Three-Stage Screening Framework](#6-three-stage-screening-framework)
7. [Member 1 — Data Acquisition Pipeline](#7-member-1--data-acquisition-pipeline)
8. [Member 2 — Preprocessing and Feature Engineering](#8-member-2--preprocessing-and-feature-engineering)
9. [Member 3 — Machine Learning and Interpretability](#9-member-3--machine-learning-and-interpretability)
10. [Member 4 — Dashboard and Deployment](#10-member-4--dashboard-and-deployment)
11. [Model Results Summary](#11-model-results-summary)
12. [Feature Importance (LightGBM)](#12-feature-importance-lightgbm)
13. [Clustering Analysis](#13-clustering-analysis)
14. [Known Issues and Audit Findings](#14-known-issues-and-audit-findings)
15. [Research Novelty and Contributions](#15-research-novelty-and-contributions)
16. [Related Literature](#16-related-literature)
17. [File Structure](#17-file-structure)
18. [How to Run](#18-how-to-run)
19. [Limitations](#19-limitations)
20. [Future Work](#20-future-work)
21. [References](#21-references)

---

## 1. Project Overview

The **Integrated Cervical Cancer Data Platform (ICCDP)** is a multi-tier clinical analytics and predictive modeling system built for the UIU Data Science lab. Its primary objective is to evaluate demographic, lifestyle, and clinical parameters to predict the presence of cervical cancer, with ground-truth pathology confirmed via tissue **Biopsy**.

The platform is motivated by a real public health problem: over 340,000 women die from cervical cancer globally each year, with the vast majority of deaths occurring in low- and middle-income countries — including in South and Southeast Asia — where laboratory infrastructure and specialist clinicians are scarce.

> **Core research question:** Can a machine learning system trained on routinely collectable survey data and low-cost clinical tests provide a clinically safe, explainable triage pathway for women in resource-constrained settings?

### What the platform does

| Component | Description |
|:---|:---|
| **Data Platform** | Ingests UCI risk factors data, WHO vaccination rates, and World Bank health indicators; cleans and integrates them into a unified panel |
| **ML System** | Trains, calibrates, and benchmarks 7 classifier architectures across 3 distinct screening workflows |
| **Clinical Dashboard** | Streamlit web application for real-time patient risk scoring, SHAP explanations, and threshold adjustment by clinicians |

---

## 2. Clinical Background

### What is cervical cancer?

The cervix is the lower, narrow part of the uterus connecting it to the vagina. Cervical cancer occurs when cells in the cervical lining grow uncontrolled. Nearly all cases are caused by long-term infection with **Human Papillomavirus (HPV)** — specifically high-risk strains **HPV-16** and **HPV-18**.

The disease progresses slowly (10–20 years from normal → precancerous lesion → invasive cancer), providing a long screening window.

### Screening methods used in this project

| Test | Method | Resource requirement |
|:---|:---|:---|
| **Pap Smear (Cytology)** | Cells scraped from cervix, examined under microscope | Lab + cytopathologist |
| **Schiller Test** | Cervix painted with iodine; abnormal tissue does not stain brown | Clinic-level |
| **Hinselmann Test** | Colposcopic inspection under magnification with acetic acid | Clinic-level |
| **Biopsy** | Tissue removal for definitive pathological diagnosis | Hospital + pathologist |

### The low-resource problem

In rural areas across South Asia, community clinics may have:
- No laboratory or microscope
- A single nurse with no specialist training
- No electronic patient records

This project designs its triage system to operate at each level of available resources — beginning with survey questions only (Model A), escalating to clinical records (Model B), and finally to active screening tests (Model C).

---

## 3. Team Roles and Responsibilities

| Member | Role | Key deliverables |
|:---|:---|:---|
| **Wadiel Rahman** | ML Engineering, Feature Selection, XAI | Modeling pipeline, SHAP explainability, calibration, audit |
| **Nishat Jahan** | Data Preprocessing, Feature Engineering | UCI dataset cleaning, KNN imputation, outlier detection, encoding |
| **An-Nafee** | Results & Deployment | Streamlit dashboard, model serialization, threshold optimizer |
| **Member 1** | Global Data Acquisition | WHO/World Bank API ingestion, country panel scaffold |

> **Note:** File-level mapping uses `member1`–`member4` labels in the source tree; the role assignments above correspond to actual team members.

---

## 4. Dataset Description

### Primary dataset: UCI Cervical Cancer Risk Factors

- **Source:** Fernandes, Cardoso & Fernandes (2017), Hospital Universitario de Caracas, Venezuela
- **Records:** 858 patients
- **Features:** 35 input variables (behavioral, demographic, STD history, diagnostic flags)
- **Target variable:** `Biopsy` (1 = cancer confirmed; 0 = healthy)
- **Missing values:** ~22% of cells contain `?` (left blank by patients due to privacy concerns)
- **DOI:** `10.24432/C5Z310`

### Class distribution

| Class | Count | Percentage |
|:---|:---|:---|
| Healthy (`Biopsy = 0`) | 803 | 93.6% |
| Cancer (`Biopsy = 1`) | 55 | 6.4% |

> ⚠️ **Severe class imbalance** — a model that predicts "Healthy" for every patient achieves 93.6% accuracy while missing 100% of cancer cases. Standard accuracy is therefore a misleading metric.

### Key features

| Feature | Type | Clinical meaning |
|:---|:---|:---|
| `Age` | Continuous | Patient age in years |
| `First sexual intercourse` | Continuous | Age at sexual debut |
| `Num of pregnancies` | Continuous | Parity count |
| `Smokes (years)` | Continuous | Duration of smoking |
| `Hormonal Contraceptives (years)` | Continuous | Duration of pill use |
| `IUD (years)` | Continuous | Duration of IUD use |
| `STDs (number)` | Continuous | Total STD diagnoses |
| `Dx:HPV` | Binary | Prior HPV diagnosis |
| `Dx:Cancer` | Binary | Prior cancer diagnosis |
| `Cytology` | Binary | Pap smear result |
| `Schiller` | Binary | Schiller iodine test result |
| `Hinselmann` | Binary | Colposcopy result |

### Supplementary datasets

| Dataset | Source | Purpose |
|:---|:---|:---|
| HPV vaccination coverage | WHO OData API | Global policy context |
| Health indicators (GDP, health expenditure) | World Bank API | Socioeconomic context layer |
| Scientific literature | Europe PMC | RAG-powered search assistant |

---

## 5. System Architecture

```
[Raw Data Sources: WHO API, World Bank API, UCI CSV, Europe PMC]
                           │
                           ▼
           ┌─────── Acquisition & Parsing ────────┐
           │                                       │
           ▼                                       ▼
   [UCI Risk Factors CSV]             [WHO/WB Country Panel]
           │                                       │
           ▼                                       ▼
   [KNN Imputation (k=5)]           [Scaffold (ISO3 × Year)]
   [Consensus Outlier Detection]    [Duplicate Aggregation]
   [Z-Score Standardization]        [Log Transform / Encoding]
           │                                       │
           └─────────────┬─────────────────────────┘
                         ▼
               [Integrated Master Dataset]
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      [Model A]      [Model B]      [Model C]
   (Behavioral)  (+ Dx History)  (+ Screenings)
          │              │              │
          └──────────────┼──────────────┘
                         ▼
          [Stratified 5-Fold CV + SMOTE]
                         ▼
          [7 Classifier Benchmark Suite]
                         ▼
          [Isotonic Probability Calibration]
                         ▼
          [Youden & Recall Threshold Optimization]
                         ▼
          [Interactive Streamlit Dashboard :8501]
```

---

## 6. Three-Stage Screening Framework

The platform maps to real-world clinical escalation — each model is designed for a specific resource environment.

```
[Patient enters system]
        │
        ▼
┌────────────────────────────────────────────────────┐
│  MODEL A — Community Screening                     │
│  Input: Age, smoking, pregnancies, contraceptives  │
│  Threshold: 1.0%   ROC-AUC: 0.6429                │
│  Purpose: Rule-out triage in rural/mobile camps    │
└───────────────────────┬────────────────────────────┘
     Risk < 1.0%        │        Risk ≥ 1.0%
         ▼              │             ▼
    [Discharge]         │    Continue to Model B
                        │
                        ▼
┌────────────────────────────────────────────────────┐
│  MODEL B — Clinical History                        │
│  Input: Model A features + Dx:HPV, Dx:Cancer      │
│  Threshold: 16.7%  ROC-AUC: 0.6576                │
│  Purpose: Secondary triage using EHR flags        │
└───────────────────────┬────────────────────────────┘
     Risk < 16.7%       │        Risk ≥ 16.7%
         ▼              │             ▼
  [Follow-up in         │    Continue to Model C
   6–12 months]         │
                        ▼
┌────────────────────────────────────────────────────┐
│  MODEL C — Clinical Decision Support               │
│  Input: Model A+B features + Cytology, Schiller,  │
│           Hinselmann                               │
│  Threshold: 16.7%  ROC-AUC: 0.9271                │
│  Purpose: Biopsy referral decision                 │
└───────────────────────┬────────────────────────────┘
     Risk < 16.7%       │        Risk ≥ 16.7%
         ▼              │             ▼
  [Active monitoring]   │   [Immediate biopsy referral]
```

### Why performance jumps from Model A to Model C

Model A uses only behavioral features. Healthy and cancer patients frequently share identical behavioral profiles (same age, parity, smoking history), making them mathematically inseparable from survey data alone. Adding active clinical screening tests (Cytology, Schiller, Hinselmann) in Model C directly measures tissue pathology, resolving this overlap and boosting ROC-AUC from 0.64 to 0.93.

---

## 7. Member 1 — Data Acquisition Pipeline

**File:** `src/member1_global_stats/acquire.py`, `clean.py`, `integrate.py`

### Techniques implemented

| # | Technique | Function | Lines | Purpose |
|:---|:---|:---|:---|:---|
| 1 | RESTful API ingestion | `_get` | 49–52 | Download WHO & WB JSON payloads programmatically |
| 2 | Country scaffold (MultiIndex) | `build_panel` | 92–110 | Create dense ISO3 × Year grid (2000–2023) |
| 3 | Aggregate filter | `load_country_meta` | 30–42 | Remove regional groupings (e.g. "World", "South Asia") |
| 4 | Duplicate key consolidation | `load_who` | 71–73 | Group-mean aggregation on `[iso3, year]` |
| 5 | Ordinal categorical encoding | `transform` | 154–160 | Convert `income_group` text → integer codes |

**Output files:**
- `data/raw/who/` — raw WHO JSON
- `data/raw/worldbank/` — raw World Bank JSON
- `data/clean/global_panel.csv` — integrated country-year panel

---

## 8. Member 2 — Preprocessing and Feature Engineering

**File:** `src/member3_riskfactors/clean.py`

### Technique 1: KNN Imputation vs. MICE

Missing values (22% of cells) are filled using KNN Imputation:

$$x_{ij} = \frac{\sum_{k \in N_i} w_k x_{kj}}{\sum_{k \in N_i} w_k}$$

KNN is compared against MICE (Multiple Imputation by Chained Equations) using `compare_imputation()` (lines 60–87). KNN with `k=5` was selected as it preserves multivariate relationships without distorting feature distributions.

> **Why not mean imputation?** Mean imputation treats each column independently, collapsing variance and destroying correlations between features like `Age` and `Num of pregnancies`.

| Method | File | Function | Lines |
|:---|:---|:---|:---|
| KNN Imputer (k=5) | `clean.py` | `compare_imputation` | 69 |
| MICE Iterative Imputer | `clean.py` | `compare_imputation` | 70 |

### Technique 2: Consensus Outlier Detection

Four classifiers vote on each patient record. A record is flagged only if **≥ 2 methods agree**:

$$\text{Outlier} = \mathbb{I}\left( \sum_{m=1}^{4} \text{Flag}_m \ge 2 \right)$$

| Method | Logic |
|:---|:---|
| Z-Score | $\|(x_i - \mu)/\sigma\| > 3$ |
| IQR | $x_i < Q_1 - 1.5\times\text{IQR}$ or $x_i > Q_3 + 1.5\times\text{IQR}$ |
| One-Class SVM | Hyper-boundary around dense normal region |
| Local Outlier Factor (LOF) | Local density deviation relative to neighbors |

**Output column:** `outlier_consensus` (binary flag)  
**Function:** `detect_outliers()` — lines 93–120

### Technique 3: Age Binning

$$f(\text{Age}) = \begin{cases} \text{"≤20"} & \text{if Age} \le 20 \\ \text{"21-30"} & \text{if } 20 < \text{Age} \le 30 \\ \text{"31-40"} & \text{if } 30 < \text{Age} \le 40 \\ \text{"40+"} & \text{if Age} > 40 \end{cases}$$

Aligns with clinical screening intervals (guidelines shift at age 30). Creates `age_band` and `age_band_code` (0–3).

### Technique 4: Z-Score Standardization

$$z = \frac{x - \mu}{\sigma}$$

Applied to all continuous columns, producing `{col}_z` variants. Required for SVM, MLP, and Logistic Regression — tree models (RF, XGBoost, LightGBM) are unaffected by scale but run through the same pipeline for uniformity.

---

## 9. Member 3 — Machine Learning and Interpretability

**File:** `src/member3_riskfactors/advanced_model.py`

### 9.1 Model architectures benchmarked

| Model | Type | Key strength | Lines |
|:---|:---|:---|:---|
| Logistic Regression | Linear | Interpretable baseline | 124 |
| Random Forest | Bagging ensemble | Robust, low overfit | 126 |
| XGBoost | Gradient boosting (depth-wise) | Strong tabular performance | 127 |
| **LightGBM** ✅ | Gradient boosting (leaf-wise) | Fastest, best AUC overall | 128 |
| CatBoost | Gradient boosting (categorical-aware) | Handles binary flags natively | 129 |
| SVM | Margin-based | High-dimensional spaces | 130 |
| MLP | Neural network | Non-linear relationships | 131 |

**Parameters:** `n_estimators=300`, `random_state=42`, `class_weight='balanced'`

### 9.2 Class imbalance resampling

Three strategies compared:

| Method | Mechanism |
|:---|:---|
| **SMOTE** | Generates synthetic minority points along nearest-neighbor lines |
| **Borderline-SMOTE** | Generates points only near the decision boundary |
| **ADASYN** | Density-weighted: more samples in harder-to-learn minority regions |

> ⚠️ **Critical:** all resampling is applied **inside each training fold** of cross-validation. Applying SMOTE before splitting leaks synthetic samples into the validation set, inflating all metrics artificially.

**Function:** `run_imbalance_experiments()` — lines 470–541  
**Parameters:** `random_state=42`, `n_neighbors=5`

### 9.3 Evaluation strategy

- **Primary:** Stratified 5-Fold Cross-Validation (preserves 6.4% class ratio in every fold)
- **Secondary:** Held-out test set (10% of data)
- **Metrics used:** ROC-AUC, PR-AUC, Sensitivity (Recall), Specificity, Accuracy, Brier Score

> **Why Stratified K-Fold over a single split?**  
> With only 55 positive cases, a single 80/10/10 split leaves ~5 cancer cases in the test set. One misclassification changes Sensitivity by 0.20. Stratified K-Fold reduces this variance significantly.

**95% Confidence Intervals:**
$$\text{CI} = \mu \pm 1.96 \times \frac{\sigma}{\sqrt{k}}$$

**Function:** `compute_ci()` — lines 109–113

### 9.4 Statistical significance testing

| Test | Purpose | Use case |
|:---|:---|:---|
| **Paired t-test** | Checks if mean difference between two models is non-zero | When fold scores are approximately normal |
| **Wilcoxon Signed-Rank** | Non-parametric rank-based alternative | When normality cannot be assumed (5 folds only) |

**Significance threshold:** `alpha = 0.05`  
**Function:** `run_statistical_comparison()` — lines 191–231

### 9.5 Probability calibration

Boosting classifiers optimize decision margins — not log-likelihood — so their raw output scores are **not** true probabilities.

| Method | Mechanism |
|:---|:---|
| **Platt Scaling** | Fits a logistic regression curve: $P(y=1) = 1/(1+\exp(Af(x)+B))$ |
| **Isotonic Regression** | Non-parametric monotonic step-function (more flexible, needs more data) |

**Winner:** Isotonic calibration achieved the lowest Brier Score across all three workflows.  
**Model C Brier Score:** `0.0385`  
**Function:** `run_calibration()` — lines 236–324

> **Brier Score** = mean squared error between predicted probability and actual outcome. Lower is better. A score of 0.0 is perfect; 0.25 is a random classifier on balanced data.

### 9.6 Feature selection (multi-method)

| Method | Mechanism | Lines |
|:---|:---|:---|
| **SHAP** | Mean absolute Shapley value across test set | — |
| **Mutual Information** | Non-parametric entropy-based dependency | 420 |
| **RFE** | Iterative elimination of lowest-weighted features | 425 |
| **Boruta** | Compares feature importance against randomized "shadow" features | 433 |

**Output:** `data/integrated/feature_ranking.csv`

> **Boruta shadow features explained:**  
> Boruta duplicates the full dataset, shuffles each column independently (removing all relationships with the target), and uses these shuffled "shadow" copies as a null baseline. A real feature is confirmed important only if it consistently outranks its own shadow.

### 9.7 Explainable AI — SHAP

Shapley values distribute prediction credit fairly across features:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \Big( f(S \cup \{i\}) - f(S) \Big)$$

| Explanation type | Scope | Output |
|:---|:---|:---|
| **Global** | Entire dataset — which features matter most overall | `shap_beeswarm.csv`, `shap_importance.csv` |
| **Local** | Single patient — why this specific prediction was made | Live SHAP waterfall in Streamlit dashboard |

**Explainer used:** `shap.TreeExplainer(base_model)` — lines 685–740  
> TreeExplainer does **not** require PyTorch. Use it for all tree-based models to avoid the Windows `c10.dll` memory error.

### 9.8 Risk stratification (Youden Index)

$$J = \text{Sensitivity} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}$$

The Youden Index identifies the ROC curve point that maximizes the separation between true positive rate and false positive rate — providing a data-driven clinical threshold.

| Threshold | Value | Application |
|:---|:---|:---|
| **Youden optimal** | 16.7% | Model B referral cut-off, Model C biopsy cut-off |
| **Recall-optimized** | 1.0% | Model A conservative rule-out (targets ≥ 90% sensitivity) |

**Function:** `run_risk_stratification()` — lines 330–401  
**Target recall:** `0.90`

---

## 10. Member 4 — Dashboard and Deployment

**File:** `app/dashboard.py`

### Deliverables

| Artifact | Generated by | Used at | Purpose |
|:---|:---|:---|:---|
| `literature_comparison.csv` | `run_literature_comparison` L.555 | Dashboard L.727 | Compare our results against published baselines |
| `screening/hybrid/clinical_benchmark_results.csv` | `run_benchmarks` L.182 | Dashboard L.699–714 | Side-by-side CV metrics for all 3 workflows |
| `final_*_model.joblib` | `train_and_save_final_model` L.591 | Dashboard L.389 | Serialized calibrated pipelines for live inference |
| `screening_thresholds_*.json` | `run_risk_stratification` L.400 | Dashboard L.393–514 | Youden & recall thresholds for slider card |
| `shap_beeswarm.csv` | `run_pipeline` L.731 | Dashboard L.628–688 | Global SHAP data + local SHAP for patient entries |

### Streamlit dashboard features

- **Patient risk calculator** — enter patient parameters, get calibrated risk % per workflow
- **Threshold optimizer** — slider to adjust Youden/recall cut-offs and see sensitivity/specificity trade-off live
- **SHAP waterfall plots** — per-patient local explanations (with `sys.modules['torch']` mock guard for Windows)
- **Literature comparison table** — our results vs. published papers on the same dataset
- **RAG search assistant** — Europe PMC-backed literature Q&A (runs with isolated import guard to avoid memory crashes)

---

## 11. Model Results Summary

### Final evaluation on held-out test set

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.9535 | 0.8000 | 0.9630 | 0.9481 |
| kNN | 0.9186 | 0.8000 | 0.9259 | 0.8630 |
| SVM | 0.9535 | 0.8000 | 0.9630 | N/A ⚠️ |
| XGBoost | 0.9651 | 0.8000 | 0.9753 | 0.9679 |
| **LightGBM** | **0.9767** | **0.8000** | **0.9877** | **0.9802** |
| AdaBoost | 0.9535 | 0.8000 | 0.9630 | 0.8494 |
| GradientBoosting | 0.9419 | 0.4000 ⚠️ | 0.9753 | 0.9086 |
| CatBoost | 0.9651 | 0.6000 | 0.9877 | 0.9457 |

> ⚠️ **SVM AUROC = NaN** because `SVC` was initialized without `probability=True`. Fix: `SVC(probability=True)`.  
> ⚠️ **GradientBoosting Sensitivity = 0.40** — missed 3 of 5 cancer cases. Despite high accuracy, this model is clinically unsafe. High accuracy masks this failure because the majority class dominates the metric.

### Workflow comparison

| Workflow | Model | ROC-AUC | PR-AUC | Brier Score |
|:---|:---|:---:|:---:|:---:|
| Model A — Community Screening | LightGBM | 0.6429 | 0.1333 | — |
| Model B — Clinical History | LightGBM | 0.6576 | 0.1760 | — |
| **Model C — Clinical Decision Support** | **LightGBM** | **0.9271** | **0.5909** | **0.0385** |

---

## 12. Feature Importance (LightGBM)

Top 15 features by LightGBM importance score on the full feature set:

| Rank | Feature | Importance Score |
|:---:|:---|:---:|
| 1 | First sexual intercourse | 319 |
| 2 | Num of pregnancies | 292 |
| 3 | Age | 218 |
| 4 | Hormonal Contraceptives (years) | 143 |
| 5 | Number of sexual partners | 141 |
| 6 | Schiller | 140 |
| 7 | STDs: Time since first diagnosis | 135 |
| 8 | Dx:CIN | 101 |
| 9 | Hinselmann | 69 |
| 10 | Citology | 63 |
| 11 | Smokes | 61 |
| 12 | Hormonal Contraceptives | 54 |
| 13 | Smokes (packs/year) | 53 |
| 14 | STDs: Time since last diagnosis | 52 |
| 15 | Dx | 39 |

**Clinical alignment:** the top 5 features — sexual debut age, parity, patient age, contraceptive duration, and number of partners — are all well-documented HPV exposure risk factors in the clinical literature. This validates that the model is learning genuine biological signal, not noise.

> ⚠️ **Design consideration:** Schiller (rank 6) and Hinselmann (rank 9) are **active clinical test results**, not patient-reportable survey answers. For a pre-screening community tool (Model A), these should be excluded from inputs. The current full-feature model serves as a combined diagnostic benchmark — not a community triage model.

---

## 13. Clustering Analysis

Unsupervised learning was applied to the training set to explore whether cancer patients form naturally separable clusters in feature space.

| Algorithm | Parameters | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | ARI |
|:---|:---|:---:|:---:|:---:|:---:|
| **KMeans** | `n_clusters=2`, `n_init=50` | 0.5181 | 1.7918 | 104.47 | 0.1002 |
| **DBSCAN** | `eps=1.7`, `min_samples=5` | 0.2374 | 2.5038 | 30.72 | 0.1124 |

**Key finding:** Adjusted Rand Index (ARI) ≈ 0.10 for both methods — the clusters found by unsupervised algorithms barely correspond to the actual cancer labels. This is expected and clinically meaningful:

> Cervical cancer risk is not linearly separable in behavioral feature space. Healthy and cancer patients share identical lifestyle profiles. This confirms that **supervised learning is necessary** for this prediction task, and that purely demographic data is insufficient for definitive diagnosis.

**DBSCAN note:** 35.3% noise rate (242 samples classified as outliers) indicates `eps=1.7` is too tight for this 35-dimensional feature space. A k-distance elbow plot should be used to tune `eps` before any future use of DBSCAN here.

---

## 14. Known Issues and Audit Findings

### 🔴 Critical

| # | Issue | Detail | Fix |
|:---|:---|:---|:---|
| 1 | **Tiny test set** | 86 test samples, ~5 positive cases — one misclassification moves Sensitivity by ±0.20 | Switch to Stratified 5-Fold CV as primary evaluation |
| 2 | **SVM AUROC = NaN** | `SVC` initialized without `probability=True` | Add `probability=True`; remove `random_state` (SVC doesn't accept it) |

### 🟡 Medium

| # | Issue | Detail | Fix |
|:---|:---|:---|:---|
| 3 | **LightGBM feature name warning** | SMOTE converts DataFrame to NumPy array, losing column names | Wrap SMOTE output: `pd.DataFrame(X_train_res, columns=X.columns)` |
| 4 | **XGBoost `verbose` warning** | XGBoost uses `verbosity`, not `verbose` | Change to `verbosity=0` |
| 5 | **Clinical test features in pre-screening model** | Schiller/Hinselmann are clinical results, not survey inputs | Run two model variants: all-features vs. survey-only |

### 🟢 Low

| # | Issue | Detail | Fix |
|:---|:---|:---|:---|
| 6 | **DBSCAN noise rate too high** | 35.3% noise at `eps=1.7` | Tune `eps` using k-distance elbow plot |
| 7 | **Local SHAP WinError 8** | SHAP auto-detects model type and tries to import `torch`, triggering `c10.dll` load failure on Windows | Use `shap.TreeExplainer(model)` explicitly — no PyTorch dependency |

### ✅ What was done correctly

- SMOTE applied **inside** CV folds only (no data leakage)
- Sensitivity used as the tuning metric during hyperparameter search (medically correct priority)
- Multi-metric evaluation (Sensitivity, Specificity, AUC, PR-AUC, Brier Score)
- Isotonic calibration applied to convert raw scores to true probabilities
- Youden Index used for threshold selection instead of arbitrary 0.5 default
- TreeExplainer used for SHAP (not KernelExplainer) — appropriate for tree models

---

## 15. Research Novelty and Contributions

The UCI dataset has been used in over a dozen published papers. To be publishable, this work needs contributions beyond rerunning standard models. The following are the genuine novelty angles of this platform:

| Contribution | Novelty |
|:---|:---|
| **Three-stage cascade framework** | Most papers run a single model; this mirrors real clinical escalation pathways |
| **Isotonic probability calibration** | Converts model output to true risk percentages — most papers report raw class predictions |
| **Youden-optimized thresholds** | Evidence-based threshold selection vs. default 0.5 cutoff used in most papers |
| **WHO/WB global panel integration** | Contextualizes the clinical model within SEA policy and vaccination coverage data |
| **Streamlit clinical dashboard** | Deployable, clinician-facing tool — not just a Jupyter notebook |
| **Multi-method feature selection** | SHAP + MI + RFE + Boruta consensus ensures stability of feature rankings |
| **SEA framing** | No published paper using this dataset has explicitly framed it for South/Southeast Asian low-income contexts |

---

## 16. Related Literature

| Paper | Year | Key contribution | Relevance |
|:---|:---|:---|:---|
| Fernandes, Cardoso & Fernandes | 2017 | Created and donated the UCI dataset | **Must cite — the dataset origin paper** |
| Al Mudawi & Alazeb | 2022 | RF, DT, XGBoost, SVM benchmark — reported 100% accuracy (RF/DT) | Benchmark comparison |
| Glučina et al. | 2023 | Class balancing + MLP+KNN+SMOTE (AUC 0.95) | SMOTE comparison baseline |
| Stacked Ensemble + XAI | 2023 | Stacking + SHAP/LIME (Acc 94.4%) | Ensemble and XAI approach |
| **Shakil, Islam & Akter** | **2024** | **Chi-sq + LASSO + SHAP + SMOTE/ADASYN; DT 97.60% acc, 98.73% sensitivity** | **Most relevant — Bangladesh context, SHAP** |
| Mahendra et al. | 2025 | MICE + RF/SVM/KNN/XGBoost comprehensive benchmark | MICE imputation comparison |
| Roy, Hasan et al. | 2025 | Stacking + SHAP/ELI5/LIME + expert knowledge integration | XAI comparison |
| Karthikeyan et al. | 2025 | H2O AutoML + FSAE + LIME/SHAP | AutoML and hybrid feature extraction |

---

## 17. File Structure

```
Integrated-Cervical-Cancer-Data-Platform/
│
├── data/
│   ├── raw/
│   │   ├── who/                          # Raw WHO JSON files
│   │   └── worldbank/                    # Raw World Bank JSON files
│   ├── clean/
│   │   └── risk_factors.csv              # Cleaned UCI dataset
│   └── integrated/
│       ├── imputation_comparison.csv     # KNN vs MICE imputation stats
│       ├── feature_ranking.csv           # Multi-method feature selection output
│       ├── shap_beeswarm.csv             # Global SHAP beeswarm data
│       ├── shap_importance.csv           # Global SHAP importance ranking
│       ├── error_analysis.csv            # Misclassification analysis
│       ├── screening_thresholds_*.json   # Youden/recall thresholds per workflow
│       ├── literature_comparison.csv     # Our results vs. published baselines
│       ├── screening_benchmark_results.csv
│       ├── hybrid_benchmark_results.csv
│       ├── clinical_benchmark_results.csv
│       ├── final_screening_model.joblib  # Serialized Model A
│       ├── final_hybrid_model.joblib     # Serialized Model B
│       └── final_clinical_model.joblib   # Serialized Model C
│
├── src/
│   ├── member1_global_stats/
│   │   ├── acquire.py                    # WHO/WB API ingestion
│   │   ├── clean.py                      # Country panel cleaning & encoding
│   │   └── integrate.py                  # Panel merging
│   ├── member2_burden_bd/
│   ├── member3_riskfactors/
│   │   ├── clean.py                      # UCI imputation, outlier, scaling
│   │   └── advanced_model.py             # Full ML pipeline (870+ lines)
│   └── member4_text_rag/
│       ├── acquire.py                    # Europe PMC document fetching
│       └── clean.py                      # Document text cleaning & indexing
│
├── app/
│   └── dashboard.py                      # Streamlit clinical dashboard (~800 lines)
│
└── docs/
    ├── publication_validation_report.md  # Audit and validation documentation
    └── misclassification_report.md       # Error analysis by patient profile
```

---

## 18. How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost shap streamlit requests boruta joblib
```

### Step 1: Acquire data

```bash
python src/member1_global_stats/acquire.py
python src/member4_text_rag/acquire.py
```

### Step 2: Clean and preprocess

```bash
python src/member1_global_stats/clean.py
python src/member3_riskfactors/clean.py
```

### Step 3: Run the full ML pipeline

```bash
python src/member3_riskfactors/advanced_model.py
```

This runs, in order:
1. `run_benchmarks()` — trains 7 models across 5 folds
2. `run_imbalance_experiments()` — SMOTE / Borderline-SMOTE / ADASYN comparison
3. `run_calibration()` — Platt and Isotonic calibration
4. `run_feature_selection()` — SHAP, MI, RFE, Boruta
5. `run_risk_stratification()` — Youden and recall threshold optimization
6. `run_statistical_comparison()` — paired t-test and Wilcoxon
7. `train_and_save_final_model()` — serialize all three workflow models
8. `run_pipeline()` — global SHAP beeswarm generation

### Step 4: Launch dashboard

```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

### Fix for Windows SHAP memory error

If you see `WinError 8 - c10.dll`:

```python
# At the top of any script that uses SHAP with tree models:
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Always use TreeExplainer for tree-based models — no PyTorch required:
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
```

---

## 19. Limitations

| Limitation | Impact | Mitigation |
|:---|:---|:---|
| **Single-center dataset (Venezuela)** | May not generalize to SEA populations (different HPV strain distribution, social factors) | Frame as proxy dataset; validate against SEA-specific cohorts in future work |
| **Small sample size (858 patients)** | Tiny test set makes hold-out evaluation unreliable; confidence intervals are wide | Use Stratified K-Fold as primary evaluation |
| **Missing data not random** | Sensitive questions (sexual history, STDs) systematically missing from higher-risk patients — imputation may underestimate true risk for flagged features | Sensitivity analysis on imputation strategies |
| **No SEA-specific features** | Current features do not include rural/urban residency, mobile phone access, religious attitudes to screening, or literacy level | Extend feature set for SEA deployment version |
| **Clinical test features in Model A** | Schiller/Hinselmann scores are clinical results, not survey answers — they inflate Model A/combined metrics | Separate evaluation with survey-only feature subset |
| **DBSCAN not properly tuned** | `eps=1.7` produces 35.3% noise; clustering results should not be used to draw scientific conclusions | Re-tune with k-distance elbow plot |

---

## 20. Future Work

1. **Add SHAP to Model A and B** — currently only global SHAP for the full-feature hybrid model; local SHAP for community screening is the most clinically useful piece
2. **Add stacking ensemble** — combine predictions from LightGBM, XGBoost, and LR as a meta-learner; recent papers show ~1–2% AUC gain
3. **Add ADASYN vs. SMOTE comparison table** — Shakil et al. (2024) showed ADASYN outperforms SMOTE on this dataset; currently only SMOTE is used in the primary pipeline
4. **SEA-specific feature extension** — add rural/urban, literacy, transportation access, mobile phone ownership, household income category
5. **VIA image integration** — visual inspection with acetic acid (VIA) photographs fed into a CNN (ResNet-50 or MobileNet) to automate Schiller/Hinselmann scoring without a colposcopist
6. **Multi-center validation** — partner with regional hospitals in Indonesia, Philippines, or Vietnam for external validation
7. **Offline mobile deployment** — export LightGBM model to ONNX format for inference on low-end Android devices without internet connectivity

---

## 21. References

1. **Fernandes K, Cardoso JS, Fernandes J.** Transfer Learning with Partial Observability Applied to Cervical Cancer Screening. *IbPRIA 2017*, Springer. DOI: `10.24432/C5Z310`
2. **Shakil MH, Islam MK, Akter S.** A precise machine learning model: detecting cervical cancer using feature selection and explainable AI. *Journal of Pathology Informatics*, 2024. PMC: `PMC11530914`
3. **Al Mudawi N, Alazeb A.** A Model for Predicting Cervical Cancer Using Machine Learning Algorithms. *Sensors (MDPI)*, 2022. DOI: `10.3390/s22114132`
4. **Glučina M et al.** Cervical cancer diagnostics using machine learning and class balancing techniques. *Applied Sciences*, 2023. DOI: `10.3390/app13021061`
5. **Mahendra M et al.** Comprehensive machine learning model for cervical cancer prediction. *Human Behavior and Emerging Technologies (Wiley)*, 2025.
6. **Roy S, Hasan M et al.** Interpretable AI for cervical cancer risk analysis: stacking ensemble + expert knowledge. *SAGE Journals*, 2025. PMC: `PMC11938887`
7. **Karthikeyan S et al.** Explainable AI for cervical cancer prediction using FSAE + H2O AutoML. *Scientific Reports*, 2025. PMC: `PMC12627774`
8. **WHO Global Cancer Observatory.** Cervical Cancer Fact Sheet, 2024. `https://gco.iarc.fr`

---

*Document maintained by Wadiel Rahman (BSDS, UIU) — Group 8, Integrated Cervical Cancer Data Platform.*
