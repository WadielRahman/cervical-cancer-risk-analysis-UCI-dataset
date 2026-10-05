"""
Cervical Cancer Patient Data Analysis -- Streamlit Dashboard

Two main features:
    1. Model Comparison Dashboard -- metrics, visualizations, SHAP explainability
    2. Interactive Patient Risk Assessment -- input features, prediction with SHAP

CRITICAL:
    - Uses the SAME preprocessing pipeline fitted during training.
    - Medical disclaimer on all predictions.
    - Never states "You have cervical cancer."

Usage:
    streamlit run app.py
"""

import json
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
import streamlit as st

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CV_RESULTS_JSON_PATH,
    EXPLAINABILITY_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PREPROCESSING_PIPELINE_PATH,
    PROCESSED_DATA_DIR,
    TEST_RESULTS_JSON_PATH,
    DATASET_INFO,
)
from src.preprocess import (
    BINARY_FEATURES,
    COLUMNS_TO_DROP,
    CONTINUOUS_FEATURES,
    PREPROCESSED_FEATURE_NAMES,
    clean_raw_data,
)

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Cervical Cancer Risk Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B3A5C;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FFF3CD;
        border: 1px solid #FFC107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .risk-high {
        background-color: #F8D7DA;
        border: 1px solid #F5C6CB;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #DEE2E6;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED RESOURCE LOADERS
# ============================================================

@st.cache_resource
def load_models():
    """Load all saved models."""
    models = {}
    model_names = [
        "LogisticRegression", "RandomForest", "MLPClassifier",
        "XGBoost", "LightGBM", "CatBoost",
    ]
    for name in model_names:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_pipeline():
    """Load the fitted preprocessing pipeline."""
    if PREPROCESSING_PIPELINE_PATH.exists():
        return joblib.load(PREPROCESSING_PIPELINE_PATH)
    return None


@st.cache_data
def load_cv_results():
    """Load cross-validation results."""
    if CV_RESULTS_JSON_PATH.exists():
        with open(CV_RESULTS_JSON_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_test_results():
    """Load final test evaluation results."""
    if TEST_RESULTS_JSON_PATH.exists():
        with open(TEST_RESULTS_JSON_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_feature_names():
    """Load feature names."""
    path = PROCESSED_DATA_DIR / "feature_names.txt"
    if path.exists():
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return PREPROCESSED_FEATURE_NAMES


@st.cache_resource
def load_shap_background():
    """Load background data for SHAP."""
    path = EXPLAINABILITY_DIR / "shap_background.npy"
    if path.exists():
        return np.load(path)
    return None


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

def show_medical_disclaimer():
    """Display the medical disclaimer."""
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This is a machine-learning / data-analytics educational project, <strong>NOT</strong>
        a clinical diagnostic system. Predictions are model-based probability estimates
        derived from the UCI Cervical Cancer Risk Factors dataset and must not be interpreted
        as medical diagnoses. <strong>Always consult a qualified healthcare professional</strong>
        for clinical evaluation.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1: MODEL COMPARISON DASHBOARD
# ============================================================

def page_model_comparison():
    """Render the model comparison dashboard."""
    st.markdown('<p class="main-header">Model Comparison Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Comparing 6 classification models on the UCI Cervical Cancer Risk Factors dataset</p>',
        unsafe_allow_html=True,
    )

    show_medical_disclaimer()

    cv_results = load_cv_results()
    test_results = load_test_results()

    if not cv_results:
        st.error("CV results not found. Please run `python src/train.py` first.")
        return

    # --- Overview Metrics ---
    st.subheader("Best Model Summary")

    # Find best CV and test models
    best_cv_name = max(cv_results, key=lambda k: cv_results[k]["cv_metrics"]["pr_auc"]["mean"])
    best_cv_prauc = cv_results[best_cv_name]["cv_metrics"]["pr_auc"]["mean"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best CV Model", best_cv_name)
    with col2:
        st.metric("CV PR-AUC", f"{best_cv_prauc:.4f}")
    with col3:
        if test_results:
            best_test_name = max(test_results, key=lambda k: test_results[k]["metrics"]["pr_auc"])
            st.metric("Best Test Model", best_test_name)
    with col4:
        if test_results:
            best_test_prauc = test_results[best_test_name]["metrics"]["pr_auc"]
            st.metric("Test PR-AUC", f"{best_test_prauc:.4f}")

    st.divider()

    # --- Tabs ---
    tab_cv, tab_test, tab_viz, tab_shap = st.tabs([
        "Cross-Validation Results",
        "Final Test Results",
        "Visualizations",
        "SHAP Explainability",
    ])

    # --- CV Results Tab ---
    with tab_cv:
        st.subheader("5-Fold Stratified Cross-Validation (Training Data Only)")
        st.caption("Primary model selection metric: **PR-AUC** (highlighted)")

        rows = []
        for name, res in cv_results.items():
            m = res["cv_metrics"]
            rows.append({
                "Model": name,
                "PR-AUC": f"{m['pr_auc']['mean']:.4f} ± {m['pr_auc']['std']:.4f}",
                "ROC-AUC": f"{m['roc_auc']['mean']:.4f} ± {m['roc_auc']['std']:.4f}",
                "F1": f"{m['f1']['mean']:.4f} ± {m['f1']['std']:.4f}",
                "Recall": f"{m['recall']['mean']:.4f} ± {m['recall']['std']:.4f}",
                "Precision": f"{m['precision']['mean']:.4f} ± {m['precision']['std']:.4f}",
                "Brier Score": f"{m['brier_score']['mean']:.4f}",
                "Best Params": str(res.get("best_tuning_params", {})),
            })

        df_cv = pd.DataFrame(rows)
        st.dataframe(df_cv, use_container_width=True, hide_index=True)

        st.info(
            "**Note:** The test set was NOT used during model selection. "
            "All decisions were made based on CV results. PR-AUC is the primary "
            "metric because it focuses on the minority class (6.4% prevalence)."
        )

    # --- Test Results Tab ---
    with tab_test:
        st.subheader("Final Held-Out Test Set Evaluation")
        st.caption("172 test samples (11 positive, 161 negative)")

        if test_results:
            rows = []
            for name, res in test_results.items():
                m = res["metrics"]
                cm = res["confusion_matrix"]
                rows.append({
                    "Model": name,
                    "PR-AUC": f"{m['pr_auc']:.4f}",
                    "ROC-AUC": f"{m['roc_auc']:.4f}",
                    "F1": f"{m['f1']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "Precision": f"{m['precision']:.4f}",
                    "Brier": f"{m['brier_score']:.4f}",
                    "TP": cm["tp"],
                    "FP": cm["fp"],
                    "FN": cm["fn"],
                    "TN": cm["tn"],
                })

            df_test = pd.DataFrame(rows)
            st.dataframe(df_test, use_container_width=True, hide_index=True)

            st.warning(
                "**Context:** Only 11 positive cases in the test set. "
                "A single misclassification changes Recall by ~9%. "
                "Cross-validation results are more reliable for model comparison."
            )
        else:
            st.warning("Test results not found. Run `python src/test.py` first.")

    # --- Visualizations Tab ---
    with tab_viz:
        st.subheader("Evaluation Visualizations")

        viz_files = {
            "Confusion Matrices": FIGURES_DIR / "confusion_matrices.png",
            "ROC Curves": FIGURES_DIR / "roc_curves.png",
            "Precision-Recall Curves": FIGURES_DIR / "pr_curves.png",
            "Metric Comparison": FIGURES_DIR / "metric_comparison.png",
        }

        for title, path in viz_files.items():
            if path.exists():
                st.markdown(f"**{title}**")
                st.image(str(path), use_container_width=True)
                st.divider()
            else:
                st.warning(f"{title} not found at {path}")

    # --- SHAP Tab ---
    with tab_shap:
        st.subheader("SHAP Feature Importance")
        st.caption("Which features drive each model's predictions?")

        # Combined importance heatmap
        heatmap_path = EXPLAINABILITY_DIR / "shap_importance_comparison.png"
        if heatmap_path.exists():
            st.markdown("**Cross-Model Feature Importance Comparison**")
            st.image(str(heatmap_path), use_container_width=True)
            st.divider()

        # Individual model SHAP plots
        shap_model = st.selectbox(
            "Select model for detailed SHAP analysis:",
            ["RandomForest", "LogisticRegression", "XGBoost", "LightGBM", "CatBoost"],
            key="shap_model_select",
        )

        col1, col2 = st.columns(2)

        bar_path = EXPLAINABILITY_DIR / f"shap_bar_{shap_model}.png"
        if bar_path.exists():
            with col1:
                st.markdown(f"**SHAP Bar Plot — {shap_model}**")
                st.image(str(bar_path), use_container_width=True)

        beeswarm_path = EXPLAINABILITY_DIR / f"shap_beeswarm_{shap_model}.png"
        if beeswarm_path.exists():
            with col2:
                st.markdown(f"**SHAP Beeswarm — {shap_model}**")
                st.image(str(beeswarm_path), use_container_width=True)

        # Waterfall plot
        waterfall_files = list(EXPLAINABILITY_DIR.glob(f"shap_waterfall_{shap_model}_*.png"))
        if waterfall_files:
            st.markdown(f"**SHAP Waterfall — Individual Patient Example ({shap_model})**")
            st.image(str(waterfall_files[0]), use_container_width=True)


# ============================================================
# PAGE 2: PATIENT RISK ASSESSMENT
# ============================================================

def page_risk_assessment():
    """Render the interactive patient risk assessment page."""
    st.markdown(
        '<p class="main-header">Interactive Patient Risk Assessment</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Enter patient features to estimate cervical cancer risk probability</p>',
        unsafe_allow_html=True,
    )

    show_medical_disclaimer()

    models = load_models()
    pipeline = load_pipeline()
    feature_names = load_feature_names()
    background = load_shap_background()

    if not models:
        st.error("No models found. Run the training pipeline first.")
        return
    if pipeline is None:
        st.error("Preprocessing pipeline not found. Run `python src/preprocess.py` first.")
        return

    # --- Model Selection ---
    available_models = [n for n in models.keys() if n != "MLPClassifier"]
    selected_model_name = st.selectbox(
        "Select prediction model:",
        available_models,
        index=available_models.index("RandomForest") if "RandomForest" in available_models else 0,
        help="RandomForest had the best cross-validation PR-AUC. MLPClassifier is excluded due to poor performance.",
    )

    st.divider()

    # --- Feature Input Form ---
    st.subheader("Patient Information")
    st.caption("Fill in the patient's clinical and demographic information below.")

    with st.form("patient_form"):
        # Group 1: Demographics
        st.markdown("**Demographics**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", min_value=13, max_value=90, value=30, step=1)
        with col2:
            n_partners = st.number_input("Number of sexual partners", min_value=0, max_value=30, value=2, step=1)
        with col3:
            first_intercourse = st.number_input("Age at first sexual intercourse", min_value=10, max_value=40, value=17, step=1)
        with col4:
            n_pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=15, value=1, step=1)

        st.divider()

        # Group 2: Smoking
        st.markdown("**Smoking History**")
        col1, col2, col3 = st.columns(3)
        with col1:
            smokes = st.selectbox("Smokes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col2:
            smokes_years = st.number_input("Smoking duration (years)", min_value=0.0, max_value=40.0, value=0.0, step=0.5)
        with col3:
            smokes_packs = st.number_input("Smoking intensity (packs/year)", min_value=0.0, max_value=40.0, value=0.0, step=0.5)

        st.divider()

        # Group 3: Contraceptives
        st.markdown("**Contraceptive History**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hc = st.selectbox("Hormonal Contraceptives", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col2:
            hc_years = st.number_input("HC duration (years)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
        with col3:
            iud = st.selectbox("IUD", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col4:
            iud_years = st.number_input("IUD duration (years)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)

        st.divider()

        # Group 4: STD History
        st.markdown("**STD History**")
        col1, col2, col3 = st.columns(3)
        with col1:
            stds = st.selectbox("Any STDs", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col2:
            stds_number = st.number_input("Number of STDs", min_value=0, max_value=5, value=0, step=1)
        with col3:
            stds_n_diag = st.number_input("Number of STD diagnoses", min_value=0, max_value=5, value=0, step=1)

        # Individual STDs (in expander)
        with st.expander("Individual STD Types (expand to specify)"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                std_condylomatosis = st.checkbox("Condylomatosis")
                std_cervical = st.checkbox("Cervical condylomatosis")
                std_vaginal = st.checkbox("Vaginal condylomatosis")
                std_vulvo = st.checkbox("Vulvo-perineal condylomatosis")
            with col2:
                std_syphilis = st.checkbox("Syphilis")
                std_pid = st.checkbox("Pelvic inflammatory disease")
                std_herpes = st.checkbox("Genital herpes")
            with col3:
                std_molluscum = st.checkbox("Molluscum contagiosum")
                std_aids = st.checkbox("AIDS")
                std_hiv = st.checkbox("HIV")
            with col4:
                std_hepb = st.checkbox("Hepatitis B")
                std_hpv = st.checkbox("HPV")

        st.divider()

        # Group 5: Previous Diagnoses
        st.markdown("**Previous Diagnoses**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dx_cancer = st.selectbox("Dx: Cancer", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col2:
            dx_cin = st.selectbox("Dx: CIN", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col3:
            dx_hpv = st.selectbox("Dx: HPV", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with col4:
            dx = st.selectbox("Dx (any)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        st.divider()

        # Group 6: Previous Screening Results
        st.markdown("**Previous Screening Results**")
        col1, col2, col3 = st.columns(3)
        with col1:
            hinselmann = st.selectbox("Hinselmann test", [0, 1], format_func=lambda x: "Negative" if x == 0 else "Positive")
        with col2:
            schiller = st.selectbox("Schiller test", [0, 1], format_func=lambda x: "Negative" if x == 0 else "Positive")
        with col3:
            citology = st.selectbox("Citology", [0, 1], format_func=lambda x: "Negative" if x == 0 else "Positive")

        submitted = st.form_submit_button(
            "Generate Risk Assessment",
            type="primary",
            use_container_width=True,
        )

    # --- Process Prediction ---
    if submitted:
        # Construct patient DataFrame
        patient_data = {
            "Age": age,
            "Number of sexual partners": n_partners,
            "First sexual intercourse": first_intercourse,
            "Num of pregnancies": n_pregnancies,
            "Smokes": smokes,
            "Smokes (years)": smokes_years,
            "Smokes (packs/year)": smokes_packs,
            "Hormonal Contraceptives": hc,
            "Hormonal Contraceptives (years)": hc_years,
            "IUD": iud,
            "IUD (years)": iud_years,
            "STDs": stds,
            "STDs (number)": stds_number,
            "STDs:condylomatosis": int(std_condylomatosis),
            "STDs:cervical condylomatosis": int(std_cervical),
            "STDs:vaginal condylomatosis": int(std_vaginal),
            "STDs:vulvo-perineal condylomatosis": int(std_vulvo),
            "STDs:syphilis": int(std_syphilis),
            "STDs:pelvic inflammatory disease": int(std_pid),
            "STDs:genital herpes": int(std_herpes),
            "STDs:molluscum contagiosum": int(std_molluscum),
            "STDs:AIDS": int(std_aids),
            "STDs:HIV": int(std_hiv),
            "STDs:Hepatitis B": int(std_hepb),
            "STDs:HPV": int(std_hpv),
            "STDs: Number of diagnosis": stds_n_diag,
            "Dx:Cancer": dx_cancer,
            "Dx:CIN": dx_cin,
            "Dx:HPV": dx_hpv,
            "Dx": dx,
            "Hinselmann": hinselmann,
            "Schiller": schiller,
            "Citology": citology,
        }

        patient_df = pd.DataFrame([patient_data])

        # Clean and preprocess
        patient_cleaned, _ = clean_raw_data(patient_df)
        patient_transformed = pipeline.transform(patient_cleaned)

        # Predict
        model = models[selected_model_name]
        y_pred = model.predict(patient_transformed)[0]
        y_proba = model.predict_proba(patient_transformed)[0, 1]

        st.divider()
        st.subheader("Risk Assessment Result")

        # --- Display Result ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric(
                label="Predicted Probability",
                value=f"{y_proba:.1%}",
                help="Probability of a positive biopsy result as estimated by the model.",
            )
            st.metric(label="Model Used", value=selected_model_name)

            # Probability bar
            st.progress(min(y_proba, 1.0))

        with col2:
            if y_proba >= 0.5:
                st.markdown(f"""
                <div class="risk-high">
                    <strong>Elevated Risk Detected</strong><br>
                    The {selected_model_name} model estimates an <strong>elevated probability
                    ({y_proba:.1%})</strong> of a positive biopsy result based on the entered
                    features. This does NOT constitute a diagnosis.<br><br>
                    <strong>Recommendation:</strong> The patient should be referred for further
                    clinical evaluation by a qualified healthcare professional.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <strong>Lower Risk Estimated</strong><br>
                    The {selected_model_name} model estimates a <strong>lower probability
                    ({y_proba:.1%})</strong> of a positive biopsy result based on the entered
                    features. This does NOT rule out cervical cancer.<br><br>
                    <strong>Recommendation:</strong> Regular screening should continue as
                    recommended by clinical guidelines.
                </div>
                """, unsafe_allow_html=True)

        # --- SHAP Explanation ---
        st.divider()
        st.subheader("Feature Contribution (SHAP Explanation)")
        st.caption("Which features pushed the prediction higher or lower?")

        try:
            # Create explainer
            if selected_model_name == "LogisticRegression":
                if background is not None:
                    masker = shap.maskers.Independent(background)
                    explainer = shap.LinearExplainer(model, masker)
                else:
                    explainer = None
            else:
                explainer = shap.TreeExplainer(model)

            if explainer is not None:
                sv = explainer.shap_values(patient_transformed)

                # Handle binary classification output
                if isinstance(sv, list):
                    shap_vals = sv[1][0]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                elif sv.ndim == 3:
                    shap_vals = sv[0, :, 1]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    shap_vals = sv[0]
                    base_val = explainer.expected_value
                    if isinstance(base_val, (list, np.ndarray)):
                        base_val = base_val[0] if len(base_val) == 1 else base_val[1]

                explanation = shap.Explanation(
                    values=shap_vals,
                    base_values=float(base_val) if not isinstance(base_val, float) else base_val,
                    data=patient_transformed[0],
                    feature_names=feature_names,
                )

                fig, ax = plt.subplots(figsize=(10, 7))
                shap.plots.waterfall(explanation, max_display=15, show=False)
                st.pyplot(fig)
                plt.close()

                st.caption(
                    "Red bars push the prediction higher (toward positive). "
                    "Blue bars push it lower (toward negative). "
                    "The base value is the model's average prediction across all training data."
                )
            else:
                st.info("SHAP explanation not available for this model configuration.")

        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

        # --- Medical disclaimer (repeated) ---
        show_medical_disclaimer()


# ============================================================
# PAGE 3: ABOUT
# ============================================================

def page_about():
    """Render the about/documentation page."""
    st.markdown('<p class="main-header">About This Project</p>', unsafe_allow_html=True)

    show_medical_disclaimer()

    st.subheader("Project Overview")
    st.markdown(f"""
    **Title:** Cervical Cancer Patient Data Analysis

    **Domain:** Healthcare / Data Analytics / Machine Learning

    **Objective:** Analyze cervical cancer risk factors using the UCI Cervical Cancer
    Risk Factors dataset and build an interactive, explainable machine-learning
    application for risk prediction.

    **Dataset:** [UCI Cervical Cancer Risk Factors]({DATASET_INFO['source_url']})
    — {DATASET_INFO['expected_rows']} patient records, 35 features, target: `Biopsy`
    """)

    st.subheader("Dataset Limitation")
    st.warning(f"""
    The UCI dataset contains **{DATASET_INFO['expected_rows']} records**.
    The university course guideline specifies a minimum of **{DATASET_INFO['university_min_rows']} rows**.
    This discrepancy is transparently documented. Records were **NOT** fabricated or
    duplicated to meet the row requirement. Methodological honesty is preserved.
    """)

    st.subheader("Methodology")
    st.markdown("""
    **Pipeline:**
    ```
    Raw Data → Train/Test Split (stratified) → Preprocess (fit on train ONLY)
    → Cross-Validation (5-fold stratified) → Model Selection (by PR-AUC)
    → Final Test Evaluation → SHAP Explainability → Dashboard
    ```

    **Models Trained:**
    1. Logistic Regression (`class_weight='balanced'`)
    2. Random Forest (`class_weight='balanced'`)
    3. MLPClassifier (no native class-weight — documented limitation)
    4. XGBoost (`scale_pos_weight`)
    5. LightGBM (`is_unbalance=True`)
    6. CatBoost (`auto_class_weights='Balanced'`)

    **Class Imbalance:** 6.41% positive rate (14.59:1 ratio). Handled via native
    class weighting in 5 of 6 models. SMOTE was evaluated and rejected (44 positive
    samples too few for reliable synthesis).

    **Feature Selection:** Investigated via Correlation Filtering, Mutual Information,
    and RFE. All 33 features retained (20.8:1 sample-to-feature ratio is adequate).
    SHAP provides post-hoc feature importance.

    **Preprocessing:** `?` → NaN, 2 columns dropped (91.7% missing),
    median imputation + StandardScaler for continuous features, most-frequent
    imputation for binary features. Pipeline fitted on training data ONLY.
    """)

    st.subheader("Leakage Prevention")
    st.success("""
    **Confirmed safeguards:**
    - Train/test split performed BEFORE any preprocessing
    - Preprocessing pipeline fitted on training data ONLY
    - Test data transformed using already-fitted pipeline (no refitting)
    - Cross-validation used only training data
    - Test set used exactly ONCE for final evaluation
    - No model retraining based on test results
    """)

    st.subheader("Key Findings")
    st.markdown("""
    1. **Schiller test** is the dominant predictor across all models (confirmed by both MI and SHAP)
    2. **Clinical screening tests** (Schiller, Hinselmann, Citology) carry the most predictive signal
    3. **Random Forest** achieved best CV PR-AUC (0.7575)
    4. **MLPClassifier** failed to detect any cancer cases — demonstrating why class-weight support matters
    5. **Native class weighting** is effective and simpler than SMOTE for this dataset
    """)

    st.subheader("Reproducibility")
    st.info("""
    All random operations use `random_state=42`. The full pipeline can be
    reproduced by running:
    ```
    python src/split.py
    python src/preprocess.py
    python src/train.py
    python src/test.py
    python src/explain.py
    streamlit run app.py
    ```
    """)


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main Streamlit application."""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        [
            "Model Comparison",
            "Patient Risk Assessment",
            "About",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Cervical Cancer Patient Data Analysis\n\n"
        "University Data Analytics Project\n\n"
        "Dataset: UCI Cervical Cancer Risk Factors"
    )

    # Route to selected page
    if page == "Model Comparison":
        page_model_comparison()
    elif page == "Patient Risk Assessment":
        page_risk_assessment()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
