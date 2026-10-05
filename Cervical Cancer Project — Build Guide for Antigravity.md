# Cervical Cancer Risk Project — Build Guide for Google Antigravity

This guide turns `cervical_cancer_project_specification.md` into an execution plan. It's organized into three major build phases, each one a complete, testable milestone before you move to the next. Feed each section to Antigravity as its own task/prompt — don't ask it to build everything in one pass.

---

## SECTION 1 — Data Foundation: Split, Preprocess, and Prove the Pipeline Works

**Goal of this phase:** a leakage-safe `split.py` and `preprocess.py`, plus proof that a saved preprocessing artifact can transform a single new patient record exactly the way it transforms training data. Get this working *before* touching any model — it's the thing most likely to silently break your live demo later.

### 1.1 Project scaffolding

Have Antigravity create the directory structure from spec §15:

```
cervical-cancer-analysis/
├── data/{raw,processed,splits}/
├── models/
├── artifacts/{metrics,figures,explainability}/
├── src/{split.py,preprocess.py,train.py,test.py}
├── app.py
├── requirements.txt
├── README.md
└── project_specification.md
```

Pin `random_state=42` (or any fixed value) everywhere, and make all data paths configurable via a constants module or CLI args — not hard-coded.

### 1.2 `split.py`

- Load the UCI Cervical Cancer Risk Factors dataset (858 rows). **Do not duplicate rows to hit a 2,000-row target** — document the discrepancy in the README instead (spec §2).
- Identify the target column and separate `X`/`y`.
- Stratified `train_test_split`, fixed seed.
- Save `train.csv` and `test.csv` under `data/splits/`.
- Print shapes and class balance for both sets — this number matters later for imbalance handling.
- **Guardrail:** this file must not fit any imputer, encoder, or scaler, and must not train anything.

### 1.3 `preprocess.py`

- Missing-value handling (the UCI dataset has a known pattern of blank/`?` entries — treat them as missing, not as a literal category, unless you can justify otherwise).
- Categorical encoding, numeric type handling.
- `StandardScaler` (or equivalent) fit **only on the training split**.
- Wrap everything in a single `ColumnTransformer` / `Pipeline` object.
- Save the *fitted* pipeline object (joblib/pickle) to `artifacts/` — this is the one object `train.py`, `test.py`, and `app.py` will all share.

### 1.4 Milestone check before moving on

Write a throwaway script (or a `pytest` test) that:

1. Loads the saved preprocessing pipeline.
2. Constructs one synthetic patient record as a dict/DataFrame, matching the dataset schema.
3. Transforms it and confirms the output shape/column order matches what `X_train` produces.

If this fails, fix it now. This is Improvement #4 from the earlier review — the bug that's cheapest to catch here and most expensive to catch during a live viva demo.

**Exit criteria for Section 1:** `split.py` and `preprocess.py` run end-to-end, the fitted pipeline is saved, and the synthetic-record test passes.

---

## SECTION 2 — Modeling: Validation, Imbalance, Selection, and Explainability

**Goal of this phase:** trained, tuned, fairly-evaluated models plus SHAP explanations — built so that the test set is touched exactly once, at the very end.

### 2.1 Handle class imbalance before anything else (Improvement #2)

Check the class balance number from 1.2. If positives are a small minority:

- Use `class_weight='balanced'` for Logistic Regression, Random Forest, SVM.
- Use `scale_pos_weight` for XGBoost/LightGBM, and CatBoost's equivalent.
- MLPClassifier has no native class-weighting — compensate via resampling *inside* the CV fold (not before splitting) if needed, or accept and document the limitation.
- Make **PR-AUC**, not accuracy or ROC-AUC alone, your primary model-selection metric (spec §10).

### 2.2 Feature selection — decide the method and document the simplification (Improvement #3)

Pick from: correlation filtering, Mutual Information, RFE, SHAP-based importance (spec §8). Two honest options:

- **Rigorous:** re-run selection inside every CV fold (more code, textbook-correct).
- **Pragmatic:** run selection once on the full training set, document this explicitly in the README as a scoping simplification for a lab project, and justify it. Either is defensible — an *undocumented* choice is not. Pick one before writing `train.py`.

### 2.3 `train.py` — model roster and tuning (Improvement #5)

Train, in order of where your effort should actually go:

1. **Logistic Regression** — interpretable baseline, tune regularization strength only.
2. **Random Forest** — classic non-linear baseline, moderate tuning (depth, n_estimators).
3. **MLPClassifier** — required by spec; reuse the handout's settings as a starting point (`hidden_layer_sizes`, `activation='relu'`, `solver='adam'`), but tune `hidden_layer_sizes` and `alpha` via CV, not by eyeballing test accuracy.
4. **XGBoost / LightGBM / CatBoost** — include for completeness, light tuning only; don't over-invest here, they'll likely cluster together on a dataset this small.
5. **SVM** — optional per spec; include only if it doesn't blow your time budget.

For every model:

- Use **stratified k-fold CV (k=5)** on the training set only, for both model selection and hyperparameter search.
- Keep search spaces small and defensible (spec §9: "avoid unnecessarily huge searches").
- **Never compare configurations against the test set to pick a winner** — this is the one habit to explicitly avoid carrying over from the DL handout lab. CV score decides the winner; the test set only confirms it once.
- Save each final fitted model to `models/`.

### 2.4 `test.py` — the one-shot final evaluation

- Load the saved preprocessing pipeline and each saved model.
- Transform the test set using the *already-fitted* pipeline (never refit anything here).
- Compute Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score, and a confusion matrix per model.
- Save results as JSON/CSV to `artifacts/metrics/` for the dashboard to read.
- **Guardrail:** this script touches the test set for evaluation only, once, after all model/hyperparameter decisions from 2.3 are already locked in.

### 2.5 SHAP explainability

- Global feature importance per model (at least for your top 2–3 models — you don't need SHAP for every model you trained).
- Local explanation capability for a single record (this will be reused live in Section 3).
- Save figures/artifacts under `artifacts/explainability/`.
- Phrase everything as "the model weighted this feature" — never as causal language (spec §14).

**Exit criteria for Section 2:** comparison table with all required metrics, every model trained via CV (no test-set peeking during selection), SHAP explanations generated and saved.

---

## SECTION 3 — Streamlit Dashboard, Documentation, and Viva Readiness

**Goal of this phase:** a working app that reuses Section 1's exact preprocessing object, displays Section 2's results honestly, and a README that survives a committee asking "why did you do it this way?"

### 3.1 `app.py` — Model Comparison & Prediction view (spec §12)

- Overview section: title, dataset description, **medical-use disclaimer stated up front**, number of models evaluated, and the honest note that the dataset is 858 rows against a 2,000-row guideline — don't bury this.
- Comparison table reading from `artifacts/metrics/`.
- Model selection + single-record prediction, showing predicted class, probability, and a probability visualization.
- Feature contribution display backed by the saved SHAP artifacts.

### 3.2 `app.py` — Interactive Patient Risk Assessment view (spec §13)

- Build the question list directly from the actual dataset schema — no invented variables.
- On submit: construct the feature record → apply the **same saved preprocessing pipeline from Section 1** → predict → generate a live SHAP explanation for that specific input.
- Wording check: never "you have cervical cancer." Always "the model estimates an elevated probability based on the information provided," with a visible prompt toward professional evaluation.
- Validate input against the expected schema and handle missing/invalid input with clear errors rather than silent failures.

### 3.3 Error handling and reproducibility pass

- Confirm every script fails loudly (not silently) on a missing file or schema mismatch (spec requirement #12).
- Confirm fixed seeds are used everywhere results are reported.
- Re-run `split.py → preprocess.py → train.py → test.py → app.py` from a clean clone to confirm nothing depends on leftover state from earlier runs.

### 3.4 README and documentation

Document, explicitly:

- The 858-vs-2000-row discrepancy and why rows weren't duplicated.
- Which feature-selection approach you chose in 2.2 and why.
- How class imbalance was handled per model.
- The CV-then-test-once discipline, stated as a design principle, not just a sentence buried in code comments — this is likely to come up directly in a viva.
- Model comparison summary and the final model(s) chosen for the live dashboard.

### 3.5 Pre-viva self-check

Be ready to answer out loud:

- Why wasn't the test set used during model/hyperparameter selection?
- Why PR-AUC over plain accuracy here specifically?
- What would happen if you'd standardized before splitting — walk through the leakage mechanism.
- Why is MLPClassifier required even though tree ensembles likely perform comparably or better on this data (same lesson as the DL handout, Titanic case).

**Exit criteria for Section 3:** dashboard runs end-to-end from a clean environment, both features work against live user input, README is complete, and you can defend every methodological choice without looking it up.