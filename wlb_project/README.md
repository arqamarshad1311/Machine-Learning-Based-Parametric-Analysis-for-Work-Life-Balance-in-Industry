# Machine Learning Based Parametric Analysis for Work-Life Balance in Industry
**Team Code Sentinels (MLI-III-2026-T010)** · MCA (AI & DS), Graphic Era (Deemed to be University)
Mentor: Mr. Jaishankar Bhatt · Team: Mohd Arqam Arshad (lead), Amaan Rana, Anshul Aswal

**Final project package** - data audit, preprocessing, feature selection, model development, evaluation, K-Means
segmentation, Streamlit dashboard and exportable HR report (see `docs/FINAL_STATUS.md` and the project report in `docs/`).

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py            # ~4 min on 1 CPU (use --fast for ~1 min)
streamlit run dashboard/app.py    # interactive dashboard (upload data/sample_upload.csv or WLB_PBL_DATASET.xlsx)
pytest -q                         # tests
```
`python run_pipeline.py` must be run once on your machine before the dashboard: it creates the model files in
`outputs/models/` (they depend on the installed scikit-learn / pandas versions, so they are always regenerated locally).


## Pipeline (matches the architecture slide)
| Stage | Code | Output |
|---|---|---|
| Data collection | `src/data_loader.py` | loads `data/raw/WLB_PBL_DATASET.xlsx` (10,000 x 16) |
| Audit | `src/audit.py` | `outputs/metrics/data_audit.json` (reproduces the report's test table) |
| Pre-processing | `src/preprocessing.py` | `data/processed/wlb_cleaned.csv`; one-hot + StandardScaler inside a sklearn `Pipeline` |
| Feature engineering | `src/feature_selection.py` | Spearman + mutual-information ranking (drops `age`, `gender` for behavioural set) |
| Model training | `src/train.py` | Logistic Regression, Random Forest, XGBoost, tuned by stratified-CV `RandomizedSearchCV` (macro-F1), class weights; optional K-Means |
| Evaluation | `src/train.py`, `src/reporting.py` | accuracy, macro-F1, High-Risk recall, ROC-AUC, confusion matrices, permutation importance |
| Dashboard / output | `dashboard/app.py`, `src/batch.py`, `src/reporting.py` | Streamlit app (CSV/Excel upload), HR report (HTML, print to PDF), predictions CSV |
| Model persistence | `src/persistence.py` | one `.joblib` per feature set x model in `outputs/models/` |

## How each Phase-II challenge is handled
| Challenge (progress report) | Implementation |
|---|---|
| 222 rows > 14 h, 29 at 24 h, 37 at 0 h | `clean()` drops 0 h/24 h entry errors (66 rows) and caps the rest at 14 h; **raw vs cleaned** are both trained and compared |
| 960 rows with work+rest+sleep+exercise > 24 h | flagged (`flag_time_budget`) and reported in the dashboard's `data_quality_note`; kept in training |
| Target leakage | two feature sets: `behavioural_only` vs `all_features`; permutation importance for both |
| Class imbalance (25% High) | stratified split, class weights / balanced sample weights, macro-F1 + High-Risk recall tracked |
| Synopsis discrepancy | classification (LR baseline), `occupation_type` is a feature, work hours is the workload proxy |
| Synthetic data | 5-fold CV reported; limitation printed in the HR report |

Both cleaning modes are judged on the same held-out, cleaned test set (20%, stratified).

## Results (cleaned data, held-out test set)
| Feature set | Model | Accuracy | Macro-F1 | High-Risk recall | ROC-AUC |
|---|---|---|---|---|---|
| behavioural_only | Logistic Regression | 0.734 | 0.733 | 0.773 | 0.894 |
| behavioural_only | Random Forest | 0.718 | 0.715 | 0.685 | 0.880 |
| behavioural_only | XGBoost | 0.725 | 0.724 | 0.733 | 0.882 |
| all_features | Logistic Regression | 0.995 | 0.996 | 0.998 | 1.000 |
| all_features | Random Forest | 0.936 | 0.935 | 0.914 | 0.994 |
| all_features | XGBoost | 0.968 | 0.967 | 0.962 | 0.998 |

**Key findings**
1. The dataset notes say the label is "a weighted combination" of these fields, so with stress / satisfaction / health included a *linear* model reaches ~99.6% - the label is essentially re-derived from its inputs (leakage). The honest estimate of predicting risk from measurable habits is **~0.73 macro-F1** (behavioural-only).
2. Top drivers: **work hours** (largest in both sets), **sleep**, then exercise, family time and commute; with all features also job satisfaction, stress and mental health.
3. Removing the 0 h / 24 h entry errors changes results only marginally (+/-0.01 macro-F1) - the models are robust to these ~0.7% of rows.
4. K-Means (k=3) on behavioural parameters gives segments that line up with risk (one segment is 56% High Risk, another only 4%).

## Dashboard
| Tab | What it does |
|---|---|
| Overview | class balance, correlations, data-quality audit, high-risk share by occupation |
| Predict one employee | sliders -> colour-coded risk, class probabilities, main adverse parameters |
| Batch + HR report | upload CSV/Excel -> risk distribution, per-employee drivers, priority list, data-quality notes, CSV + HTML report downloads |
| Model comparison | all 12 experiments, confusion matrices, leakage warning |
| Feature importance | permutation importance, heatmap, K-Means segments |

Use the sidebar to switch between the *behavioural-only* model (recommended for proactive screening) and the *all-features* model (explains a current situation but is leaked).

## Repository layout
```
data/raw | data/processed | data/sample_upload.csv     src/   dashboard/   tests/   outputs/{figures,metrics,models,reports}   docs/
```
Re-running `python run_pipeline.py` regenerates everything in `outputs/`.
