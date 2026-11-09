# Machine Learning Based Parametric Analysis for Work-Life Balance in Industry

**Team Code Sentinels (MLI-III-2026-T010)** · MCA (AI & DS), Graphic Era (Deemed to be University), Dehradun
**Mentor:** Mr. Jaishankar Bhatt
**Team:** Mohd Arqam Arshad (lead), Amaan Rana, Anshul Aswal

This project predicts an employee's work-life balance (WLB) risk category (Low, Medium or High) from measurable daily
parameters such as work hours, commute, sleep, rest, exercise, family and social time, and self-reported stress,
satisfaction and health. It also shows which parameters drive that risk, so HR teams can spot problems early and plan
wellness measures, instead of waiting for burnout or attrition.

The package contains the data audit, preprocessing, feature selection, model training and evaluation, K-Means
segmentation, a Streamlit dashboard and an exportable HR report. The full write-up is in the project report in `docs/`.

## Repository

- **Repo:** https://github.com/arqamarshad1311/Machine-Learning-Based-Parametric-Analysis-for-Work-Life-Balance-in-Industry
- **Branch:** `main`
- **Important commits:** `ace5fb3c39e9fff410842a1bc05e0ad1ac5d3114`, `f84d2ba141de01f9c859cac2f991aba0e4497293`

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py            # full run (about 4 min on 1 CPU); add --fast for about 1 min
streamlit run dashboard/app.py    # interactive dashboard
pytest -q                         # unit tests
```

Run `python run_pipeline.py` once before starting the dashboard. It creates the model files in `outputs/models/`.
Those files depend on the installed scikit-learn and pandas versions, so they are always regenerated locally and
are not meant to be copied between machines.

In the dashboard's **Batch + HR report** tab you can upload `data/sample_upload.csv` (200 sample employees) or the
full `WLB_PBL_DATASET.xlsx`.

## Dataset

`data/raw/WLB_PBL_DATASET.xlsx` holds **10,000 employee records** (IDs 10001 to 20000) with 16 attributes and 14
occupation types, in two sheets: *WLB Dataset* and *Column Descriptions*.

| Column | Description |
|---|---|
| `employee_id` | Unique identifier (10001 to 20000) |
| `gender`, `age` | Male / Female; age 22 to 65 |
| `occupation_type` | Occupation category (Teacher, Manager, Freelancer, Driver and others) |
| `avg_work_hours_per_day` | Average daily work hours (usually 7 to 12, extremes from 0 to 24) |
| `avg_rest_hours_per_day` | Time not spent working, sleeping or exercising |
| `avg_sleep_hours_per_day` | Average daily sleep (usually 6 to 8) |
| `avg_exercise_hours_per_day` | Average daily exercise (usually 0 to 2) |
| `commute_time_mins` | Daily commute in minutes (0 to 120) |
| `stress_level` | Self-reported, 1 (low) to 10 (high) |
| `job_satisfaction` | Self-reported, 1 (low) to 10 (high) |
| `family_time_hours`, `social_time_hours` | Daily time with family / socializing |
| `mental_health_score`, `physical_health_score` | Self-reported, 1 (poor) to 10 (excellent) |
| `wlb_risk_category` | **Target:** Low 40%, Medium 35%, High 25% |

The target is derived from a weighted combination of the other columns, which is the source of the leakage discussed
below. The cleaned file `data/processed/wlb_cleaned.csv` (9,934 rows) adds three flag columns:
`flag_work_outlier`, `flag_entry_error` and `flag_time_budget`.

## Pipeline

| Stage | Code | Output |
|---|---|---|
| Data collection | `src/data_loader.py` | Loads the raw dataset (10,000 x 16) |
| Audit | `src/audit.py` | `outputs/metrics/data_audit.json` (reproduces the report's audit table) |
| Preprocessing | `src/preprocessing.py` | `data/processed/wlb_cleaned.csv`; one-hot encoding and scaling inside a scikit-learn `Pipeline` |
| Feature selection | `src/feature_selection.py` | Spearman and mutual-information ranking; `age` and `gender` are dropped for negligible signal |
| Model training | `src/train.py` | Logistic Regression, Random Forest and XGBoost, tuned with 5-fold stratified `RandomizedSearchCV` on macro-F1, with class weights; K-Means for segmentation |
| Evaluation | `src/train.py`, `src/reporting.py` | Accuracy, macro-F1, High-Risk recall, ROC-AUC, confusion matrices, permutation importance |
| Dashboard and reports | `dashboard/app.py`, `src/batch.py`, `src/reporting.py` | Streamlit app (CSV or Excel upload), HR report (HTML, printable to PDF), predictions CSV |
| Model persistence | `src/persistence.py` | One `.joblib` file per feature set and model in `outputs/models/` |

`run_pipeline.py` runs all of these in order: load and audit, clean, select features, train and tune, evaluate,
compute permutation importance and K-Means, draw figures, save the model bundle and build the HR report.

## How the data problems are handled

| Problem found in the audit | What the code does |
|---|---|
| 222 rows above 14 work hours, 29 at exactly 24 h, 37 at 0 h | `clean()` drops the 66 rows at 0 h or 24 h (likely entry errors) and caps the rest at 14 h. Both the raw and cleaned modes are trained and compared |
| 960 rows where work + rest + sleep + exercise exceeds 24 h | Flagged (`flag_time_budget`), reported in the dashboard's `data_quality_note` column, and kept in training |
| Target leakage | Two feature sets, `behavioural_only` and `all_features`, are evaluated side by side, with permutation importance for both |
| Class imbalance (25% High Risk) | Stratified splits, class weights or balanced sample weights, and macro-F1 plus High-Risk recall as the main metrics |
| Mismatch with the original synopsis | Classification with Logistic Regression as baseline; `occupation_type` is used as a feature; work hours act as the workload proxy |
| Synthetic data | 5-fold cross-validation is reported and the limitation is stated in the HR report |

Both cleaning modes are judged on the same held-out cleaned test set (20%, stratified).

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

1. **Target leakage.** The dataset notes describe the label as a weighted combination of the other fields, so with
   stress, satisfaction and health scores included, even a linear model reaches about 0.996 macro-F1. The label is
   essentially re-derived from its inputs. The honest estimate of predicting risk from measurable habits is about
   **0.73 macro-F1** (behavioural features only).
2. **Top drivers.** Work hours rank first in both feature sets, followed by sleep, then exercise, family time and
   commute. With all features, job satisfaction, stress and mental health also rank high.
3. **Robustness.** Removing the 0 h and 24 h entry errors changes macro-F1 by less than 0.01, so the models are robust
   to these roughly 0.7% of rows.
4. **Segments.** K-Means (k = 3) on behavioural parameters gives segments that line up with risk: the high-risk segment
   works about 11.4 h a day with about 3.7 h of rest and 1.5 h of family time and is roughly 64% High Risk, while the
   low-risk segment is only about 4% High Risk. Exact values are in `outputs/metrics/kmeans_crosstab.csv`.

## Dashboard

Start it with `streamlit run dashboard/app.py`. Use the sidebar to choose the feature set: **behavioural only**
(recommended for proactive screening) or **all features** (explains a current situation, but is affected by leakage).

| Tab | What it does |
|---|---|
| Overview | Class balance, correlations, data-quality audit, high-risk share by occupation |
| Predict one employee | Sliders for each parameter give a colour-coded risk, class probabilities and the main adverse parameters |
| Batch + HR report | Upload a CSV or Excel file to see the risk distribution, per-employee drivers, a priority list and data-quality notes; download predictions (CSV) and the HR report (HTML, print to PDF) |
| Model comparison | All 12 experiments (6 models and feature sets, each in raw and cleaned mode), confusion matrices and a leakage warning |
| Feature importance | Permutation importance, correlation heatmap and K-Means segments |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── run_pipeline.py          # end-to-end pipeline entry point
├── data/
│   ├── raw/                 # WLB_PBL_DATASET.xlsx
│   ├── processed/           # wlb_cleaned.csv
│   └── sample_upload.csv    # 200 sample employees for the dashboard
├── src/                     # config, data_loader, audit, preprocessing, feature_selection,
│                            # train, batch, reporting, persistence
├── dashboard/app.py         # Streamlit app
├── tests/                   # pytest suite
├── outputs/                 # figures/, metrics/, models/, reports/  (regenerated by the pipeline)
└── docs/                    # project report, diagrams
```

There is no separate `notebooks/` folder; the exploratory analysis is produced by `src/reporting.py`.
Re-running `python run_pipeline.py` regenerates everything in `outputs/`.

## Outputs

| Folder | Contents |
|---|---|
| `outputs/figures/` | EDA and model figures |
| `outputs/models/` | `wlb_models.joblib` bundle plus one file per feature set and model |
| `outputs/metrics/` | `model_comparison.csv`, `data_audit.json`, feature rankings, permutation importance, K-Means tables, `summary.json` |
| `outputs/reports/` | `test_predictions.csv` and the HR report (HTML) |

## Limitations and responsible use

- The dataset appears synthetic, so findings describe the drivers of this label and may not match real workplaces.
- The data has no managerial support, autonomy, dependents or financial stress, and covers a single organization.
- Predictions are decision-support signals, not verdicts. Interventions should be chosen by HR together with the
  employee and never triggered automatically, and the tool should not be used for performance reviews or dismissal.
- Remove or pseudonymise employee names before upload, and run the dashboard inside your own secure environment.

## Requirements

Python 3.10 or newer is recommended. Dependencies (see `requirements.txt`): pandas, numpy, scikit-learn, xgboost,
matplotlib, seaborn, joblib, openpyxl, jinja2, streamlit and pytest.
