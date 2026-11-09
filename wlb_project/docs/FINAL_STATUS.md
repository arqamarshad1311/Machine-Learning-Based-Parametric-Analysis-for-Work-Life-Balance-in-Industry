# Final project status

Everything promised in the final project report (MCA (AI & DS) MLI-III-2026-T010) is implemented in this repository.

| Report section | Deliverable | Where |
|---|---|---|
| 3.2 Step 1 - data audit | automated audit (schema, missing, duplicates, ranges, time budget) | `src/audit.py`, `outputs/metrics/data_audit.json` |
| 3.2 Step 2 - raw vs cleaned | two cleaning modes; 66 entry errors dropped, hours capped at 14 | `src/preprocessing.py`, `data/processed/wlb_cleaned.csv` |
| 3.2 Step 3 - feature selection | Spearman + mutual information; `age`, `gender` dropped | `src/feature_selection.py`, `outputs/metrics/feature_ranking_*.csv` |
| 3.2 Step 4 - models | Logistic Regression, Random Forest, XGBoost, tuned with 5-fold stratified CV (macro-F1) on both feature sets | `src/train.py` |
| 3.2 Step 5 - evaluation | accuracy, macro-F1, High-Risk recall, ROC-AUC, confusion matrices, permutation importance | `outputs/metrics`, `outputs/figures` |
| 4.5 K-Means segmentation | k = 3 behavioural segments | `outputs/metrics/kmeans_*.csv`, `outputs/figures/07_*.png` |
| 5.1 Dashboard | upload CSV **or Excel**, risk distribution chart, colour-coded predictions, per-employee drivers, correlation heatmap, importance chart, priority list, HR report export | `dashboard/app.py`, `src/batch.py` |
| 5.4 Validation & testing | 15 pytest tests: data layer, modelling claims, reporting, persistence, batch upload logic | `tests/` |
| 5.6 Output artifacts | a serialized model for **every** feature set x model, plus the best-model bundle used by the dashboard | `outputs/models/` (created by `run_pipeline.py`) |
| 5.5 Installation & usage | README quick start; sample file for the upload tab | `README.md`, `data/sample_upload.csv` |

## Changes made in the final update
* Dashboard accepts `.xlsx` as well as `.csv` (the report's usage guide uploads `WLB_PBL_DATASET.xlsx`).
* Batch tab now shows the predicted risk distribution, heatmap and importance chart next to the employee table, colour-codes risk, and adds a `data_quality_note` per employee (0 h / 24 h entries, capped hours, daily hours > 24).
* `run_pipeline.py` saves every trained model (`src/persistence.py`), not only the best per feature set.
* XGBoost is optional at import time; if it is missing the rest of the pipeline still runs and prints a warning.
* `requirements.txt` now lists `openpyxl` (needed to read the Excel dataset) and `jinja2`; pandas >= 2.1.
* New tests: `tests/test_models_and_batch.py`.
* README corrected (960 time-budget rows, not ~920).

## Known limitations (also in report section 6.2)
Synthetic label (target leakage with subjective scores), single dataset, no live HR rollout, static snapshot, batch only.
Demo video, poster and Turnitin check are team submission items and are not part of this code package.
