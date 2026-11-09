"""End-to-end pipeline:  python run_pipeline.py [--fast]

raw data -> audit -> cleaning -> feature selection -> model training/tuning -> evaluation
-> permutation importance -> K-Means -> figures -> saved model bundle -> HR report
"""
import argparse
import json
import joblib
from src.config import *
from src.data_loader import load_raw
from src.audit import run_audit, save_audit
from src.preprocessing import clean
from src import train as T, reporting as R
from src.persistence import save_all_models


def main(fast: bool):
    for d in (FIG_DIR, MODEL_DIR, REPORT_DIR, METRIC_DIR, CLEAN_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)

    print("== 1. Load + audit ==")
    raw = load_raw()
    audit = run_audit(raw); save_audit(audit)
    for k, v in audit["tests"].items(): print(f"  {k:32s} {v}")
    print(f"  work>14h: {audit['work_hours_gt_14']} | 24h: {audit['work_hours_eq_24']} | 0h: {audit['work_hours_eq_0']} "
          f"| time budget>24h: {audit['time_budget_gt_24']}")
    R.eda_figures(raw)

    print("== 2. Cleaning ==")
    cleaned = clean(raw, "cleaned"); cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"  {len(raw)} -> {len(cleaned)} rows saved to {CLEAN_PATH.name}")

    print("== 3-4. Feature selection, training, tuning, evaluation ==")
    results, cms, fitted, sel_tables, test = T.run_experiments(raw, fast=fast)
    results.to_csv(METRIC_DIR / "model_comparison.csv", index=False)
    for (mode, fs), tab in sel_tables.items():
        if mode == "cleaned": tab.to_csv(METRIC_DIR / f"feature_ranking_{fs}.csv", index=False)

    print("== 5. Importance, K-Means, figures ==")
    bundles = T.best_per_feature_set(results, fitted)
    importances = {}
    for fs, b in bundles.items():
        yte = test[TARGET].map(CLASS_TO_INT)
        importances[fs] = T.permutation_table(b["pipeline"], test[b["features"]], yte)
        importances[fs].to_csv(METRIC_DIR / f"permutation_importance_{fs}.csv", index=False)
    seg, seg_ct, seg_profile = T.kmeans_segments(cleaned)
    seg_ct.to_csv(METRIC_DIR / "kmeans_crosstab.csv"); seg_profile.to_csv(METRIC_DIR / "kmeans_profile.csv")
    R.model_figures(results, cms, importances, sel_tables, seg_ct, seg_profile)

    print("== 6. Save model bundle + HR report ==")
    joblib.dump({"bundles": bundles, "importances": importances, "reference": cleaned}, MODEL_DIR / "wlb_models.joblib")
    saved = save_all_models(fitted, results, mode="cleaned")      # one file per feature set x model
    print(f"  saved {len(saved)} individual models to {MODEL_DIR}")
    main_fs = "all_features"
    pred = R.predict_with_drivers(test, bundles[main_fs], cleaned, importances[main_fs])
    pred.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    p = R.build_hr_report(pred, results, importances[main_fs])
    (METRIC_DIR / "summary.json").write_text(json.dumps(
        {fs: {"model": b["model_name"], **{k: round(v, 4) for k, v in b["metrics"].items() if isinstance(v, float)}}
         for fs, b in bundles.items()}, indent=2))
    print(f"  report: {p}\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true", help="smaller tuning budget")
    main(ap.parse_args().fast)
