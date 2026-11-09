"""Interactive WLB dashboard.   Run:  streamlit run dashboard/app.py   (after python run_pipeline.py)

Tabs: Overview | Predict one employee | Batch + HR report | Model comparison | Feature importance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st
from src.config import *
from src import reporting as R
from src.batch import (read_table, prepare_batch, risk_distribution, priority_list,
                       distribution_figure, style_risk)

st.set_page_config(page_title="WLB Risk Analyzer", page_icon="⚖️", layout="wide")


@st.cache_resource
def load():
    p = MODEL_DIR / "wlb_models.joblib"
    if not p.exists():
        st.error("Run `python run_pipeline.py` first to train the models."); st.stop()
    return joblib.load(p)


art = load()
ref = art["reference"]
results = pd.read_csv(METRIC_DIR / "model_comparison.csv")

st.title("⚖️ Work-Life Balance Risk Analyzer")
st.caption("Team Code Sentinels · MCA (AI & DS) · Graphic Era (Deemed to be University) · Mentor: Mr. Jaishankar Bhatt")

fs_label = {"behavioural_only": "Behavioural parameters only (no leakage)",
            "all_features": "All features (incl. stress, health, satisfaction)"}
fs = st.sidebar.radio("Feature set", list(fs_label), format_func=fs_label.get, index=0)
bundle, imp = art["bundles"][fs], art["importances"][fs]
st.sidebar.metric("Model used", bundle["model_name"])
st.sidebar.metric("Test macro-F1", f"{bundle['metrics']['macro_f1']:.3f}")
st.sidebar.metric("High-risk recall", f"{bundle['metrics']['high_risk_recall']:.3f}")
st.sidebar.caption("Predictions are decision-support signals, not verdicts. Interventions should be chosen by HR "
                   "together with the employee.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Predict one employee", "Batch + HR report",
                                         "Model comparison", "Feature importance"])

with tab1:
    c = st.columns(4)
    c[0].metric("Employees (cleaned)", f"{len(ref):,}")
    for col, k in zip(c[1:], CLASS_ORDER):
        col.metric(k, f"{(ref[TARGET] == k).mean():.0%}")
    st.image(str(FIG_DIR / "01_target_and_correlation.png"))
    st.image(str(FIG_DIR / "03_data_quality.png"), caption="Data-quality audit: entry errors and time-budget violations")
    occ = ref.groupby("occupation_type")[TARGET].apply(lambda s: (s == "High Risk").mean()).sort_values()
    st.subheader("High-risk share by occupation"); st.bar_chart(occ)

with tab2:
    st.write("Enter measurable parameters; the model predicts the WLB risk category.")
    feats, vals = bundle["features"], {}
    cols = st.columns(3)
    for i, f in enumerate(feats):
        with cols[i % 3]:
            if f in CATEGORICAL:
                vals[f] = st.selectbox(f, sorted(ref[f].unique()))
            else:
                lo, hi = float(ref[f].min()), float(ref[f].max())
                vals[f] = st.slider(f, lo, hi, float(ref[f].median()), step=1.0 if ref[f].dtype.kind == "i" else 0.1)
    row = pd.DataFrame([vals])
    proba = bundle["pipeline"].predict_proba(row[feats])[0]
    pred = CLASS_ORDER[int(proba.argmax())]
    show = {"Low Risk": st.success, "Medium Risk": st.warning, "High Risk": st.error}[pred]   # colour-coded result
    show(f"Predicted: **{pred}**  (confidence {proba.max():.0%})")
    st.bar_chart(pd.Series(proba, index=CLASS_ORDER, name="probability"))
    st.info("Main adverse parameters: " + R.risk_drivers(row.iloc[0], ref, imp))
    total = sum(vals.get(k, 0) for k in (WORK, REST, SLEEP, EXERCISE))
    if total > DAY_HOURS:
        st.warning(f"Work + rest + sleep + exercise = {total:.1f} h (> 24 h): check the inputs.")

with tab3:
    st.write("Upload a batch of employee records (CSV or Excel, e.g. `WLB_PBL_DATASET.xlsx`) to see the predicted "
             "risk distribution, the drivers for each person and a downloadable HR report.")
    st.caption("Required columns for the selected model: " + ", ".join(bundle["features"]))
    st.caption("Tip: remove or pseudonymise names before upload and run the dashboard inside your own secure environment.")
    up = st.file_uploader("Employee file", type=["csv", "xlsx"])
    df_in = None
    try:
        df_in = read_table(up, up.name) if up else ref.sample(300, random_state=1)
        if not up:
            st.caption("No file uploaded - showing 300 sample employees.")
    except Exception as exc:                               # unreadable / wrong file
        st.error(f"Could not read the file: {exc}")

    if df_in is not None:
        ready, info = prepare_batch(df_in, bundle["features"])
        if ready is None:
            st.error(f"Missing columns: {info['missing_columns']}")
        else:
            if info["n_skipped"]:
                st.warning(f"{info['n_skipped']} of {info['n_input']} rows have missing / non-numeric values in the "
                           "model's columns and were not scored.")
            if info["n_entry_errors"] or info["n_time_budget"] or info["n_work_capped"]:
                st.info(f"Data-quality flags: {info['n_entry_errors']} rows with 0 h or 24 h work, "
                        f"{info['n_work_capped']} rows above {MAX_PLAUSIBLE_WORK_HOURS} h (capped at {WORK_CAP} h), "
                        f"{info['n_time_budget']} rows whose daily hours add up to more than 24 h. "
                        "See the `data_quality_note` column.")
            pred = R.predict_with_drivers(ready, bundle, ref, imp)
            pred["data_quality_note"] = ready["data_quality_note"].values

            dist = risk_distribution(pred)
            m = st.columns(4)
            m[0].metric("Employees scored", f"{len(pred):,}")
            for col, k in zip(m[1:], CLASS_ORDER):
                col.metric(k, f"{dist[k]:,}  ({dist[k] / max(len(pred), 1):.0%})")
            left, right = st.columns([1, 1])
            with left:
                st.pyplot(distribution_figure(pred))
            with right:
                st.subheader("Top parameters driving risk")
                st.bar_chart(imp.head(8).set_index("feature").importance)
            with st.expander("Correlation heatmap (training data)"):
                st.image(str(FIG_DIR / "02_correlation_heatmap.png"))

            st.subheader("Employees - predicted risk and main drivers")
            st.dataframe(style_risk(pred), width="stretch")
            st.subheader("Priority list (highest predicted High-Risk probability)")
            st.dataframe(priority_list(pred), width="stretch")

            st.download_button("Download predictions (CSV)", pred.to_csv(index=False), "wlb_predictions.csv",
                               mime="text/csv")
            path = R.build_hr_report(pred, results, imp, REPORT_DIR / "hr_report_dashboard.html")
            st.download_button("Export HR report (HTML → print to PDF)", path.read_text(encoding="utf-8"),
                               "wlb_hr_report.html", mime="text/html")

with tab4:
    st.dataframe(results.drop(columns=["best_params"]).round(3), width="stretch")
    st.image(str(FIG_DIR / "04_model_comparison.png"))
    st.image(str(FIG_DIR / "06_confusion_matrices.png"), caption="Confusion matrices (cleaned data)")
    st.warning("All-feature models score ~0.94-1.00 because the label is derived from these same fields (target leakage). "
               "The behavioural-only models (~0.72-0.73) are the honest estimate of predicting risk from measurable habits.")

with tab5:
    st.subheader("Permutation importance (macro-F1 drop)")
    st.bar_chart(imp.set_index("feature").importance)
    st.image(str(FIG_DIR / "02_correlation_heatmap.png"))
    st.image(str(FIG_DIR / "07_kmeans_segments.png"), caption="Optional K-Means segmentation")
