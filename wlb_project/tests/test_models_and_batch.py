"""Tests for modelling, reporting, persistence and the dashboard's batch logic (no Streamlit needed).

Uses fast Logistic-Regression pipelines so the whole file runs in well under a minute.
"""
import io

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import *
from src.data_loader import load_raw
from src.preprocessing import build_preprocessor, clean, encode_target
from src.feature_selection import select
from src import train as T, reporting as R
from src.batch import (read_table, prepare_batch, risk_distribution, priority_list, style_risk)
from src.persistence import save_all_models, load_model, slug


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def trained(raw):
    """Fit one LR per feature set on the cleaned training split; evaluate on the cleaned test split."""
    train_raw, test_raw = T.split_data(raw)
    tr, te = clean(train_raw), clean(test_raw)
    ytr, yte = encode_target(tr[TARGET]), encode_target(te[TARGET])
    out = {"train": tr, "test": te, "ytest": yte, "models": {}}
    for fs, cols in FEATURE_SETS.items():
        feats, _ = select(tr[cols], ytr)
        pipe = Pipeline([("prep", build_preprocessor(feats)),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(tr[feats], ytr)
        metrics, _ = T.evaluate(pipe, te[feats], yte)
        out["models"][fs] = {"pipe": pipe, "features": feats, "metrics": metrics}
    return out


# ---------------- modelling claims made in the report ----------------
def test_feature_selection_drops_age_and_gender(trained):
    feats = trained["models"]["behavioural_only"]["features"]
    assert "age" not in feats and "gender" not in feats
    assert "avg_work_hours_per_day" in feats


def test_leakage_gap_between_feature_sets(trained):
    beh = trained["models"]["behavioural_only"]["metrics"]["macro_f1"]
    alls = trained["models"]["all_features"]["metrics"]["macro_f1"]
    assert 0.65 < beh < 0.80          # honest baseline (~0.73 in the report)
    assert alls > 0.95                # leaked target (~0.996 in the report)
    assert alls - beh > 0.15


def test_work_hours_is_top_behavioural_driver(trained):
    m = trained["models"]["behavioural_only"]
    tab = T.permutation_table(m["pipe"], trained["test"][m["features"]], trained["ytest"], n_repeats=3)
    assert tab.feature.iloc[0] == "avg_work_hours_per_day"
    assert tab.feature.iloc[1] == "avg_sleep_hours_per_day"


def test_kmeans_high_work_segment_is_riskiest(raw):
    c = clean(raw)
    seg, ct, profile = T.kmeans_segments(c)
    assert len(seg) == len(c) and ct.shape == (3, 3)
    share_high = ct["High Risk"] / ct.sum(axis=1)
    assert profile["avg_work_hours_per_day"].idxmax() == share_high.idxmax()
    assert share_high.max() > 0.5 and share_high.min() < 0.1


# ---------------- reporting ----------------
def test_predictions_drivers_and_hr_report(trained, tmp_path):
    m = trained["models"]["all_features"]
    bundle = {"pipeline": m["pipe"], "features": m["features"]}
    imp = T.permutation_table(m["pipe"], trained["test"][m["features"]], trained["ytest"], n_repeats=2)
    pred = R.predict_with_drivers(trained["test"].head(60), bundle, trained["train"], imp)
    assert {"predicted_risk", "p_low", "p_medium", "p_high", "top_risk_drivers"} <= set(pred.columns)
    assert set(pred.predicted_risk) <= set(CLASS_ORDER)
    assert np.allclose(pred[["p_low", "p_medium", "p_high"]].sum(axis=1), 1, atol=0.01)
    results = pd.DataFrame([{"data_mode": "cleaned", "feature_set": "all_features", "model": "LR", "accuracy": .9,
                             "macro_f1": .9, "high_risk_recall": .9, "roc_auc_ovr": .9}])
    path = R.build_hr_report(pred, results, imp, tmp_path / "r.html")
    text = path.read_text(encoding="utf-8")
    assert "Work-Life Balance Risk Report" in text and "<table>" in text


# ---------------- persistence ----------------
def test_save_and_reload_models(trained, tmp_path):
    m = trained["models"]["behavioural_only"]
    fitted = {("cleaned", "behavioural_only", "Logistic Regression"): (m["pipe"], m["features"]),
              ("raw", "behavioural_only", "Logistic Regression"): (m["pipe"], m["features"])}
    written = save_all_models(fitted, mode="cleaned", out_dir=tmp_path)
    assert len(written) == 1 and written[0].name == "behavioural_only__logistic_regression.joblib"
    back = load_model("behavioural_only", "Logistic Regression", tmp_path)
    X = trained["test"][m["features"]].head(25)
    assert (back["pipeline"].predict(X) == m["pipe"].predict(X)).all()
    assert slug("Random Forest") == "random_forest"


# ---------------- dashboard batch logic ----------------
FEATS = FEATURE_SETS["behavioural_only"]


def _batch():
    df = pd.DataFrame({
        ID_COL: [1, 2, 3, 4],
        "gender": ["Male"] * 4, "occupation_type": ["Engineer"] * 4, "age": [30] * 4,
        WORK: [8.0, 24.0, 16.0, np.nan], REST: [8.0, 2.0, 2.0, 5.0], SLEEP: [7.0, 7.0, 7.0, 7.0],
        EXERCISE: [1.0, 0.0, 0.0, 1.0], "commute_time_mins": [30] * 4,
        "family_time_hours": [3] * 4, "social_time_hours": [1] * 4})
    return df


def test_prepare_batch_flags_caps_and_skips():
    ready, info = prepare_batch(_batch(), FEATS)
    assert info["n_input"] == 4 and info["n_scored"] == 3 and info["n_skipped"] == 1
    assert info["n_entry_errors"] == 1 and info["n_work_capped"] == 2
    assert ready[WORK].max() == WORK_CAP
    notes = ready["data_quality_note"].tolist()
    assert notes[0] == "" and "check entry" in notes[1] and "capped" in notes[2]


def test_prepare_batch_missing_columns():
    ready, info = prepare_batch(_batch().drop(columns=[SLEEP]), FEATS)
    assert ready is None and info["missing_columns"] == [SLEEP]


def test_read_table_csv_and_excel(tmp_path):
    df = _batch()
    csv = tmp_path / "a.csv"; df.to_csv(csv, index=False)
    assert read_table(str(csv)).shape == df.shape
    buf = io.BytesIO(); df.to_excel(buf, index=False, sheet_name="WLB Dataset"); buf.seek(0)
    assert read_table(buf, "x.xlsx").shape == df.shape
    buf = io.BytesIO(); df.to_excel(buf, index=False, sheet_name="Other"); buf.seek(0)   # sheet-name fallback
    assert read_table(buf, "x.xlsx").shape == df.shape


def test_real_dataset_xlsx_is_readable_by_batch_reader(raw):
    df = read_table(str(RAW_PATH))
    ready, info = prepare_batch(df, FEATS)
    assert info["n_input"] == 10000 and info["n_scored"] == 10000
    assert info["n_entry_errors"] == 66 and info["n_time_budget"] == 960


def test_distribution_priority_and_styling():
    pred = pd.DataFrame({"employee_id": [1, 2, 3, 4], "predicted_risk": ["High Risk", "Low Risk", "High Risk", "Low Risk"],
                         "p_high": [0.6, 0.0, 0.9, 0.1]})
    d = risk_distribution(pred)
    assert list(d.index) == CLASS_ORDER and d.tolist() == [2, 0, 2]
    assert priority_list(pred).employee_id.tolist() == [3, 1]
    html = style_risk(pred).to_html()
    assert PALETTE_HIGH in html


from src.reporting import PALETTE
PALETTE_HIGH = PALETTE["High Risk"]
