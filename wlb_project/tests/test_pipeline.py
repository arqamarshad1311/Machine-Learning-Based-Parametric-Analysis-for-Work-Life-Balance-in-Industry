import pandas as pd
import pytest
from src.config import *
from src.data_loader import load_raw
from src.audit import run_audit
from src.preprocessing import clean, build_preprocessor


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_schema_and_quality(raw):
    a = run_audit(raw)
    assert a["n_rows"] == 10000 and a["n_cols"] == 16
    assert a["missing_values"] == 0 and a["duplicate_rows"] == 0
    assert a["target_distribution"] == {"Low Risk": 0.4, "Medium Risk": 0.35, "High Risk": 0.25}


def test_known_issues_match_report(raw):
    a = run_audit(raw)
    assert (a["work_hours_gt_14"], a["work_hours_eq_24"], a["work_hours_eq_0"]) == (222, 29, 37)
    assert a["time_budget_gt_24"] > 0


def test_cleaning(raw):
    c = clean(raw, "cleaned")
    assert not c[WORK].isin([0, 24]).any() and c[WORK].max() <= WORK_CAP
    assert len(clean(raw, "raw")) == len(raw)


def test_preprocessor_shapes(raw):
    c = clean(raw)
    X = build_preprocessor(FEATURE_SETS["all_features"]).fit_transform(c[FEATURE_SETS["all_features"]])
    assert X.shape[0] == len(c) and X.shape[1] == 12 + 2 + 14   # 12 numeric + 2 gender + 14 occupations
