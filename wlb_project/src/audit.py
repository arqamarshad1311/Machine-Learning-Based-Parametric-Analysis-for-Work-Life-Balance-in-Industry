"""Data-quality audit: reproduces the Phase-II testing & validation table."""
import json
import pandas as pd
from .config import *


def run_audit(df: pd.DataFrame) -> dict:
    budget = df[[WORK, REST, SLEEP, EXERCISE]].sum(axis=1)
    dist = df[TARGET].value_counts(normalize=True).reindex(CLASS_ORDER).round(3)
    res = {
        "n_rows": len(df), "n_cols": df.shape[1],
        "schema_ok": set(ALL_COLUMNS) == set(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[ID_COL].duplicated().sum()),
        "target_distribution": dist.to_dict(),
        "work_hours_gt_14": int((df[WORK] > MAX_PLAUSIBLE_WORK_HOURS).sum()),
        "work_hours_eq_24": int((df[WORK] == 24).sum()),
        "work_hours_eq_0": int((df[WORK] == 0).sum()),
        "time_budget_gt_24": int((budget > DAY_HOURS).sum()),
        "ordinal_corr_with_risk": (df[NUMERIC_ALL].corrwith(df[TARGET].map(CLASS_TO_INT)).round(2).to_dict()),
    }
    res["tests"] = {
        "Schema and data-type check": "Pass" if res["schema_ok"] else "Fail",
        "Missing-value check": "Pass" if res["missing_values"] == 0 else "Fail",
        "Duplicate check": "Pass" if res["duplicate_rows"] + res["duplicate_ids"] == 0 else "Fail",
        "Range/plausibility check": "Fail (flagged)" if res["work_hours_gt_14"] + res["work_hours_eq_0"] else "Pass",
        "Daily time-budget consistency": "Fail (flagged)" if res["time_budget_gt_24"] else "Pass",
    }
    return res


def save_audit(res: dict):
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "data_audit.json").write_text(json.dumps(res, indent=2))
