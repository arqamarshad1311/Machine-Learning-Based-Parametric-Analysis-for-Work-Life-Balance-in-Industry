"""Batch-scoring helpers used by the dashboard (deliberately free of Streamlit so they can be unit-tested).

Responsibilities
  * read an uploaded CSV / Excel file (the report's installation guide uploads WLB_PBL_DATASET.xlsx)
  * check that the columns the selected model needs are present and numeric
  * apply the same work-hour cap as the training pipeline and add a plain-English data-quality note per employee
  * summarise predictions (risk distribution, priority list, colour styling)
"""
import numpy as np
import pandas as pd

from .config import *
from .reporting import PALETTE


def read_table(source, filename: str = None) -> pd.DataFrame:
    """Read a CSV or Excel file (path or file-like object). For Excel the 'WLB Dataset' sheet is used if present,
    otherwise the first sheet."""
    name = (filename or str(source)).lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(source, sheet_name=RAW_SHEET)
        except ValueError:                       # sheet name not found -> fall back to the first sheet
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_excel(source)
    return pd.read_csv(source)


def prepare_batch(df: pd.DataFrame, features: list):
    """Validate and lightly clean a batch for scoring.

    Returns (ready, info):
      ready - DataFrame of scorable rows (numeric columns coerced, work hours capped at WORK_CAP) with a
              `data_quality_note` column, or None if required columns are missing
      info  - counts for the dashboard: rows read / scored / skipped and the number of flagged rows
    Unlike training, rows with 0 h / 24 h work hours are NOT dropped (HR still wants a score for every person);
    they are flagged so the entry can be checked.
    """
    info = {"n_input": int(len(df)), "missing_columns": [f for f in features if f not in df.columns],
            "n_scored": 0, "n_skipped": 0, "n_entry_errors": 0, "n_work_capped": 0, "n_time_budget": 0}
    if info["missing_columns"]:
        return None, info

    work = df.copy()
    for c in (f for f in features if f not in CATEGORICAL):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    unscorable = work[features].isna().any(axis=1)
    info["n_skipped"] = int(unscorable.sum())
    work = work.loc[~unscorable].copy()
    info["n_scored"] = int(len(work))

    note = pd.Series("", index=work.index, dtype=object)

    def add(mask, text):
        mask = mask.reindex(work.index, fill_value=False)
        note.loc[mask] = np.where(note.loc[mask] == "", text, note.loc[mask] + "; " + text)

    # time-budget check uses the ORIGINAL hours (before capping) so it matches the audit (960 rows on the full dataset)
    if all(c in work for c in (WORK, REST, SLEEP, EXERCISE)):
        budget = work[[WORK, REST, SLEEP, EXERCISE]].sum(axis=1) > DAY_HOURS
        add(budget, "work+rest+sleep+exercise > 24 h")
        info["n_time_budget"] = int(budget.sum())
    if WORK in work:
        entry = work[WORK].isin([0, 24])
        over = work[WORK] > MAX_PLAUSIBLE_WORK_HOURS
        add(entry, "work hours 0 or 24 - check entry")
        add(over & ~entry, f"work hours capped at {WORK_CAP}")
        info["n_entry_errors"], info["n_work_capped"] = int(entry.sum()), int(over.sum())
        work[WORK] = work[WORK].clip(upper=WORK_CAP)
    work["data_quality_note"] = note
    return work, info


def risk_distribution(pred: pd.DataFrame) -> pd.Series:
    """Employees per predicted risk class, always in Low / Medium / High order."""
    return pred["predicted_risk"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)


def priority_list(pred: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Highest predicted High-Risk probability first - the employees HR should talk to first."""
    hi = pred[pred["predicted_risk"] == "High Risk"].sort_values("p_high", ascending=False)
    return hi.head(n).reset_index(drop=True)


def distribution_figure(pred: pd.DataFrame):
    """Colour-coded bar chart of the predicted risk distribution (matplotlib Figure)."""
    import matplotlib.pyplot as plt
    d = risk_distribution(pred)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(d.index, d.values, color=[PALETTE[c] for c in d.index])
    for i, v in enumerate(d.values):
        ax.text(i, v, f"{v}", ha="center", va="bottom")
    ax.set_ylabel("Employees"); ax.set_title("Predicted WLB risk distribution")
    fig.tight_layout()
    return fig


def style_risk(pred: pd.DataFrame):
    """pandas Styler that colour-codes the predicted_risk column (green / amber / red)."""
    def colour(v):
        c = PALETTE.get(v)
        return f"background-color:{c};color:white;font-weight:600" if c else ""
    styler = pred.style
    apply = getattr(styler, "map", None) or styler.applymap      # Styler.map needs pandas>=2.1
    return apply(colour, subset=["predicted_risk"])
