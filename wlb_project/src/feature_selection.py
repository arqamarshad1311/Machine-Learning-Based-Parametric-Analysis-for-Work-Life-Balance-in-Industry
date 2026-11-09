"""Correlation + mutual-information based feature ranking and selection."""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from .config import *

CORR_MIN, MI_MIN = 0.05, 0.01


def rank_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Rank candidate features by |Spearman r| with ordinal risk and mutual information."""
    Xn = X.copy()
    cat = [c for c in Xn.columns if c in CATEGORICAL]
    for c in cat:
        Xn[c] = Xn[c].astype("category").cat.codes
    mi = mutual_info_classif(Xn, y, discrete_features=[c in cat for c in Xn.columns],
                             random_state=RANDOM_STATE)
    corr = [abs(pd.Series(Xn[c]).corr(y, method="spearman")) if c not in cat else np.nan for c in Xn.columns]
    out = pd.DataFrame({"feature": Xn.columns, "abs_spearman": corr, "mutual_info": mi})
    out["selected"] = (out["abs_spearman"].fillna(0) >= CORR_MIN) | (out["mutual_info"] >= MI_MIN)
    return out.sort_values("mutual_info", ascending=False).reset_index(drop=True)


def select(X: pd.DataFrame, y: pd.Series):
    table = rank_features(X, y)
    return table[table.selected].feature.tolist(), table
