"""Cleaning, encoding and scaling.

Two cleaning modes implement the "with and without outliers" comparison planned in Phase-II:
  * "raw"     - keep every record, only add audit flags
  * "cleaned" - drop 0 h / 24 h work-hour entry errors, cap remaining work hours at 14 h
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import *


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flag_work_outlier"] = ((df[WORK] > MAX_PLAUSIBLE_WORK_HOURS) | (df[WORK] <= 0)).astype(int)
    df["flag_entry_error"] = df[WORK].isin([0, 24]).astype(int)
    df["flag_time_budget"] = (df[[WORK, REST, SLEEP, EXERCISE]].sum(axis=1) > DAY_HOURS).astype(int)
    return df


def clean(df: pd.DataFrame, mode: str = "cleaned") -> pd.DataFrame:
    df = add_flags(df)
    if mode == "raw":
        return df
    df = df[df["flag_entry_error"] == 0].copy()           # 0 h / 24 h = entry errors
    df[WORK] = df[WORK].clip(upper=WORK_CAP)               # cap remaining extreme values
    return df.reset_index(drop=True)


def build_preprocessor(features) -> ColumnTransformer:
    num = [f for f in features if f not in CATEGORICAL]
    cat = [f for f in features if f in CATEGORICAL]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def encode_target(s: pd.Series) -> pd.Series:
    return s.map(CLASS_TO_INT)
