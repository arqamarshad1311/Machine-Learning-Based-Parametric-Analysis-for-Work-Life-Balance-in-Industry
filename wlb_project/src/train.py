"""Model development (M3): Logistic Regression, Random Forest, XGBoost + K-Means.

Experiment grid (from the Phase-II plan):
    data mode    : raw (with outliers)  vs  cleaned (entry errors dropped, hours capped)
    feature set  : behavioural_only     vs  all_features   (leakage-aware comparison)
Every model is tuned with stratified cross-validation on macro-F1 and uses class weights.
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
try:                                   # XGBoost is optional at import time so the rest of the pipeline stays usable
    from xgboost import XGBClassifier
except ImportError:                    # pragma: no cover - depends on the environment
    XGBClassifier = None

from .config import *
from .feature_selection import select
from .preprocessing import build_preprocessor, clean, encode_target


def model_zoo(fast: bool):
    """Return {name: (estimator, search grid, n_iter)}. XGBoost is skipped (with a warning) if not installed."""
    n_iter = 3 if fast else 8
    zoo = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10, 100]}, 5),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [6, 10, 16, None],
             "clf__min_samples_leaf": [1, 3, 5, 10]}, n_iter),
        "XGBoost": (
            XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                          random_state=RANDOM_STATE, n_jobs=-1) if XGBClassifier else None,
            {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [3, 4, 6],
             "clf__learning_rate": [0.03, 0.05, 0.1, 0.2], "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.7, 0.85, 1.0]}, n_iter),
    }
    if XGBClassifier is None:
        print("WARNING: xgboost is not installed - skipping the XGBoost model (pip install xgboost).")
        zoo.pop("XGBoost")
    return zoo


def evaluate(pipe, X, y):
    proba = pipe.predict_proba(X)
    pred = proba.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "high_risk_recall": recall_score(y, pred, labels=[2], average=None)[0],
        "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
    }, confusion_matrix(y, pred, labels=[0, 1, 2])


def split_data(df: pd.DataFrame):
    return train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE)


def run_experiments(df_raw: pd.DataFrame, fast: bool = False, verbose: bool = True):
    train_raw, test_raw = split_data(df_raw)
    test = clean(test_raw, "cleaned")             # both modes are judged on the same, clean inputs
    yte = encode_target(test[TARGET])
    cv = StratifiedKFold(5 if not fast else 3, shuffle=True, random_state=RANDOM_STATE)
    results, cms, fitted, sel_tables, importances = [], {}, {}, {}, {}

    for mode in ["cleaned", "raw"]:
        tr = clean(train_raw, mode)
        ytr = encode_target(tr[TARGET])
        for fs_name, fs_cols in FEATURE_SETS.items():
            feats, sel_table = select(tr[fs_cols], ytr)
            sel_tables[(mode, fs_name)] = sel_table
            for name, (est, grid, n_iter) in model_zoo(fast).items():
                t0 = time.time()
                pipe = Pipeline([("prep", build_preprocessor(feats)), ("clf", est)])
                search = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, int(np.prod([len(v) for v in grid.values()]))),
                                            scoring="f1_macro", cv=cv, random_state=RANDOM_STATE, n_jobs=1)
                fit_kw = {}
                if name == "XGBoost":      # XGBoost has no class_weight -> balanced sample weights
                    fit_kw["clf__sample_weight"] = compute_sample_weight("balanced", ytr)
                search.fit(tr[feats], ytr, **fit_kw)
                best = search.best_estimator_
                m, cm = evaluate(best, test[feats], yte)
                row = {"data_mode": mode, "feature_set": fs_name, "model": name,
                       "cv_macro_f1": search.best_score_, **m, "n_train": len(tr),
                       "n_features": len(feats), "best_params": str(search.best_params_),
                       "seconds": round(time.time() - t0, 1)}
                results.append(row)
                cms[(mode, fs_name, name)] = cm
                fitted[(mode, fs_name, name)] = (best, feats)
                if verbose:
                    print(f"[{mode:7s}|{fs_name:16s}|{name:19s}] cv={row['cv_macro_f1']:.3f} "
                          f"acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} highR={m['high_risk_recall']:.3f} "
                          f"auc={m['roc_auc_ovr']:.3f} ({row['seconds']}s)", flush=True)
    return pd.DataFrame(results), cms, fitted, sel_tables, test


def best_per_feature_set(results: pd.DataFrame, fitted: dict, mode="cleaned"):
    """Pick the best (by test macro-F1) model of each feature set for the deliverable app."""
    out = {}
    for fs in FEATURE_SETS:
        r = results[(results.data_mode == mode) & (results.feature_set == fs)].sort_values("macro_f1", ascending=False).iloc[0]
        model, feats = fitted[(mode, fs, r.model)]
        out[fs] = {"model_name": r.model, "pipeline": model, "features": feats, "metrics": r.to_dict()}
    return out


def permutation_table(pipe, X, y, n_repeats=10):
    pi = permutation_importance(pipe, X, y, scoring="f1_macro", n_repeats=n_repeats,
                                random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "std": pi.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def kmeans_segments(df_clean: pd.DataFrame, k: int = 3):
    """Optional unsupervised risk segmentation on behavioural parameters."""
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in BEHAVIOURAL_NUM if c != "age"]
    Xs = StandardScaler().fit_transform(df_clean[cols])
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    seg = pd.Series(km.labels_, name="segment")
    ct = pd.crosstab(seg, df_clean[TARGET]).reindex(columns=CLASS_ORDER)
    profile = df_clean.assign(segment=km.labels_).groupby("segment")[cols].mean().round(2)
    return seg, ct, profile
