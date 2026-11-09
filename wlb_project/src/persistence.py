"""Save / load every trained model so batch scoring does not need re-training.

`run_pipeline.py` writes one file per (feature set, model) for the cleaned-data experiments, e.g.
`outputs/models/behavioural_only__logistic_regression.joblib`. Each file is a dict holding the fitted
scikit-learn Pipeline (pre-processor + classifier), the feature list it expects and its test metrics.

Note: joblib files are tied to the library versions used for training. If scikit-learn / pandas differ on
another machine, simply re-run `python run_pipeline.py` to regenerate them.
"""
import re
from pathlib import Path

import joblib

from .config import MODEL_DIR


def slug(name: str) -> str:
    """'Logistic Regression' -> 'logistic_regression'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def model_path(feature_set: str, model_name: str, out_dir: Path = MODEL_DIR) -> Path:
    return Path(out_dir) / f"{feature_set}__{slug(model_name)}.joblib"


def save_all_models(fitted: dict, results=None, mode: str = "cleaned", out_dir: Path = MODEL_DIR) -> list:
    """Persist every fitted model of one data mode.

    fitted  : {(mode, feature_set, model_name): (pipeline, feature_list)}  as returned by run_experiments
    results : optional model_comparison DataFrame, used to attach the held-out test metrics
    Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for (m, fs, name), (pipe, feats) in fitted.items():
        if m != mode:
            continue
        metrics = {}
        if results is not None:
            r = results[(results.data_mode == m) & (results.feature_set == fs) & (results.model == name)]
            if len(r):
                metrics = {k: float(r.iloc[0][k]) for k in ("accuracy", "macro_f1", "high_risk_recall", "roc_auc_ovr")}
        path = model_path(fs, name, out_dir)
        joblib.dump({"pipeline": pipe, "features": list(feats), "model_name": name, "feature_set": fs,
                     "data_mode": m, "metrics": metrics}, path, compress=3)
        written.append(path)
    return written


def load_model(feature_set: str, model_name: str, out_dir: Path = MODEL_DIR) -> dict:
    return joblib.load(model_path(feature_set, model_name, out_dir))
