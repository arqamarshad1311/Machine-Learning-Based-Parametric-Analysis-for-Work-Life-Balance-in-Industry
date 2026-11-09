"""Figures, risk-driver explanation and exportable HR report."""
import datetime as dt
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .config import *

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Low Risk": "#2e9e6b", "Medium Risk": "#e8a317", "High Risk": "#d1453b"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def eda_figures(df_raw: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    d = df_raw[TARGET].value_counts(normalize=True).reindex(CLASS_ORDER) * 100
    ax[0].bar(d.index, d.values, color=[PALETTE[c] for c in d.index]); ax[0].set_title("Target distribution (%)")
    for i, v in enumerate(d.values): ax[0].text(i, v + .5, f"{v:.0f}%", ha="center")
    r = df_raw[NUMERIC_ALL].corrwith(df_raw[TARGET].map(CLASS_TO_INT)).sort_values()
    ax[1].barh(r.index, r.values, color=["#d1453b" if v > 0 else "#2f6fb5" for v in r.values])
    ax[1].set_title("Correlation with ordinal WLB risk")
    _save(fig, "01_target_and_correlation.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    tmp = df_raw[NUMERIC_ALL].copy(); tmp["risk (ordinal)"] = df_raw[TARGET].map(CLASS_TO_INT)
    sns.heatmap(tmp.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap"); _save(fig, "02_correlation_heatmap.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(x=df_raw[WORK], ax=ax[0], color="#9ec5e8"); ax[0].set_title("Daily work hours - outliers (0 h and 24 h)")
    budget = df_raw[[WORK, REST, SLEEP, EXERCISE]].sum(axis=1)
    ax[1].hist(budget, bins=50, color="#9ec5e8"); ax[1].axvline(24, color="red", ls="--")
    ax[1].set_title("Work + rest + sleep + exercise (hours)")
    _save(fig, "03_data_quality.png")


def model_figures(results, cms, importances, sel_tables, seg_ct, seg_profile):
    r = results.copy(); r["scenario"] = r.data_mode + " | " + r.feature_set
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.barplot(data=r, x="scenario", y="macro_f1", hue="model", ax=ax)
    ax.set_ylim(0.6, 1.01); ax.set_title("Test macro-F1 by scenario and model"); ax.tick_params(axis="x", rotation=12)
    _save(fig, "04_model_comparison.png")

    for fs, tab in importances.items():
        top = tab.head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.barh(top.feature, top.importance, xerr=top["std"], color="#2f6fb5")
        ax.set_title(f"Permutation importance (macro-F1 drop) - {fs}")
        _save(fig, f"05_permutation_importance_{fs}.png")

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, ((mode, fs, name), cm) in zip(axs.ravel(), [(k, v) for k, v in cms.items() if k[0] == "cleaned"]):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Low", "Med", "High"], yticklabels=["Low", "Med", "High"])
        ax.set_title(f"{name}\n{fs}", fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    _save(fig, "06_confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    (seg_ct.div(seg_ct.sum(axis=1), axis=0) * 100).plot(kind="bar", stacked=True, ax=ax,
                                                       color=[PALETTE[c] for c in CLASS_ORDER])
    ax.set_title("K-Means segments vs. true risk (%)"); ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=0)
    _save(fig, "07_kmeans_segments.png")


# ---------- explanation of individual predictions (model-agnostic) ----------
def risk_drivers(row: pd.Series, ref: pd.DataFrame, importance: pd.DataFrame, n: int = 3) -> str:
    """Top adverse parameters for one employee: z-score vs population, signed by correlation with risk."""
    corr = ref[NUMERIC_ALL].corrwith(ref[TARGET].map(CLASS_TO_INT)) if TARGET in ref else None
    out = []
    for f in importance.feature:
        if f not in row.index or f in CATEGORICAL or corr is None or f not in corr.index:
            continue
        z = (row[f] - ref[f].mean()) / (ref[f].std() or 1)
        adverse = z * np.sign(corr[f])
        if adverse > 0.5:
            out.append((adverse * max(importance.set_index("feature").importance.get(f, 0), 1e-6), f, row[f]))
    out.sort(reverse=True)
    return "; ".join(f"{f.replace('_', ' ')} = {v:g}" for _, f, v in out[:n]) or "no single dominant driver"


def predict_with_drivers(df_in: pd.DataFrame, bundle: dict, ref: pd.DataFrame, importance: pd.DataFrame) -> pd.DataFrame:
    X = df_in[bundle["features"]]
    proba = bundle["pipeline"].predict_proba(X)
    out = df_in[[ID_COL]].copy() if ID_COL in df_in else pd.DataFrame(index=df_in.index)
    out["predicted_risk"] = [CLASS_ORDER[i] for i in proba.argmax(1)]
    for i, c in enumerate(CLASS_ORDER):
        out[f"p_{c.split()[0].lower()}"] = proba[:, i].round(3)
    out["top_risk_drivers"] = [risk_drivers(r, ref, importance) for _, r in df_in.iterrows()]
    return out


def build_hr_report(pred: pd.DataFrame, results: pd.DataFrame, importance: pd.DataFrame, path=None):
    """Self-contained HTML report that HR can open, print to PDF, or share."""
    path = path or REPORT_DIR / "hr_report.html"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dist = pred.predicted_risk.value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    hi = pred[pred.predicted_risk == "High Risk"].sort_values("p_high", ascending=False).head(25)
    has_note = "data_quality_note" in pred.columns           # present for dashboard uploads
    rows = "".join(f"<tr><td>{r.get(ID_COL, '')}</td><td>{r.p_high:.0%}</td><td>{html.escape(r.top_risk_drivers)}</td>"
                   + (f"<td>{html.escape(str(r.get('data_quality_note', '') or ''))}</td>" if has_note else "") + "</tr>"
                   for _, r in hi.iterrows())
    note_th = "<th>Data note</th>" if has_note else ""
    imp = "".join(f"<tr><td>{r.feature}</td><td>{r.importance:.3f}</td></tr>" for _, r in importance.head(8).iterrows())
    best = results.sort_values("macro_f1", ascending=False).head(6)
    mt = "".join(f"<tr><td>{r.data_mode}</td><td>{r.feature_set}</td><td>{r.model}</td><td>{r.accuracy:.3f}</td>"
                 f"<td>{r.macro_f1:.3f}</td><td>{r.high_risk_recall:.3f}</td><td>{r.roc_auc_ovr:.3f}</td></tr>"
                 for _, r in best.iterrows())
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;color:#222}"
           "h1{color:#0f5a8a}table{border-collapse:collapse;width:100%;margin:10px 0 24px}"
           "td,th{border:1px solid #ccc;padding:6px 9px;text-align:left;font-size:14px}th{background:#eef3f8}"
           ".k{display:flex;gap:14px}.c{flex:1;padding:14px;border-radius:8px;color:#fff;text-align:center}"
           ".c b{font-size:28px;display:block}")
    cards = "".join(f'<div class="c" style="background:{PALETTE[c]}"><b>{dist[c]}</b>{c}</div>' for c in CLASS_ORDER)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>WLB HR Report</title><style>{css}</style></head><body>
<h1>Work-Life Balance Risk Report</h1>
<p>Generated {dt.date.today():%d %b %Y} &middot; {len(pred)} employees analysed &middot; Team Code Sentinels, Graphic Era (Deemed to be University)</p>
<div class="k">{cards}</div>
<h2>Parameters with the strongest influence on WLB risk</h2>
<table><tr><th>Parameter</th><th>Permutation importance (macro-F1 drop)</th></tr>{imp}</table>
<h2>Employees to prioritise (highest predicted High-Risk probability)</h2>
<table><tr><th>Employee ID</th><th>P(High risk)</th><th>Main adverse parameters</th>{note_th}</tr>{rows}</table>
<h2>Model validation summary (held-out test set)</h2>
<table><tr><th>Data</th><th>Features</th><th>Model</th><th>Acc</th><th>Macro-F1</th><th>High recall</th><th>ROC-AUC</th></tr>{mt}</table>
<p><i>Note: the label in this dataset is a derived/synthetic score; results show how well the parameters explain it,
not real-world clinical outcomes. Use as a decision-support aid, not an automated decision.</i></p></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path
