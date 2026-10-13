#!/usr/bin/env python3
"""Aggregate the tiered nested-CV predictions into reportable tables.

Metrics are computed per repeat by pooling that repeat's five outer folds, so the
spread across repeats is the resampling uncertainty that the confidence intervals
report. Threshold is fixed at 0.5 throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

KEY = ["Task", "Tier", "Variant", "FeatureSet", "Model"]
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "MCC",
           "ROC_AUC", "PR_AUC"]


def metrics_at(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, yh),
                Precision=precision_score(y, yh, zero_division=0),
                Recall=recall_score(y, yh, zero_division=0),
                Specificity=tn / (tn + fp) if (tn + fp) else np.nan,
                F1=f1_score(y, yh, zero_division=0),
                MCC=matthews_corrcoef(y, yh),
                ROC_AUC=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                PR_AUC=average_precision_score(y, p) if len(set(y)) > 1 else np.nan,
                TP=tp, FP=fp, TN=tn, FN=fn, N=len(y))


def per_repeat_table(preds):
    rows = []
    for k, g in preds.groupby(KEY + ["Repeat"], sort=False):
        rows.append(dict(zip(KEY + ["Repeat"], k),
                         **metrics_at(g.y_true.values, g.y_prob.values)))
    return pd.DataFrame(rows)


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def summarise(per_rep):
    rows = []
    for k, g in per_rep.groupby(KEY, sort=False):
        r = dict(zip(KEY, k), n_repeats=len(g))
        for m in METRICS:
            lo, hi = ci95(g[m])
            r[m] = g[m].mean()
            r[f"{m}_lo"] = lo
            r[f"{m}_hi"] = hi
        rows.append(r)
    return pd.DataFrame(rows)


def paired_tests(per_rep, within, compare, metric="MCC"):
    """Paired t-tests on per-repeat values, Holm-corrected within each family."""
    out = []
    keys = [c for c in KEY if c not in compare]
    for k, g in per_rep.groupby(keys, sort=False):
        piv = g.pivot_table(index="Repeat", columns=compare, values=metric)
        levels = list(piv.columns)
        fam = []
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                a, b = piv[levels[i]].dropna(), piv[levels[j]].dropna()
                idx = a.index.intersection(b.index)
                if len(idx) < 3:
                    continue
                d = (a.loc[idx] - b.loc[idx]).values
                t, p = stats.ttest_rel(a.loc[idx], b.loc[idx])
                lo, hi = ci95(d)
                fam.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)),
                                comparison=f"{levels[i]} - {levels[j]}", metric=metric,
                                delta=d.mean(), delta_lo=lo, delta_hi=hi,
                                p_raw=p, n_repeats=len(idx)))
        ordered = sorted(fam, key=lambda r: r["p_raw"])
        m, running = len(ordered), 0.0
        for rank, row in enumerate(ordered):
            running = min(1.0, max(running, row["p_raw"] * (m - rank)))
            row["p_holm"] = running
        out += fam
    return pd.DataFrame(out)


def main():
    preds = pd.read_csv("out2/tiered_per_fold_predictions.csv.gz")
    per_rep = per_repeat_table(preds)
    per_rep.to_csv("out2/tiered_per_repeat_metrics.csv", index=False)
    summ = summarise(per_rep).sort_values(KEY)
    summ.to_csv("out2/tiered_summary_metrics.csv", index=False)
    t_model = paired_tests(per_rep, within=None, compare="Model")
    t_feat = paired_tests(per_rep, within=None, compare="FeatureSet")
    t_var = paired_tests(per_rep, within=None, compare="Variant")
    tests = pd.concat([t_model.assign(family="model"),
                       t_feat.assign(family="feature_set"),
                       t_var.assign(family="matching")], ignore_index=True)
    tests.to_csv("out2/tiered_model_comparison_stats.csv", index=False)
    print("designs:", summ.groupby(["Task", "Tier", "Variant"]).ngroups,
          "| rows:", len(summ), "| tests:", len(tests))


if __name__ == "__main__":
    main()
