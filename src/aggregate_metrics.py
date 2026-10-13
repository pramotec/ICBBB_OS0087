"""
aggregate_metrics.py -- turn per-fold predictions into the revised tables.

Metrics are computed per REPEAT (pooling that repeat's five outer test folds,
so every peptide contributes exactly once per repeat). The 10 repeat values
give the mean and the 95% confidence interval reported in the revised tables;
no single number is quoted without its interval.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "MCC",
           "ROC_AUC", "PR_AUC"]


def metrics_at(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(
        Accuracy=accuracy_score(y, yh),
        Precision=precision_score(y, yh, zero_division=0),
        Recall=recall_score(y, yh, zero_division=0),
        Specificity=tn / (tn + fp) if (tn + fp) else np.nan,
        F1=f1_score(y, yh, zero_division=0),
        MCC=matthews_corrcoef(y, yh),
        ROC_AUC=roc_auc_score(y, p),
        PR_AUC=average_precision_score(y, p),
        TP=tp, FP=fp, TN=tn, FN=fn, N=len(y))


def per_repeat_table(preds, threshold="fixed"):
    rows = []
    for (task, regime, model, rep), g in preds.groupby(["Task", "Regime", "Model", "Repeat"]):
        if threshold == "fixed":
            m = metrics_at(g.y_true.values, g.y_prob.values, 0.5)
        else:                      # fold-specific F1-optimal threshold from inner OOF
            yh = (g.y_prob.values >= g.thr_f1opt.values).astype(int)
            y = g.y_true.values
            tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
            m = dict(Accuracy=accuracy_score(y, yh),
                     Precision=precision_score(y, yh, zero_division=0),
                     Recall=recall_score(y, yh, zero_division=0),
                     Specificity=tn / (tn + fp) if (tn + fp) else np.nan,
                     F1=f1_score(y, yh, zero_division=0),
                     MCC=matthews_corrcoef(y, yh),
                     ROC_AUC=roc_auc_score(y, g.y_prob.values),
                     PR_AUC=average_precision_score(y, g.y_prob.values),
                     TP=tp, FP=fp, TN=tn, FN=fn, N=len(y))
        rows.append(dict(Task=task, Regime=regime, Model=model, Repeat=rep,
                         Threshold=threshold, **m))
    return pd.DataFrame(rows)


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def summarise(per_rep):
    rows = []
    for (task, regime, model, thr), g in per_rep.groupby(["Task", "Regime", "Model", "Threshold"]):
        r = dict(Task=task, Regime=regime, Model=model, Threshold=thr, N_repeats=len(g))
        for m in METRICS:
            lo, hi = ci95(g[m])
            r[f"{m}_mean"] = g[m].mean()
            r[f"{m}_sd"] = g[m].std(ddof=1)
            r[f"{m}_lo95"] = lo
            r[f"{m}_hi95"] = hi
        for c in ["TP", "FP", "TN", "FN"]:
            r[f"{c}_mean"] = g[c].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    preds = pd.read_csv("out/per_fold_predictions.csv.gz")
    per_rep = pd.concat([per_repeat_table(preds, "fixed"),
                         per_repeat_table(preds, "f1opt")], ignore_index=True)
    per_rep.to_csv("out/per_repeat_metrics.csv", index=False)
    summ = summarise(per_rep)
    summ.to_csv("out/summary_metrics.csv", index=False)
    print(summ.shape)


if __name__ == "__main__":
    main()
