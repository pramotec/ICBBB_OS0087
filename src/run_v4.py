#!/usr/bin/env python3
"""Repeated nested cross-validation on the v4 designs (both tasks).

Version 4 applies the authors' resolution of the resubmission checklist: six
provisional peptides recovered by our own literature search are confirmed as
positives, and nine peptides previously labelled negative are relabelled as
positives of the mechanism their cited assay reports (four on the apoptosis
side, eight on the necrosis side, three of them on both). Relabelling moves a
peptide out of the Tier-1 negative pool as well as into the positive set, so
both tasks change and all twelve designs are run.

Same protocol as run_tiered.py: StratifiedGroupKFold on the 0.8-identity clusters
recomputed over the enlarged union, tuning strictly inside training folds, RF /
SVM / XGB and the SVM+RF+XGB stack, AAC+PCP(38) and ALL(438), 10 repeats for
tier 1 and 5 for tiers 2 and 3. Each repeat is written to its own file so an
interrupted run resumes.
"""
from __future__ import annotations

import glob
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import acp_features as af
import acp_models as am

TASKS = ["apoptosis", "necrosis"]
TIERS = ["tier1", "tier2", "tier3"]
VARIANTS = ["unmatched", "matched"]
FEATURE_SETS = ["AAC_PCP", "ALL"]
REPORT_MODELS = ["RF", "SVM", "XGB", "STACK_SVM_RF_XGB"]
N_FOLDS = 5
REPEATS = {"tier1": 10, "tier2": 5, "tier3": 5}
SEED = am.RANDOM_SEED
PARTS = "out2/v4_parts"


def build_features(seqs, feature_set):
    X, cols = af.build_features(seqs, "ALL")
    if feature_set == "ALL":
        return X, cols
    keep = [i for i, c in enumerate(cols) if not c.startswith("DPC_")]
    return X[:, keep], [cols[i] for i in keep]


def run_repeat(task, tier, variant, fset, repeat, df, X, cols):
    y = df.Label.values.astype(int)
    groups = df.cluster_08.values
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
    key = dict(Task=task, Version="v4", Tier=tier, Variant=variant,
               FeatureSet=fset, Repeat=repeat)
    pred_rows, pre_rows = [], []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        seed = SEED + 100 * repeat + fold
        pre = am.make_preprocessor(cols).fit(X[tr])
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        ytr, yte, gtr = y[tr], y[te], groups[tr]
        pre_rows.append(dict(**key, Fold=fold, n_train=len(tr), n_test=len(te),
                             n_train_pos=int(ytr.sum()), n_test_pos=int(yte.sum()),
                             n_features_in=pre.n_in_, n_features_out=pre.n_out_))
        oof, ptest = {}, {}
        for name in ["RF", "SVM", "XGB"]:
            tuned, _ = am.tune_base(name, Xtr, ytr, gtr, seed)
            oof[name] = am.oof_probabilities(tuned, Xtr, ytr, gtr, seed)
            ptest[name] = clone(tuned).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        members = ["SVM", "RF", "XGB"]
        meta = LogisticRegression(max_iter=1000, class_weight="balanced")
        meta.fit(np.column_stack([oof[m] for m in members]), ytr)
        ptest["STACK_SVM_RF_XGB"] = meta.predict_proba(
            np.column_stack([ptest[m] for m in members]))[:, 1]
        for name in REPORT_MODELS:
            pred_rows.append(pd.DataFrame(dict(**key, Fold=fold, Model=name,
                                               Sequence=df.Sequence.values[te],
                                               y_true=yte, y_prob=ptest[name])))
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(pre_rows)


def run_and_save(task, tier, variant, fset, repeat, df, X, cols):
    stem = f"{PARTS}/{task}_{tier}_{variant}_{fset}_r{repeat}"
    preds, pres = run_repeat(task, tier, variant, fset, repeat, df, X, cols)
    preds.to_csv(stem + "_pred.csv.gz", index=False, compression="gzip")
    pres.to_csv(stem + "_pre.csv", index=False)
    return stem


def main():
    os.makedirs(PARTS, exist_ok=True)
    jobs, skipped = [], 0
    for task in TASKS:
        for tier in TIERS:
            for variant in VARIANTS:
                df = pd.read_csv(f"out2/{task}_{tier}_{variant}_v4.csv")
                for fset in FEATURE_SETS:
                    X, cols = build_features(df.Sequence.tolist(), fset)
                    for rep in range(REPEATS[tier]):
                        stem = f"{PARTS}/{task}_{tier}_{variant}_{fset}_r{rep}"
                        if os.path.exists(stem + "_pred.csv.gz"):
                            skipped += 1
                            continue
                        jobs.append(delayed(run_and_save)(task, tier, variant, fset,
                                                          rep, df, X, cols))
    print(f"dispatching {len(jobs)} repeat-jobs ({len(jobs) * N_FOLDS} outer folds); "
          f"{skipped} already done", flush=True)
    if jobs:
        Parallel(n_jobs=8, verbose=5)(jobs)
    preds = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{PARTS}/*_pred.csv.gz"))],
                      ignore_index=True)
    pres = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{PARTS}/*_pre.csv"))],
                     ignore_index=True)
    preds.to_csv("out2/v4_per_fold_predictions.csv.gz", index=False, compression="gzip")
    pres.to_csv("out2/v4_preprocessing_stats.csv", index=False)
    print("prediction rows:", len(preds), "| folds:", len(pres), flush=True)


if __name__ == "__main__":
    main()
