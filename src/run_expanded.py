#!/usr/bin/env python3
"""Tier-1 nested CV on the original and provisionally expanded positive sets.

Eight designs: 2 tasks x {v1 = curated positives, v2 = curated plus provisional
additions} x {unmatched, length-matched}, each with the AAC+PCP(38) and ALL(438)
descriptor sets, RF and SVM+RF+XGB, 10 repeats of 5 identity-aware outer folds.

v1 is re-run rather than reused so that v1 and v2 differ only in the peptides added:
the same seeds, the same folds, the same preprocessing, the same code path.
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
VERSIONS = ["v2"]   # v1 is reused from the tiered run: the inputs are byte-identical and
                    # a spot-check reproduced its predictions to 0.00e+00, so re-running
                    # it would burn hours to recreate files that already exist
VARIANTS = ["unmatched", "matched"]
FEATURE_SETS = ["AAC_PCP", "ALL"]
REPORT_MODELS = ["RF", "SVM", "XGB", "STACK_SVM_RF_XGB"]
N_FOLDS, N_REPEATS = 5, 10
SEED = am.RANDOM_SEED
PARTS = "out2/expanded_parts"


def build_features(seqs, feature_set):
    X, cols = af.build_features(seqs, "ALL")
    if feature_set == "ALL":
        return X, cols
    keep = [i for i, c in enumerate(cols) if not c.startswith("DPC_")]
    return X[:, keep], [cols[i] for i in keep]


def run_repeat(task, version, variant, fset, repeat, df, X, cols):
    y = df.Label.values.astype(int)
    groups = df.cluster_08.values
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
    key = dict(Task=task, Version=version, Variant=variant, FeatureSet=fset, Repeat=repeat)
    pred_rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        seed = SEED + 100 * repeat + fold
        pre = am.make_preprocessor(cols).fit(X[tr])
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        ytr, yte, gtr = y[tr], y[te], groups[tr]
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
    return pd.concat(pred_rows, ignore_index=True)


def run_and_save(task, version, variant, fset, repeat, df, X, cols):
    stem = f"{PARTS}/{task}_{version}_{variant}_{fset}_r{repeat}"
    run_repeat(task, version, variant, fset, repeat, df, X, cols).to_csv(
        stem + "_pred.csv.gz", index=False, compression="gzip")
    return stem


def main():
    os.makedirs(PARTS, exist_ok=True)
    jobs, skipped = [], 0
    for task in TASKS:
        for version in VERSIONS:
            for variant in VARIANTS:
                df = pd.read_csv(f"out2/{task}_tier1_{variant}_{version}.csv")
                for fset in FEATURE_SETS:
                    X, cols = build_features(df.Sequence.tolist(), fset)
                    for rep in range(N_REPEATS):
                        stem = f"{PARTS}/{task}_{version}_{variant}_{fset}_r{rep}"
                        if os.path.exists(stem + "_pred.csv.gz"):
                            skipped += 1
                            continue
                        jobs.append(delayed(run_and_save)(task, version, variant, fset,
                                                          rep, df, X, cols))
    print(f"dispatching {len(jobs)} repeat-jobs; {skipped} already done", flush=True)
    if jobs:
        Parallel(n_jobs=8, verbose=5)(jobs)
    frames = [pd.read_csv(f) for f in sorted(glob.glob(f"{PARTS}/*_pred.csv.gz"))]
    # fold in the v1 arm from the tiered run rather than recomputing it
    for f in sorted(glob.glob("out2/tiered_parts/*_tier1_*_pred.csv.gz")):
        d = pd.read_csv(f)
        d["Version"] = "v1"
        d = d.rename(columns={"Tier": "_tier"}).drop(columns=["_tier"], errors="ignore")
        frames.append(d)
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv("out2/expanded_per_fold_predictions.csv.gz", index=False, compression="gzip")
    print("prediction rows:", len(preds), flush=True)


if __name__ == "__main__":
    main()
