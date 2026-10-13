"""
run_holdout_variability.py -- how much of the manuscript's hold-out vs CV gap is
sampling noise?

Replicates the ORIGINAL protocol of app.R exactly (fixed default
hyperparameters, one stratified 80/20 split, 5-fold CV inside the training
part, decision threshold 0.5) and repeats it over N_SEEDS random seeds. The
seed used in the manuscript (42) is marked so the published point estimate can
be located inside the resulting distribution.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.getcwd())
import acp_features as af

TASKS = ["apoptosis", "necrosis", "pyroptosis"]
N_SEEDS = 100
MANUSCRIPT_SEED = 42
MODELS = ["RF", "SVM", "XGB", "STACK_SVM_RF", "STACK_SVM_RF_XGB"]


def app_r_defaults(seed):
    """DEFAULT_HP of app.R: rf_trees=500, xgb_rounds=300/depth=6/eta=0.1, svm cost=1."""
    return {
        "RF": RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed),
        "XGB": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             tree_method="hist", random_state=seed, n_jobs=1, verbosity=0),
    }


def one_seed(task, X, y, seed):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                              random_state=seed, shuffle=True)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr))
    oof, ptest = {}, {}
    for name, est in app_r_defaults(seed).items():
        o = np.zeros(len(ytr))
        for itr, ite in folds:
            from sklearn.base import clone
            o[ite] = clone(est).fit(Xtr[itr], ytr[itr]).predict_proba(Xtr[ite])[:, 1]
        oof[name] = o
        ptest[name] = est.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    for stack, members in [("STACK_SVM_RF", ["SVM", "RF"]),
                           ("STACK_SVM_RF_XGB", ["SVM", "RF", "XGB"])]:
        Ztr = np.column_stack([oof[m] for m in members])
        Zte = np.column_stack([ptest[m] for m in members])
        mo = np.zeros(len(ytr))
        for itr, ite in folds:
            mo[ite] = LogisticRegression(max_iter=1000).fit(Ztr[itr], ytr[itr]).predict_proba(Ztr[ite])[:, 1]
        oof[stack] = mo
        ptest[stack] = LogisticRegression(max_iter=1000).fit(Ztr, ytr).predict_proba(Zte)[:, 1]
    rows = []
    for name in MODELS:
        rows.append(dict(Task=task, Seed=seed, Model=name,
                         F1_cv=f1_score(ytr, (oof[name] >= 0.5).astype(int)),
                         F1_holdout=f1_score(yte, (ptest[name] >= 0.5).astype(int))))
    return pd.DataFrame(rows)


def main():
    os.makedirs("out", exist_ok=True)
    jobs = []
    for task in TASKS:
        df = pd.read_csv(f"data/{task}_clusters.csv")
        X, _ = af.build_features(df.Sequence.tolist(), "ALL")
        y = df.Label.values.astype(int)
        for s in range(N_SEEDS):
            jobs.append(delayed(one_seed)(task, X, y, s))
    print(f"dispatching {len(jobs)} seed-jobs", flush=True)
    res = Parallel(n_jobs=8, verbose=5)(jobs)
    out = pd.concat(res, ignore_index=True)
    out["is_manuscript_seed"] = out.Seed == MANUSCRIPT_SEED
    out.to_csv("out/holdout_seed_variability.csv", index=False)
    print("rows:", len(out), flush=True)


if __name__ == "__main__":
    main()
