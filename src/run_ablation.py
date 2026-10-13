"""
run_ablation.py -- do the 438 descriptors beat trivial ones?

Random Forest only, identity-aware folds (CD-HIT 0.8 clusters), the same
10 x 5 repeated nested cross-validation and the same fold-internal preprocessing
as run_cv.py. The only thing that changes is which descriptors enter the model:

    length_only   PCP_Length
    length+MW     PCP_Length, PCP_MW
    PCP(18)       the physicochemical block
    AAC(20)       amino-acid composition
    DPC(400)      dipeptide composition
    ALL(438)      everything (identical to the main analysis)

If a 438-descriptor model cannot beat a single length feature, the descriptor
blocks are not carrying mechanism information and the manuscript's
interpretability argument does not hold.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.getcwd())
import acp_features as af
import acp_models as am

TASKS = ["apoptosis", "necrosis", "pyroptosis"]
N_REPEATS, N_FOLDS, SEED = 10, 5, am.RANDOM_SEED

SUBSETS = {
    "length_only":   ["PCP_Length"],
    "length+MW":     ["PCP_Length", "PCP_MW"],
    "PCP(18)":       af.PCP_COLS,
    "AAC(20)":       af.AAC_NAMES,
    "DPC(400)":      af.DPC_NAMES,
    "ALL(438)":      af.AAC_NAMES + af.DPC_NAMES + af.PCP_COLS,
}


def run_repeat(task, name, cols_keep, df, X, y, groups, cols):
    sel = [cols.index(c) for c in cols_keep]
    Xs = X[:, sel]
    names = [cols[i] for i in sel]
    rows = []
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 0)
    for rep in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(cv.split(Xs, y, groups)):
            seed = SEED + 100 * rep + fold
            pre = am.make_preprocessor(names).fit(Xs[tr])
            Xtr, Xte = pre.transform(Xs[tr]), pre.transform(Xs[te])
            tuned, _ = am.tune_base("RF", Xtr, y[tr], groups[tr], seed)
            p = clone(tuned).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
            rows.append(pd.DataFrame(dict(Task=task, Features=name, N_desc=len(sel),
                                          Repeat=rep, Fold=fold, y_true=y[te], y_prob=p)))
    return pd.concat(rows, ignore_index=True)


def main():
    os.makedirs("out", exist_ok=True)
    jobs = []
    for task in TASKS:
        df = pd.read_csv(f"data/{task}_clusters.csv")
        X, cols = af.build_features(df.Sequence.tolist(), "ALL")
        y = df.Label.values.astype(int)
        g = df.cluster_08.values
        for name, keep in SUBSETS.items():
            jobs.append(delayed(run_repeat)(task, name, keep, df, X, y, g, cols))
    print(f"dispatching {len(jobs)} jobs", flush=True)
    res = Parallel(n_jobs=8, verbose=5)(jobs)
    out = pd.concat(res, ignore_index=True)
    out.to_csv("out/ablation_predictions.csv.gz", index=False, compression="gzip")
    print("rows:", len(out), flush=True)


if __name__ == "__main__":
    main()
