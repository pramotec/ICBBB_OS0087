"""
run_lengthmatched.py -- does any signal survive when length is removed as a cue?

The descriptor ablation showed that on the necrosis task peptide length alone matches the
full 438-descriptor model, and that necrosis positives are 7 residues longer than their
negatives (median 22 vs 15, Mann-Whitney p = 3e-15). That makes length a confound rather
than a feature. This script removes it by construction.

Design
    1:1 nearest-neighbour length matching without replacement, caliper 2 residues:
    every retained positive is paired with an unused negative of nearly the same length.
    The matched set is balanced (prevalence 0.5) and length-matched (mean |dL| ~ 0.5),
    so MCC and ROC-AUC measure only what sequence composition adds beyond size.

    Evaluation is identical to run_cv.py: Random Forest, StratifiedGroupKFold on the
    0.8-identity clusters, 10 x 5 repeated nested CV, all preprocessing fold-internal.

Feature sets
    length_only   sanity check -- should collapse to chance if matching worked
    AAC+PCP(38)   composition and physicochemistry without the 400-column DPC block
    ALL(438)      the descriptor set used in the manuscript
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
CALIPER = 2
N_REPEATS, N_FOLDS, SEED = 10, 5, am.RANDOM_SEED

SUBSETS = {
    "length_only": ["PCP_Length"],
    "AAC+PCP(38)": af.AAC_NAMES + af.PCP_COLS,
    "ALL(438)": af.AAC_NAMES + af.DPC_NAMES + af.PCP_COLS,
}


def length_matched(df, caliper=CALIPER, seed=SEED):
    """1:1 nearest-neighbour matching on length, without replacement."""
    d = df.copy()
    d["L"] = d.Sequence.astype(str).str.len()
    pos = d[d.Label == 1].sample(frac=1, random_state=seed)
    neg = d[d.Label == 0]
    used, keep = set(), []
    for i, pr in pos.iterrows():
        cand = neg[(~neg.index.isin(used)) & (abs(neg.L - pr.L) <= caliper)]
        if len(cand):
            j = cand.iloc[(cand.L - pr.L).abs().argsort()].index[0]
            used.add(j)
            keep += [i, j]
    return d.loc[keep].reset_index(drop=True)


def run_one(task, name, cols_keep, dfm, X, cols):
    sel = [cols.index(c) for c in cols_keep]
    Xs, names = X[:, sel], [cols[i] for i in sel]
    y = dfm.Label.values.astype(int)
    g = dfm.cluster_08.values
    rows = []
    for rep in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(cv.split(Xs, y, g)):
            seed = SEED + 100 * rep + fold
            pre = am.make_preprocessor(names).fit(Xs[tr])
            Xtr, Xte = pre.transform(Xs[tr]), pre.transform(Xs[te])
            tuned, _ = am.tune_base("RF", Xtr, y[tr], g[tr], seed)
            p = clone(tuned).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
            rows.append(pd.DataFrame(dict(Task=task, Features=name, N_desc=len(sel),
                                          Repeat=rep, Fold=fold, y_true=y[te], y_prob=p)))
    return pd.concat(rows, ignore_index=True)


def main():
    os.makedirs("out", exist_ok=True)
    jobs, meta = [], []
    for task in TASKS:
        df = pd.read_csv(f"data/{task}_clusters.csv")
        dfm = length_matched(df)
        X, cols = af.build_features(dfm.Sequence.tolist(), "ALL")
        a = dfm[dfm.Label == 1].L
        b = dfm[dfm.Label == 0].L
        meta.append(dict(Task=task, n_total=len(dfm), n_pos=int((dfm.Label == 1).sum()),
                         pct_pos_retained=round(100 * (dfm.Label == 1).sum() / (df.Label == 1).sum(), 1),
                         pos_median_len=float(a.median()), neg_median_len=float(b.median()),
                         n_clusters=int(dfm.cluster_08.nunique())))
        dfm.to_csv(f"data/{task}_lengthmatched.csv", index=False)
        for name, keep in SUBSETS.items():
            jobs.append(delayed(run_one)(task, name, keep, dfm, X, cols))
    pd.DataFrame(meta).to_csv("out/lengthmatched_design.csv", index=False)
    print(f"dispatching {len(jobs)} jobs", flush=True)
    res = Parallel(n_jobs=8, verbose=5)(jobs)
    pd.concat(res, ignore_index=True).to_csv("out/lengthmatched_predictions.csv.gz",
                                             index=False, compression="gzip")
    print("done", flush=True)


if __name__ == "__main__":
    main()
