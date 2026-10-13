"""
run_cv.py -- 10 x 5 repeated nested cross-validation, random vs identity-aware folds.

Outputs (out/):
    per_fold_predictions.csv.gz   one row per (task, regime, repeat, fold, model, peptide)
    preprocessing_stats.csv       PCA components retained per outer fold
    tuned_hyperparameters.csv     selected hyperparameters per outer fold
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.base import clone

sys.path.insert(0, os.getcwd())
import acp_features as af
import acp_models as am

TASKS = ["apoptosis", "necrosis", "pyroptosis"]
REGIMES = ["random", "cluster08"]
MODELS = ["RF", "SVM", "XGB", "STACK_SVM_RF", "STACK_SVM_RF_XGB"]
N_REPEATS = 10
N_FOLDS = 5
FEATURE_SET = "ALL"
SEED = am.RANDOM_SEED


def load_task(task, data_dir="data"):
    df = pd.read_csv(f"{data_dir}/{task}_clusters.csv")
    X, cols = af.build_features(df.Sequence.tolist(), FEATURE_SET)
    return df, X, cols


def outer_splits(regime, X, y, groups, repeat):
    """Return list of (train_idx, test_idx) for one repeat."""
    if regime == "random":
        cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=1,
                                     random_state=SEED + repeat)
        return list(cv.split(X, y))
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=SEED + repeat)
    return list(cv.split(X, y, groups))


def run_repeat(task, regime, repeat, df, X, y, groups, cols):
    pred_rows, pre_rows, hp_rows = [], [], []
    g = groups if regime == "cluster08" else None
    for fold, (tr, te) in enumerate(outer_splits(regime, X, y, groups, repeat)):
        seed = SEED + 100 * repeat + fold
        pre = am.make_preprocessor(cols).fit(X[tr])
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        ytr, yte = y[tr], y[te]
        gtr = g[tr] if g is not None else None
        pre_rows.append(dict(Task=task, Regime=regime, Repeat=repeat, Fold=fold,
                             n_train=len(tr), n_test=len(te),
                             n_train_pos=int(ytr.sum()), n_train_neg=int((ytr == 0).sum()),
                             n_test_pos=int(yte.sum()), n_test_neg=int((yte == 0).sum()),
                             n_features_in=pre.n_in_, n_after_variance_filter=pre.n_after_vt_,
                             n_dpc_components=pre.n_dpc_components_,
                             dpc_variance_retained=pre.dpc_variance_retained_,
                             n_features_out=pre.n_out_))
        oof, ptest, thr = {}, {}, {}
        for name in ["RF", "SVM", "XGB"]:
            tuned, params = am.tune_base(name, Xtr, ytr, gtr, seed)
            hp_rows.append(dict(Task=task, Regime=regime, Repeat=repeat, Fold=fold,
                                Model=name, **{f"hp_{k}": v for k, v in params.items()}))
            oof[name] = am.oof_probabilities(tuned, Xtr, ytr, gtr, seed)
            ptest[name] = clone(tuned).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
            thr[name] = am.f1_optimal_threshold(ytr, oof[name])
        for stack, members in [("STACK_SVM_RF", ["SVM", "RF"]),
                               ("STACK_SVM_RF_XGB", ["SVM", "RF", "XGB"])]:
            Ztr = np.column_stack([oof[m] for m in members])
            Zte = np.column_stack([ptest[m] for m in members])
            meta = LogisticRegression(max_iter=1000, class_weight="balanced")
            meta.fit(Ztr, ytr)
            ptest[stack] = meta.predict_proba(Zte)[:, 1]
            thr[stack] = am.f1_optimal_threshold(ytr, meta.predict_proba(Ztr)[:, 1])
        for name in MODELS:
            pred_rows.append(pd.DataFrame(dict(
                Task=task, Regime=regime, Repeat=repeat, Fold=fold, Model=name,
                Sequence=df.Sequence.values[te], y_true=yte, y_prob=ptest[name],
                thr_f1opt=thr[name])))
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(pre_rows), pd.DataFrame(hp_rows)


def main():
    os.makedirs("out", exist_ok=True)
    jobs, meta = [], []
    cache = {}
    for task in TASKS:
        df, X, cols = load_task(task)
        cache[task] = (df, X, cols)
    for task in TASKS:
        df, X, cols = cache[task]
        y = df.Label.values.astype(int)
        groups = df.cluster_08.values
        for regime in REGIMES:
            for rep in range(N_REPEATS):
                jobs.append(delayed(run_repeat)(task, regime, rep, df, X, y, groups, cols))
                meta.append((task, regime, rep))
    print(f"dispatching {len(jobs)} repeat-jobs", flush=True)
    res = Parallel(n_jobs=8, verbose=5)(jobs)
    preds = pd.concat([r[0] for r in res], ignore_index=True)
    pres = pd.concat([r[1] for r in res], ignore_index=True)
    hps = pd.concat([r[2] for r in res], ignore_index=True)
    preds.to_csv("out/per_fold_predictions.csv.gz", index=False, compression="gzip")
    pres.to_csv("out/preprocessing_stats.csv", index=False)
    hps.to_csv("out/tuned_hyperparameters.csv", index=False)
    print("rows:", len(preds), "| folds:", len(pres), flush=True)


if __name__ == "__main__":
    main()
