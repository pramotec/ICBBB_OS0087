"""
acp_models.py -- leakage-aware, nested, repeated cross-validation for the
ACP death-mode classifiers.

Every preprocessing and model-selection decision that the original manuscript
left unspecified is made explicit here and is fitted STRICTLY inside training
folds:

  preprocessing (ACPPreprocessor)
      1. VarianceThreshold(0)        drop constant descriptors
      2. StandardScaler              zero mean / unit variance
      3. PCA on the DPC block only, n_components chosen to retain 95% of the
         DPC variance; the AAC and PCP blocks are passed through unreduced.
         This is the dimensionality-reduction step the manuscript describes
         but the released Shiny code never performed.

  model selection
      inner k-fold grid / randomised search over the grids stated in the
      manuscript; the outer test fold is never seen by the search.

  stacking (Wolpert 1992), made explicit
      within each outer training fold:
        a. tune each base learner by inner CV
        b. regenerate out-of-fold probabilities for the tuned learner with a
           fresh inner 5-fold split  -> meta-features Z_train
        c. fit logistic-regression meta-learner on Z_train
        d. refit base learners on the whole outer training fold, predict the
           outer test fold -> Z_test, apply the meta-learner
      No outer-test information enters a, b, c or d.

  decision threshold
      metrics are reported both at the fixed 0.5 cut-off and at an F1-optimal
      threshold selected on the inner out-of-fold probabilities only.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_SEED = 42
PCA_VARIANCE = 0.95


class DPCBlockPCA(BaseEstimator, TransformerMixin):
    """PCA on the DPC columns only; AAC/PCP columns pass through untouched."""

    def __init__(self, dpc_mask=None, n_components=PCA_VARIANCE, random_state=RANDOM_SEED):
        self.dpc_mask = dpc_mask
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        m = np.asarray(self.dpc_mask, dtype=bool)
        self.mask_ = m
        if m.sum() == 0:
            self.pca_ = None
            return self
        ncomp = self.n_components
        if isinstance(ncomp, float):
            ncomp = min(ncomp, 0.999999)
        self.pca_ = PCA(n_components=ncomp, svd_solver="full",
                        random_state=self.random_state).fit(X[:, m])
        self.n_components_ = int(self.pca_.n_components_)
        self.variance_retained_ = float(self.pca_.explained_variance_ratio_.sum())
        return self

    def transform(self, X):
        if self.pca_ is None:
            return X
        return np.hstack([X[:, ~self.mask_], self.pca_.transform(X[:, self.mask_])])


def make_preprocessor(feature_names):
    """Pipeline: variance filter -> scaler -> DPC-block PCA (fold-internal)."""
    is_dpc = np.array([n.startswith("DPC_") for n in feature_names])

    class _Pre(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.vt_ = VarianceThreshold(0.0).fit(X)
            keep = self.vt_.get_support()
            Xv = self.vt_.transform(X)
            self.sc_ = StandardScaler().fit(Xv)
            self.pca_ = DPCBlockPCA(dpc_mask=is_dpc[keep]).fit(self.sc_.transform(Xv))
            self.n_in_ = X.shape[1]
            self.n_after_vt_ = int(keep.sum())
            self.n_dpc_components_ = getattr(self.pca_, "n_components_", 0)
            self.dpc_variance_retained_ = getattr(self.pca_, "variance_retained_", np.nan)
            self.n_out_ = self.transform(X[:1]).shape[1]
            return self

        def transform(self, X):
            return self.pca_.transform(self.sc_.transform(self.vt_.transform(X)))

    return _Pre()


# ---------------------------------------------------------------- base models
def base_estimators(seed=RANDOM_SEED, n_jobs=1):
    return {
        "RF": RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=seed, n_jobs=n_jobs),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
                   random_state=seed),
        "XGB": XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", tree_method="hist",
                             random_state=seed, n_jobs=n_jobs, verbosity=0),
    }


# Grids follow the manuscript's stated search space.
PARAM_GRIDS = {
    "RF": {"max_features": ["sqrt", 0.1, 0.3], "min_samples_leaf": [1, 3]},
    "SVM": {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1]},
    "XGB": {"n_estimators": [50, 100, 200], "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.3]},
}
XGB_N_ITER = 8
# Forests are grown with 200 trees during the hyperparameter search and with the
# manuscript's 500 trees for the out-of-fold and test-set predictions; the
# ranking of candidate settings is stable well below 500 trees.
RF_TREES_SEARCH = 200
RF_TREES_FINAL = 500


def _inner_cv(groups, n_splits, seed):
    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _split(cv, X, y, groups):
    return cv.split(X, y, groups) if groups is not None else cv.split(X, y)


def tune_base(name, Xtr, ytr, gtr, seed, inner_splits=3, n_jobs=1):
    """Inner-CV hyperparameter search. Returns a fitted-unspecified tuned estimator."""
    est = base_estimators(seed)[name]
    search_est = clone(est)
    if name == "RF":
        search_est.set_params(n_estimators=RF_TREES_SEARCH)
    cv = _inner_cv(gtr, inner_splits, seed)
    splits = list(_split(cv, Xtr, ytr, gtr))
    if name == "XGB":
        srch = RandomizedSearchCV(search_est, PARAM_GRIDS[name], n_iter=XGB_N_ITER,
                                  scoring="roc_auc", cv=splits, random_state=seed,
                                  n_jobs=n_jobs, refit=False)
    else:
        srch = GridSearchCV(search_est, PARAM_GRIDS[name], scoring="roc_auc", cv=splits,
                            n_jobs=n_jobs, refit=False)
    srch.fit(Xtr, ytr)
    return clone(est).set_params(**srch.best_params_), srch.best_params_


def oof_probabilities(est, Xtr, ytr, gtr, seed, n_splits=5):
    """Out-of-fold probabilities used as stacking meta-features."""
    oof = np.full(len(ytr), np.nan)
    cv = _inner_cv(gtr, n_splits, seed + 1000)
    for itr, ite in _split(cv, Xtr, ytr, gtr):
        e = clone(est).fit(Xtr[itr], ytr[itr])
        oof[ite] = e.predict_proba(Xtr[ite])[:, 1]
    if np.isnan(oof).any():                       # group CV can leave gaps
        med = np.nanmedian(oof)
        oof = np.where(np.isnan(oof), med, oof)
    return oof


def f1_optimal_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    if len(thr) == 0:
        return 0.5
    return float(thr[max(0, int(np.argmax(f1[:-1])))])
