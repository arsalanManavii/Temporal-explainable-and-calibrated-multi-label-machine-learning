"""Shared utilities: data loading, temporal feature sets, preprocessing,
multi-label stratified CV, metrics, calibration measures, DeLong test and
bootstrap confidence intervals.

Everything here is fold-safe: preprocessing statistics are always fitted on
the training part of a fold only (no test information leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, brier_score_loss)

from config import (DATA_DIR, OUTCOMES, STAGES, TIMEPOINT_EXCLUSION, SEED,
                    N_SPLITS, N_REPEATS)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw():
    """Load X (111 features) and y (12 binary targets incl. DEATH)."""
    X = pd.read_csv(DATA_DIR / "X_features.csv")
    y_raw = pd.read_csv(DATA_DIR / "y_targets.csv")
    variables = pd.read_csv(DATA_DIR / "variables_metadata.csv")

    y = y_raw.drop(columns=["LET_IS"]).copy()
    y["DEATH"] = (y_raw["LET_IS"] > 0).astype("int8")
    assert list(y.columns) == OUTCOMES
    assert X.shape == (1700, 111) and y.shape == (1700, 12)
    return X, y, variables, y_raw


def temporal_feature_sets(X: pd.DataFrame):
    """Return {stage: [feature names available at that time point]}."""
    feature_columns = pd.DataFrame({
        "uci_column": np.arange(2, 113),   # ID is column 1
        "feature": X.columns,
    })
    sets = {}
    for stage in STAGES:
        excluded = TIMEPOINT_EXCLUSION[stage]
        keep = ~feature_columns["uci_column"].isin(excluded)
        sets[stage] = feature_columns.loc[keep, "feature"].tolist()
    # sanity: must reproduce the documented counts 102/105/108/111
    assert [len(sets[s]) for s in STAGES] == [102, 105, 108, 111]
    return sets


# ---------------------------------------------------------------------------
# Feature typing (identical logic to the original analysis)
# ---------------------------------------------------------------------------

def feature_type_groups(variables: pd.DataFrame):
    continuous = variables[variables["type"] == "Continuous"]["name"].tolist()
    continuous += ["AGE", "S_AD_KBRIG", "D_AD_KBRIG", "S_AD_ORIT", "D_AD_ORIT"]
    binary = variables[(variables["type"] == "Binary") &
                       (variables["role"] == "Feature")]["name"].tolist()
    categorical = variables[(variables["type"] == "Categorical") &
                            (variables["role"] == "Feature")]["name"].tolist()
    categorical = [c for c in categorical if c != "ZSN_A"]
    ordinal = ["NA_R_1_n", "NA_R_2_n", "NA_R_3_n", "NOT_NA_2_n", "NOT_NA_3_n"]
    ordinal += categorical
    partially_ordered = ["ZSN_A"]
    return dict(continuous=continuous, binary=binary, ordinal=ordinal,
                partially_ordered=partially_ordered)


def build_preprocessor(X_train: pd.DataFrame, groups: dict, scale: bool,
                       engineered: list | None = None):
    """ColumnTransformer fitted on X_train only.

    scale=True  -> for linear models (StandardScaler on continuous/ordinal)
    scale=False -> for tree models
    engineered  -> names of engineered (continuous-like) columns to add
    """
    engineered = engineered or []
    cols = list(X_train.columns)
    g_cont = [c for c in groups["continuous"] if c in cols] + \
             [c for c in engineered if c in cols and c not in groups["continuous"]]
    g_bin = [c for c in groups["binary"] if c in cols]
    g_ord = [c for c in groups["ordinal"] if c in cols]
    g_po = [c for c in groups["partially_ordered"] if c in cols]
    assert len(g_cont) + len(g_bin) + len(g_ord) + len(g_po) == X_train.shape[1], \
        "feature groups do not cover all columns"

    cont_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    ord_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        cont_steps.append(("scaler", StandardScaler()))
        ord_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(transformers=[
        ("continuous", Pipeline(cont_steps), g_cont),
        ("binary", Pipeline([("imputer", SimpleImputer(strategy="median",
                                                       add_indicator=True))]), g_bin),
        ("ordinal", Pipeline(ord_steps), g_ord),
        ("partially_ordered", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), g_po),
    ], remainder="drop")


# ---------------------------------------------------------------------------
# Multi-label stratified K-fold (iterative stratification, Sechidis et al. 2011)
# ---------------------------------------------------------------------------

def iterative_stratified_folds(Y: np.ndarray, n_splits: int, seed: int = SEED):
    """Yield (train_idx, test_idx) for one stratified split of a multi-label Y.

    Iterative stratification: labels are processed from rarest to most common;
    each positive example goes to the fold that most 'desires' that label.
    Negative-only rows are distributed to keep fold sizes equal.
    """
    rng = np.random.default_rng(seed)
    n, k = Y.shape
    desired = np.full(n_splits, n / n_splits, dtype=float)   # rows wanted per fold
    desired_label = {j: np.full(n_splits, Y[:, j].sum() / n_splits, dtype=float)
                     for j in range(k)}
    fold_of = np.full(n, -1, dtype=int)
    remaining = set(range(n))

    order = np.argsort(Y.sum(axis=0))          # rarest label first
    for j in order:
        pos = [i for i in remaining if Y[i, j] == 1]
        rng.shuffle(pos)
        for i in pos:
            f = int(np.argmax(desired_label[j]))
            if desired[f] <= 0:
                f = int(np.argmax(desired))
            fold_of[i] = f
            remaining.discard(i)
            desired[f] -= 1
            for jj in range(k):
                if Y[i, jj] == 1:
                    desired_label[jj][f] -= 1
    rest = sorted(remaining)
    rng.shuffle(rest)
    for i in rest:
        f = int(np.argmax(desired))
        fold_of[i] = f
        desired[f] -= 1
        remaining.discard(i)

    for f in range(n_splits):
        test_idx = np.where(fold_of == f)[0]
        train_idx = np.where(fold_of != f)[0]
        yield train_idx, test_idx


def repeated_multilabel_cv(Y: np.ndarray, n_splits: int = N_SPLITS,
                           n_repeats: int = N_REPEATS, seed: int = SEED):
    """Yield (repeat, fold, train_idx, test_idx) over repeated stratified CV."""
    for r in range(n_repeats):
        for f, (tr, te) in enumerate(
                iterative_stratified_folds(Y, n_splits, seed=seed + 100 * r)):
            yield r, f, tr, te


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan


def auprc(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except ValueError:
        return np.nan


def confusion_at(y_true, y_prob, thr):
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return dict(F1=f1, Sensitivity=rec, Specificity=spec, PPV=prec, NPV=npv,
                TP=tp, FP=fp, FN=fn, TN=tn)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if b == n_bins - 1 \
            else (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def integrated_calibration_index(y_true, y_prob):
    """ICI: mean |observed - predicted| using a loess-style smoother
    (implemented with logistic regression on a natural spline basis)."""
    from sklearn.linear_model import LogisticRegressionCV
    y_prob = np.clip(np.asarray(y_prob, float), 1e-6, 1 - 1e-6)
    lp = np.log(y_prob / (1 - y_prob))
    knots = np.quantile(lp, [0.05, 0.275, 0.5, 0.725, 0.95])
    B = np.column_stack([lp] + [np.maximum(lp - kn, 0) ** 3 for kn in knots] +
                        [lp ** 2, lp ** 3])
    try:
        lr = LogisticRegressionCV(Cs=10, cv=5, max_iter=2000, random_state=SEED)
        lr.fit(B, np.asarray(y_true, int))
        obs = lr.predict_proba(B)[:, 1]
    except Exception:
        obs = y_prob
    return float(np.mean(np.abs(obs - y_prob)))


def brier(y_true, y_prob):
    return brier_score_loss(np.asarray(y_true, int), np.asarray(y_prob, float))


# ---------------------------------------------------------------------------
# Bootstrap CIs (percentile) - patient-level resampling
# ---------------------------------------------------------------------------

def bootstrap_ci(values_fn, n: int, n_boot: int = 2000, seed: int = SEED,
                 alpha: float = 0.05):
    """values_fn(idx_array) -> scalar statistic on the resampled patients."""
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = values_fn(idx)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), stats


def paired_bootstrap_delta(y_true, score_a, score_b, metric=auroc,
                           n_boot: int = 2000, seed: int = SEED):
    """Paired bootstrap of metric(B) - metric(A) on the same patients.
    Returns (delta_obs, ci_lo, ci_hi, p_two_sided)."""
    y_true = np.asarray(y_true); n = len(y_true)
    delta_obs = metric(y_true, score_b) - metric(y_true, score_a)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            deltas[b] = np.nan
            continue
        deltas[b] = metric(y_true[idx], score_b[idx]) - metric(y_true[idx], score_a[idx])
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p = float(min(max(p, 1.0 / n_boot), 1.0))
    return float(delta_obs), float(lo), float(hi), p


# ---------------------------------------------------------------------------
# DeLong test for paired AUROC comparison (DeLong 1988; fast implementation)
# ---------------------------------------------------------------------------

def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(pos[r, :])
        ty[r, :] = _midrank(neg[r, :])
        tz[r, :] = _midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if np.ndim(sx) == 0:
        sx = np.array([[sx]]); sy = np.array([[sy]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_pvalue(y_true, score_a, score_b):
    """Two-sided p-value for H0: AUC(a) == AUC(b) on paired samples."""
    from scipy import stats as sps
    y_true = np.asarray(y_true, int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    scores = np.vstack((np.asarray(score_a, float), np.asarray(score_b, float)))[:, order]
    aucs, cov = _fast_delong(scores, label_1_count)
    l = np.array([[1, -1]])
    var = float((l @ cov @ l.T)[0, 0])
    # delta = AUC(b) - AUC(a)  (consistent with paired_bootstrap_delta)
    if var <= 0:
        return float(aucs[1] - aucs[0]), 1.0
    z = abs(float((l @ aucs)[0])) / np.sqrt(var)
    return float(aucs[1] - aucs[0]), float(2 * sps.norm.sf(z))


# ---------------------------------------------------------------------------
# Threshold selection (fitted on TRAIN predictions only - fold-safe)
# ---------------------------------------------------------------------------

def threshold_max_f1(y_true, y_prob):
    """Threshold maximising F1 on the given (training) data."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    # roc_curve thresholds: descending; compute F1 for each
    best_thr, best_f1 = 0.5, -1
    pp = np.asarray(y_prob)
    for t in np.unique(np.quantile(pp, np.linspace(0.01, 0.99, 99))):
        m = classification_metrics(y_true, (pp >= t).astype(int))
        if m["F1"] > best_f1:
            best_f1, best_thr = m["F1"], float(t)
    return best_thr


def threshold_clinical(y_true, y_prob, min_recall: float = 0.90):
    """Lowest false-positive threshold achieving >= min_recall (high-risk
    outcomes). Falls back to F1-max threshold if unreachable."""
    pp = np.asarray(y_prob, float)
    pos = pp[np.asarray(y_true) == 1]
    if len(pos) == 0:
        return threshold_max_f1(y_true, y_prob)
    # threshold at the (1-min_recall) quantile of positive scores
    thr = float(np.quantile(pos, 1 - min_recall, method="lower"))
    rec = (pos >= thr).mean()
    if rec >= min_recall:
        return thr
    return threshold_max_f1(y_true, y_prob)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - h) / d, (c + h) / d)


def log(*args, tee=True):
    import datetime
    msg = " ".join(str(a) for a in args)
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    if tee:
        print(line, flush=True)
