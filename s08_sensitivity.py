"""s08_sensitivity.py - pre-specified sensitivity analyses probing (A) censoring-linked
missing indicators of the day-2/day-3 clinical-course variables, (B) ICU treatment-response
variables, and (C) class-weighting effects on the primary model.

A) no-late-indicators : primary XGBoost, 4 stages, missing-indicators for the
   day-2/day-3 clinical-course variables DISABLED (administrative-missingness /
   censoring-by-prior-death sensitivity).
B) no-treatment       : primary XGBoost, admission & 72h stages, ICU treatment
   variables removed (reverse-causality sensitivity).
C) unweighted         : XGBoost 72h with scale_pos_weight = 1 + sigmoid
   calibration (honest baseline for the calibration narrative).

No other part of the pipeline is retrained; primary results remain the
pre-specified analysis. Outputs: results/tables/sens_*.csv + OOF npz files.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (OOF_DIR, TABLES_DIR, OUTCOMES, STAGES, STAGE_LABELS, SEED,
                    N_SPLITS, N_REPEATS, XGB_PARAMS)
from common import (load_raw, temporal_feature_sets, feature_type_groups,
                    build_preprocessor, repeated_multilabel_cv, auroc, auprc,
                    delong_pvalue, log)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

LATE_COURSE = ["R_AB_2_n", "R_AB_3_n", "NA_R_2_n", "NA_R_3_n",
               "NOT_NA_2_n", "NOT_NA_3_n"]


def treatment_cols(X):
    return [c for c in X.columns if c.endswith("_S_n")] + ["NITR_S"]


def build_preproc_variant(X_train, groups, scale, drop_cols=(), no_indicator_cols=()):
    """ColumnTransformer with per-block indicator control."""
    cols = [c for c in X_train.columns if c not in drop_cols]
    noi = set(no_indicator_cols)
    g_cont = [c for c in groups["continuous"] if c in cols]
    g_bin = [c for c in groups["binary"] if c in cols]
    g_ord = [c for c in groups["ordinal"] if c in cols]
    g_po = [c for c in groups["partially_ordered"] if c in cols]
    # split blocks by indicator policy
    def blocks(g):
        wi = [c for c in g if c not in noi]
        ni = [c for c in g if c in noi]
        return wi, ni
    transformers = []
    for name, g, sc in [("continuous", g_cont, scale), ("binary", g_bin, False),
                        ("ordinal", g_ord, scale)]:
        wi, ni = blocks(g)
        if wi:
            steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
            if sc:
                steps.append(("scaler", StandardScaler()))
            transformers.append((name, Pipeline(steps), wi))
        if ni:
            steps = [("imputer", SimpleImputer(strategy="median", add_indicator=False))]
            if sc:
                steps.append(("scaler", StandardScaler()))
            transformers.append((name + "_noin", Pipeline(steps), ni))
    wi, ni = blocks(g_po)
    for tag, gg, ind in [("po", wi, True), ("po_noin", ni, False)]:
        if gg:
            transformers.append((tag, Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=ind)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), gg))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_variant(tag, stages, drop_cols=(), no_indicator_cols=(), spw_mode="per"):
    X, y, variables, _ = load_raw()
    fs = temporal_feature_sets(X)
    groups = feature_type_groups(variables)
    Y = y.values.astype(np.int8)
    out_rows = []
    for s in stages:
        Xs = X[fs[s]].drop(columns=[c for c in drop_cols if c in X[fs[s]].columns])
        n = Xs.shape[0]
        oof = np.zeros((N_REPEATS, n, len(OUTCOMES)), dtype=np.float32)
        t0 = time.time()
        for r, f, tr, te in repeated_multilabel_cv(Y, N_SPLITS, N_REPEATS, SEED):
            pre = build_preproc_variant(Xs.iloc[tr], groups, scale=False,
                                        no_indicator_cols=no_indicator_cols)
            pre.fit(Xs.iloc[tr])
            Xtr = pre.transform(Xs.iloc[tr]); Xte = pre.transform(Xs.iloc[te])
            for j in range(len(OUTCOMES)):
                pos = int(Y[tr, j].sum()); neg = len(tr) - pos
                spw = (neg / pos) if (pos > 0 and spw_mode == "per") else 1.0
                m = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
                m.fit(Xtr, Y[tr, j])
                oof[r, te, j] = m.predict_proba(Xte)[:, 1]
        np.savez_compressed(OOF_DIR / f"{tag}_{s}.npz", prob=oof)
        P = oof.mean(axis=0)
        for j, o in enumerate(OUTCOMES):
            out_rows.append(dict(variant=tag, stage=s, outcome=o,
                                 AUROC=auroc(Y[:, j], P[:, j]),
                                 AUPRC=auprc(Y[:, j], P[:, j])))
        log(f"  [{tag}|{s}] done {time.time()-t0:.0f}s")
    df = pd.DataFrame(out_rows)
    df.to_csv(TABLES_DIR / f"sens_{tag}.csv", index=False)
    return df


def compare_with_primary(tag, stages):
    """DeLong vs the pre-specified primary OOF (same folds)."""
    X, y, variables, _ = load_raw()
    Y = y.values.astype(int)
    rows = []
    for s in stages:
        Pnew = np.load(OOF_DIR / f"{tag}_{s}.npz")["prob"].mean(axis=0)
        Pref = np.load(OOF_DIR / f"XGB_{s}.npz")["prob"].mean(axis=0)
        for j, o in enumerate(OUTCOMES):
            d, p = delong_pvalue(Y[:, j], Pref[:, j], Pnew[:, j])
            rows.append(dict(variant=tag, stage=s, outcome=o,
                             AUROC_primary=auroc(Y[:, j], Pref[:, j]),
                             AUROC_variant=auroc(Y[:, j], Pnew[:, j]),
                             delta=d, p_DeLong=p))
        mnew = np.nanmean([auroc(Y[:, j], Pnew[:, j]) for j in range(12)])
        mref = np.nanmean([auroc(Y[:, j], Pref[:, j]) for j in range(12)])
        rows.append(dict(variant=tag, stage=s, outcome="MACRO",
                         AUROC_primary=mref, AUROC_variant=mnew,
                         delta=mnew - mref, p_DeLong=np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / f"sens_{tag}_vs_primary.csv", index=False)
    return df


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    X, y, variables, _ = load_raw()
    trt = treatment_cols(X)
    if which == "A":
        log("A) no late-course missing indicators, 4 stages")
        run_variant("noIND", STAGES, no_indicator_cols=LATE_COURSE)
        compare_with_primary("noIND", STAGES)
    elif which == "B":
        log(f"B) no ICU treatment variables ({trt}), admission+72h")
        run_variant("noTRT", ["admission", "72hr"], drop_cols=trt)
        compare_with_primary("noTRT", ["admission", "72hr"])
    elif which == "C":
        log("C) unweighted XGB 72h (+sigmoid calibration)")
        run_variant("unW", ["72hr"], spw_mode="one")
        # calibration of the unweighted model (sigmoid, fold-internal)
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold
        from common import brier, expected_calibration_error
        fs = temporal_feature_sets(X)
        groups = feature_type_groups(variables)
        Y = y.values.astype(np.int8)
        Xs = X[fs["72hr"]]
        n = Xs.shape[0]
        cal_oof = np.zeros((N_REPEATS, n, len(OUTCOMES)), dtype=np.float32)
        for r, f, tr, te in repeated_multilabel_cv(Y, N_SPLITS, N_REPEATS, SEED):
            pre = build_preproc_variant(Xs.iloc[tr], groups, scale=False)
            pre.fit(Xs.iloc[tr])
            Xtr = pre.transform(Xs.iloc[tr]); Xte = pre.transform(Xs.iloc[te])
            inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
            for j in range(len(OUTCOMES)):
                base = XGBClassifier(**XGB_PARAMS, scale_pos_weight=1.0)
                cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=inner)
                cal.fit(Xtr, Y[tr, j])
                cal_oof[r, te, j] = cal.predict_proba(Xte)[:, 1]
        np.savez_compressed(OOF_DIR / "unW_cal_72hr.npz", prob=cal_oof)
        Pu = np.load(OOF_DIR / "unW_72hr.npz")["prob"].mean(axis=0)
        Pc = cal_oof.mean(axis=0)
        rows = []
        for j, o in enumerate(OUTCOMES):
            rows.append(dict(outcome=o,
                             ECE_unweighted_uncal=expected_calibration_error(Y[:, j], Pu[:, j]),
                             ECE_unweighted_cal=expected_calibration_error(Y[:, j], Pc[:, j]),
                             Brier_unweighted_uncal=brier(Y[:, j], Pu[:, j]),
                             Brier_unweighted_cal=brier(Y[:, j], Pc[:, j])))
        pd.DataFrame(rows).to_csv(TABLES_DIR / "sens_unW_calibration.csv", index=False)
        log("C done")
    log(f"SENSITIVITY {which} DONE")


if __name__ == "__main__":
    main()
