"""s07_utility.py - clinical-utility and supporting analyses.

Sections:
  A. best-calibrated OOF matrices (per-outcome method chosen by OOF Brier)
  B. Decision Curve Analysis (calibrated 72h probabilities)
  C. Risk-group stratification (quintiles of calibrated risk)
  D. RQ3: clinical-course engineered features ADDED to raw features
     (full repeated CV, XGB, 72h stage) + comparison tests + SHAP ranks
  E. Table 1: baseline cohort characteristics (overall / survivors /
     non-survivors, with p-values)
  F. Events-per-variable table
  G. Imputation-strategy probe (LR probe model, 5-fold CV, 72h)
  H. Literature comparison table (curated constants from published papers)
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (OOF_DIR, TABLES_DIR, OUTCOMES, OUTCOME_NAMES, STAGES,
                    STAGE_LABELS, SEED, N_SPLITS, N_REPEATS, XGB_PARAMS)
from common import (load_raw, temporal_feature_sets, feature_type_groups,
                    build_preprocessor, iterative_stratified_folds,
                    repeated_multilabel_cv, auroc, auprc, brier, delong_pvalue,
                    paired_bootstrap_delta, wilson_ci, classification_metrics,
                    log)
from scipy import stats

PRIMARY = "XGB"


# ===========================================================================
# A. best-calibrated OOF
# ===========================================================================
def build_cal_best():
    out = {}
    for s in ["admission", "72hr"]:
        base = np.load(OOF_DIR / f"{PRIMARY}_{s}.npz")["prob"].mean(axis=0)
        sig = np.load(OOF_DIR / f"CAL_sigmoid_{s}.npz")["prob"].mean(axis=0)
        iso_f = OOF_DIR / f"CAL_isotonic_{s}.npz"
        iso = np.load(iso_f)["prob"].mean(axis=0) if iso_f.exists() else None
        X, y, _, _ = load_raw()
        Y = y.values.astype(int)
        loro = pd.read_csv(TABLES_DIR / "calibration_loro.csv").set_index("outcome")
        best = base.copy()
        choice = {}
        for j, o in enumerate(OUTCOMES):
            bm = str(loro.loc[o, "method"])   # pre-specified LORO selection
            best[:, j] = {"sigmoid": sig[:, j], "isotonic": iso[:, j]}[bm]
            choice[o] = bm
        np.save(OOF_DIR / f"cal_best_{s}.npy", best.astype(np.float32))
        out[s] = (best, choice)
        log(f"cal_best {s}: " + json.dumps(choice))
    return out


# ===========================================================================
# B. Decision Curve Analysis
# ===========================================================================
def run_dca(Y, cal72, caladm):
    thrs = np.round(np.arange(0.005, 0.605, 0.005), 4)
    rows = []
    n = Y.shape[0]
    for j, o in enumerate(OUTCOMES):
        prev = Y[:, j].mean()
        for t in thrs:
            w = t / (1 - t)
            for tag, P in [("calibrated_72h", cal72[:, j]),
                           ("calibrated_admission", caladm[:, j])]:
                tp = ((P >= t) & (Y[:, j] == 1)).sum()
                fp = ((P >= t) & (Y[:, j] == 0)).sum()
                nb = (tp - fp * w) / n
                rows.append(dict(outcome=o, model=tag, threshold=t,
                                 net_benefit=float(nb), n_flagged=int(tp + fp)))
            tp_all = Y[:, j].sum(); fp_all = n - tp_all
            rows.append(dict(outcome=o, model="treat_all", threshold=t,
                             net_benefit=float((tp_all - fp_all * (t / (1 - t))) / n),
                             n_flagged=n))
            rows.append(dict(outcome=o, model="treat_none", threshold=t,
                             net_benefit=0.0, n_flagged=0))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "dca_curves.csv", index=False)

    # summary: net benefit at clinically relevant thresholds
    summary = []
    key_thr = {"DEATH": [0.05, 0.10, 0.15, 0.20, 0.30],
               "ZSN": [0.10, 0.20, 0.30, 0.40],
               "OTEK_LANC": [0.05, 0.10, 0.15, 0.20],
               "REC_IM": [0.05, 0.10, 0.15, 0.20],
               "RAZRIV": [0.02, 0.05, 0.10],
               "FIBR_PREDS": [0.05, 0.10, 0.15, 0.20]}
    for o, ths in key_thr.items():
        for t in ths:
            sub = df[(df.outcome == o) & (np.isclose(df.threshold, t))]
            r = dict(outcome=o, threshold=t)
            for mname in ["calibrated_72h", "calibrated_admission", "treat_all"]:
                v = sub[sub.model == mname]["net_benefit"]
                r[mname] = float(v.iloc[0]) if len(v) else np.nan
            summary.append(r)
    pd.DataFrame(summary).to_csv(TABLES_DIR / "dca_summary.csv", index=False)
    log("DCA saved")


# ===========================================================================
# C. Risk-group stratification
# ===========================================================================
def run_risk_groups(Y, cal72):
    rows = []
    for o in ["DEATH", "ZSN", "OTEK_LANC", "REC_IM", "FIBR_JELUD", "RAZRIV"]:
        j = OUTCOMES.index(o)
        p = cal72[:, j]
        q = pd.qcut(p, 5, labels=False, duplicates="drop")
        for g in sorted(np.unique(q)):
            m = q == g
            k = int(Y[m, j].sum()); nn = int(m.sum())
            lo, hi = wilson_ci(k, nn)
            rows.append(dict(outcome=o, quintile=int(g) + 1, n=nn,
                             events=k, observed_rate=k / nn,
                             obs_lo=lo, obs_hi=hi,
                             mean_predicted=float(p[m].mean()),
                             pred_range=f"{p[m].min():.3f}-{p[m].max():.3f}"))
    pd.DataFrame(rows).to_csv(TABLES_DIR / "risk_groups.csv", index=False)
    log("risk groups saved")


# ===========================================================================
# D. RQ3 - engineered clinical-course features (added, not replacing)
# ===========================================================================
def build_clinical_course_features(X_input: pd.DataFrame) -> pd.DataFrame:
    X_new = X_input.copy()
    specs = {
        "pain_relapse": ["R_AB_1_n", "R_AB_2_n", "R_AB_3_n"],
        "opioid_use": ["NA_R_1_n", "NA_R_2_n", "NA_R_3_n"],
        "nsaid_use": ["NOT_NA_1_n", "NOT_NA_2_n", "NOT_NA_3_n"],
    }
    for stem, cols in specs.items():
        X_new[f"{stem}_total"] = X_input[cols].sum(axis=1, min_count=1)
        active = X_input[cols].gt(0).sum(axis=1).astype(float)
        observed = X_input[cols].notna().sum(axis=1)
        active[observed == 0] = np.nan
        X_new[f"{stem}_active_days"] = active
        X_new[f"{stem}_max"] = X_input[cols].max(axis=1)
    late = ["R_AB_2_n", "R_AB_3_n"]
    pl = X_input[late].gt(0).any(axis=1).astype(float)
    pl[X_input[late].notna().sum(axis=1) == 0] = np.nan
    X_new["pain_relapse_late"] = pl
    return X_new


ENGINEERED = ["pain_relapse_total", "pain_relapse_active_days",
              "pain_relapse_max", "pain_relapse_late", "opioid_use_total",
              "opioid_use_active_days", "opioid_use_max", "nsaid_use_total",
              "nsaid_use_active_days", "nsaid_use_max"]


def run_rq3(X, Y, groups):
    from xgboost import XGBClassifier
    fs = temporal_feature_sets(X)
    X72e = build_clinical_course_features(X[fs["72hr"]])
    g2 = {k: list(v) for k, v in groups.items()}
    g2["continuous"] = g2["continuous"] + [c for c in ENGINEERED
                                            if c != "pain_relapse_late"]
    g2["binary"] = g2["binary"] + ["pain_relapse_late"]

    n = X72e.shape[0]
    oof = np.zeros((N_REPEATS, n, len(OUTCOMES)), dtype=np.float32)
    t0 = time.time()
    for r, f, tr, te in repeated_multilabel_cv(Y, N_SPLITS, N_REPEATS, SEED):
        pre = build_preprocessor(X72e.iloc[tr], g2, scale=False)
        pre.fit(X72e.iloc[tr])
        Xtr = pre.transform(X72e.iloc[tr]); Xte = pre.transform(X72e.iloc[te])
        for j in range(len(OUTCOMES)):
            pos = int(Y[tr, j].sum()); spw = (len(tr) - pos) / pos if pos else 1.0
            m = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
            m.fit(Xtr, Y[tr, j])
            oof[r, te, j] = m.predict_proba(Xte)[:, 1]
        if f == N_SPLITS - 1:
            log(f"  [RQ3] repeat {r} done ({time.time()-t0:.0f}s)")
    np.savez_compressed(OOF_DIR / "XGBRQ3_72hr.npz", prob=oof)

    # engineered-feature SHAP importance (fixed split, mean over outcomes)
    from sklearn.model_selection import train_test_split
    import shap
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    pre = build_preprocessor(X72e.iloc[tr], g2, scale=False)
    pre.fit(X72e.iloc[tr])
    Xtr = pre.transform(X72e.iloc[tr]); Xte = pre.transform(X72e.iloc[te])
    names = pre.get_feature_names_out()
    eng_rows = []
    for j, o in enumerate(OUTCOMES):
        pos = int(Y[tr, j].sum()); spw = (len(tr) - pos) / pos if pos else 1.0
        m = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        m.fit(Xtr, Y[tr, j])
        sv = shap.TreeExplainer(m).shap_values(Xte)
        mabs = np.abs(sv).mean(axis=0)
        order = np.argsort(mabs)[::-1]
        rank_of = {names[ix]: rk + 1 for rk, ix in enumerate(order)}
        for e in ENGINEERED:
            hit = [(nm, rk) for nm, rk in rank_of.items()
                   if e in nm and "missingindicator" not in nm]
            for nm, rk in hit:
                eng_rows.append(dict(outcome=o, feature=nm, rank=rk,
                                     mean_abs_shap=float(mabs[order[rk - 1]])))
    pd.DataFrame(eng_rows).to_csv(TABLES_DIR / "rq3_engineered_shap.csv",
                                  index=False)

    # comparison vs base XGB 72h
    base = np.load(OOF_DIR / "XGB_72hr.npz")["prob"].mean(axis=0)
    new = oof.mean(axis=0)
    rows = []
    for j, o in enumerate(OUTCOMES):
        d_auc, p_del = delong_pvalue(Y[:, j], base[:, j], new[:, j])
        d_ap, lo, hi, p_bt = paired_bootstrap_delta(
            Y[:, j], base[:, j], new[:, j], metric=auprc, n_boot=1000)
        rows.append(dict(outcome=o,
                         AUROC_base=auroc(Y[:, j], base[:, j]),
                         AUROC_rq3=auroc(Y[:, j], new[:, j]),
                         delta_AUROC=d_auc, p_DeLong=p_del,
                         AUPRC_base=auprc(Y[:, j], base[:, j]),
                         AUPRC_rq3=auprc(Y[:, j], new[:, j]),
                         delta_AUPRC=d_ap, dAUPRC_lo=lo, dAUPRC_hi=hi,
                         p_boot_AUPRC=p_bt))
    df = pd.DataFrame(rows)
    df["macro"] = False
    mac = dict(outcome="MACRO",
               AUROC_base=float(np.nanmean(df.AUROC_base)),
               AUROC_rq3=float(np.nanmean(df.AUROC_rq3)),
               delta_AUROC=float(np.nanmean(df.delta_AUROC)),
               AUPRC_base=float(np.nanmean(df.AUPRC_base)),
               AUPRC_rq3=float(np.nanmean(df.AUPRC_rq3)),
               delta_AUPRC=float(np.nanmean(df.delta_AUPRC)),
               macro=True)
    df = pd.concat([df, pd.DataFrame([mac])], ignore_index=True)
    df.to_csv(TABLES_DIR / "rq3_comparison.csv", index=False)
    log("RQ3 saved")


# ===========================================================================
# E. Table 1 - baseline characteristics
# ===========================================================================
T1_VARS = [
    # (column, label, kind)  kind: cont / male / binary01 / ordinal
    ("AGE", "Age, years", "cont"),
    ("SEX", "Male sex", "male"),
    ("INF_ANAM", "Prior myocardial infarction (count)", "ordinal"),
    ("STENOK_AN", "Exertional angina in anamnesis (grade)", "ordinal"),
    ("FK_STENOK", "Angina functional class", "ordinal"),
    ("IBS_POST", "CHD in recent weeks", "ordinal"),
    ("GB", "Essential hypertension (any stage)", "binarygt0"),
    ("SIM_GIPERT", "Symptomatic hypertension", "binary01"),
    ("DLIT_AG", "Duration of angina", "ordinal"),
    ("ZSN_A", "Baseline heart-failure class (ZSN_A)", "ordinal"),
    ("endocr_01", "Diabetes mellitus in anamnesis", "binary01"),
    ("endocr_02", "Obesity in anamnesis", "binary01"),
    ("zab_leg_01", "Chronic lung disease", "binary01"),
    ("ant_im", "Anterior MI localisation (grade)", "ordinal"),
    ("lat_im", "Lateral MI localisation (grade)", "ordinal"),
    ("inf_im", "Inferior MI localisation (grade)", "ordinal"),
    ("post_im", "Posterior MI localisation (grade)", "ordinal"),
    ("TIME_B_S", "Time from onset to admission (code 1-9)", "ordinal"),
    ("S_AD_ORIT", "Systolic BP at admission, mmHg", "cont"),
    ("D_AD_ORIT", "Diastolic BP at admission, mmHg", "cont"),
    ("ritm_ecg_p_01", "Normal sinus rhythm at admission (HR 60-90)", "binary01"),
    ("K_BLOOD", "Serum potassium, mmol/L", "cont"),
    ("NA_BLOOD", "Serum sodium, mmol/L", "cont"),
    ("L_BLOOD", "Leukocyte count, 10^9/L", "cont"),
    ("ROE", "ESR, mm/h", "cont"),
    ("AST_BLOOD", "AST, U/L", "cont"),
    ("ALT_BLOOD", "ALT, U/L", "cont"),
    ("LID_S_n", "Lidocaine in ICU (days; in-hospital treatment)", "ordinal"),
    ("GEPAR_S_n", "Heparin in ICU (days; in-hospital treatment)", "ordinal"),
    ("NA_R_1_n", "Opioid analgesics day 1 (doses)", "ordinal"),
    ("R_AB_1_n", "Anginal attacks day 1 (count)", "ordinal"),
]


def fmt_cont(v):
    v = v.dropna()
    return f"{v.mean():.1f} ± {v.std():.1f}"


def fmt_median(v):
    v = v.dropna()
    return f"{v.median():.1f} [{v.quantile(.25):.1f}–{v.quantile(.75):.1f}]"


def fmt_binary(v, code=1):
    v = pd.Series(v) if not isinstance(v, pd.Series) else v
    n = int((v == code).sum()); tot = int(v.notna().sum())
    return f"{n} ({n/tot*100:.1f}%)" if tot else "0"


def fmt_ordinal(v):
    v = v.dropna()
    return f"{v.median():.0f} [{v.quantile(.25):.0f}–{v.quantile(.75):.0f}]"


def run_table1(X, y_raw):
    death = (y_raw["LET_IS"] > 0).values
    rows = []
    for col, label, kind in T1_VARS:
        v = X[col]
        g0, g1 = v[~death], v[death]
        if kind == "cont":
            overall, surv, nons = fmt_median(v), fmt_median(g0), fmt_median(g1)
            a, b = v[~death].dropna(), v[death].dropna()
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        elif kind == "male":
            overall, surv, nons = fmt_binary(v, 1), fmt_binary(g0, 1), fmt_binary(g1, 1)
            tab = pd.crosstab(death, v.fillna(-1))
            p = stats.chi2_contingency(tab).pvalue if tab.shape == (2, 2) else np.nan
        elif kind == "binary01":
            overall, surv, nons = fmt_binary(v, 1), fmt_binary(g0, 1), fmt_binary(g1, 1)
            tab = pd.crosstab(death, (v.fillna(-1) == 1))
            try:
                p = stats.chi2_contingency(tab).pvalue
            except Exception:
                p = np.nan
        elif kind == "binarygt0":
            overall, surv, nons = fmt_binary(v.fillna(0) > 0, True), fmt_binary(g0.fillna(0) > 0, True), fmt_binary(g1.fillna(0) > 0, True)
            tab = pd.crosstab(death, v.fillna(-1) > 0)
            try:
                p = stats.chi2_contingency(tab).pvalue
            except Exception:
                p = np.nan
        else:  # ordinal
            overall, surv, nons = fmt_ordinal(v), fmt_ordinal(g0), fmt_ordinal(g1)
            a, b = v[~death].dropna(), v[death].dropna()
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(b) else np.nan
        rows.append(dict(variable=label, column=col,
                         overall=overall, survivors=surv,
                         non_survivors=nons,
                         missing_pct=round(v.isna().mean() * 100, 1),
                         p_value=("%.3g" % p) if p == p else "NA"))
    hdr = dict(variable="N", column="", overall=str(len(X)),
               survivors=str((~death).sum()), non_survivors=str(death.sum()),
               missing_pct=0.0, p_value="")
    foot = dict(variable="TIME_B_S scale: code 1 <2 h, 2 = 2-4 h, 3 = 4-6 h, 4 = 6-8 h, 5 = 8-12 h, 6 = 12-24 h, 7 >1 d, 8 >2 d, 9 >3 d. ICU treatment variables are in-hospital exposures, not baseline characteristics.",
                column="", overall="", survivors="", non_survivors="",
                missing_pct="", p_value="")
    df = pd.DataFrame([hdr] + rows + [foot])
    df.to_csv(TABLES_DIR / "table1_cohort.csv", index=False)
    log("Table 1 saved")


# ===========================================================================
# F. EPV
# ===========================================================================
def run_epv(y, fs):
    rows = []
    for o in OUTCOMES:
        ev = int(y[o].sum())
        r = dict(outcome=o, name=OUTCOME_NAMES[o], events=ev,
                 prevalence_pct=round(ev / len(y) * 100, 2))
        for s in STAGES:
            k = len(fs[s])
            r[f"n_features_{s}"] = k
            r[f"EPV_{s}"] = round(ev / k, 2)
        rows.append(r)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "epv.csv", index=False)
    log("EPV saved")


# ===========================================================================
# G. Imputation probe
# ===========================================================================
def run_imputation_probe(X, Y):
    from sklearn.pipeline import Pipeline
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import SimpleImputer, IterativeImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    fs = temporal_feature_sets(X)
    X72 = X[fs["72hr"]]
    folds = list(iterative_stratified_folds(Y, 5, seed=SEED))
    strats = {
        "Median": SimpleImputer(strategy="median"),
        "Median+Indicator": SimpleImputer(strategy="median", add_indicator=True),
        "Iterative": IterativeImputer(max_iter=10, random_state=SEED),
        "Iterative+Indicator": IterativeImputer(max_iter=10, random_state=SEED,
                                                add_indicator=True),
    }
    rows = []
    for name, imp in strats.items():
        t0 = time.time()
        oof = np.zeros((len(X72), 12))
        for tr, te in folds:
            # imputation+scaling is outcome-independent -> fit once per fold
            from sklearn.base import clone
            imp_f = clone(imp)
            Xtr_i = imp_f.fit_transform(X72.iloc[tr])
            Xte_i = imp_f.transform(X72.iloc[te])
            sc = StandardScaler().fit(Xtr_i)
            Xtr_s, Xte_s = sc.transform(Xtr_i), sc.transform(Xte_i)
            for j in range(12):
                lr = LogisticRegression(max_iter=5000, class_weight="balanced",
                                        random_state=SEED)
                lr.fit(Xtr_s, Y[tr, j])
                oof[te, j] = lr.predict_proba(Xte_s)[:, 1]
        aurocs = [auroc(Y[:, j], oof[:, j]) for j in range(12)]
        auprcs = [auprc(Y[:, j], oof[:, j]) for j in range(12)]
        rows.append(dict(strategy=name, macro_AUROC=float(np.nanmean(aurocs)),
                         macro_AUPRC=float(np.nanmean(auprcs)),
                         runtime_sec=round(time.time() - t0, 1)))
        log(f"  imputation probe {name}: macroAUROC={rows[-1]['macro_AUROC']:.4f} "
            f"macroAUPRC={rows[-1]['macro_AUPRC']:.4f} ({rows[-1]['runtime_sec']}s)")
    pd.DataFrame(rows).to_csv(TABLES_DIR / "imputation_probe.csv", index=False)


# ===========================================================================
# H. Literature comparison (curated from published sources)
# ===========================================================================
def run_literature():
    rows = [
        dict(study="Newaz et al. 2023 (Inform Med Unlocked)", model="XGBoost + resampling",
             task="Composite complication (binary)", dataset="UCI MI (1700)",
             validation="Train/test split", metric="AUC", value=0.8088,
             per_outcome="No", calibration="No", explainability="No",
             temporal_stages="Admission only"),
        dict(study="Ghafari et al. 2023 (J Tehran Heart Cent)", model="RF/XGBoost",
             task="In-hospital mortality", dataset="UCI MI (1700)",
             validation="Train/test split", metric="AUC", value=0.7865,
             per_outcome="No", calibration="No", explainability="No",
             temporal_stages="All features pooled"),
        dict(study="Khamis et al. 2024 (ETASR)", model="Gradient boosting",
             task="Composite complication (binary)", dataset="UCI MI (1700)",
             validation="Train/test split", metric="AUC", value=0.9920,
             per_outcome="No", calibration="No", explainability="No",
             temporal_stages="All features pooled"),
        dict(study="Satty et al. 2024 (ETASR)", model="ML comparison",
             task="Complications", dataset="UCI MI (1700)",
             validation="Split (imputation criticized)", metric="AUC", value=0.9010,
             per_outcome="No", calibration="No", explainability="No",
             temporal_stages="All features pooled"),
        dict(study="Vicente et al. 2023 (IEEE CSCI)", model="LightGBM",
             task="Mortality", dataset="UCI MI (1700)",
             validation="Train/test split", metric="F1", value=0.9120,
             per_outcome="No", calibration="No", explainability="Yes",
             temporal_stages="All features pooled"),
        dict(study="Makhmudov et al. 2025 (Bioengineering) [Base 2]",
             model="Multitask NN", task="11 complications + 8 death causes",
             dataset="UCI MI (1700)", validation="80/20 split + validation",
             metric="AUC (complications, val)", value=0.7847,
             per_outcome="Partial", calibration="No", explainability="Permutation importance",
             temporal_stages="All features pooled"),
        dict(study="Guo et al. 2026 (Sci Rep) [Base 1]", model="GNN (GraphTransformerNet)",
             task="12 complications incl. mortality", dataset="UCI MI (1700)",
             validation="10x5-fold CV", metric="Mean AUC (12 outcomes)", value=0.7330,
             per_outcome="Yes", calibration="No", explainability="SHAP + attention",
             temporal_stages="Derived temporal features"),
        dict(study="Guo et al. 2026 [Base 1] - mortality head", model="GNN",
             task="In-hospital mortality", dataset="UCI MI (1700)",
             validation="10x5-fold CV", metric="AUC", value=0.8828,
             per_outcome="-", calibration="No", explainability="SHAP",
             temporal_stages="Derived temporal features"),
        dict(study="This study (primary XGBoost, 72h)", model="Tuned XGBoost (OvR)",
             task="12 complications incl. mortality", dataset="UCI MI (1700)",
             validation="5x5-fold stratified CV (OOF)", metric="AUC DEATH",
             value=None, per_outcome="Yes", calibration="Yes (post-hoc)",
             explainability="SHAP", temporal_stages="4 cumulative stages + tests"),
        dict(study="This study (primary XGBoost, 72h)", model="Tuned XGBoost (OvR)",
             task="12 complications incl. mortality", dataset="UCI MI (1700)",
             validation="5x5-fold stratified CV (OOF)", metric="Macro AUC (12 outcomes)",
             value=None, per_outcome="Yes", calibration="Yes (post-hoc)",
             explainability="SHAP", temporal_stages="4 cumulative stages + tests"),
    ]
    df = pd.DataFrame(rows)
    # fill "this study" values from overall_metrics.csv
    om = pd.read_csv(TABLES_DIR / "overall_metrics.csv")
    xgb72 = om[(om.model == PRIMARY) & (om.stage == "72hr")].iloc[0]
    raw = pd.read_csv(TABLES_DIR / "raw_metrics.csv")
    d72 = raw[(raw.model == PRIMARY) & (raw.stage == "72hr") &
              (raw.outcome == "DEATH")]["AUROC"].iloc[0]
    df.loc[df.metric == "AUC DEATH", "value"] = round(float(d72), 4)
    df.loc[df.metric == "Macro AUC (12 outcomes)", "value"] = round(
        float(xgb72.macro_AUROC), 4)
    df.to_csv(TABLES_DIR / "literature_comparison.csv", index=False)
    log("literature comparison saved")


def main():
    X, y, variables, y_raw = load_raw()
    Y = y.values.astype(int)
    fs = temporal_feature_sets(X)
    groups = feature_type_groups(variables)

    log("A. best-calibrated OOF")
    cal = build_cal_best()
    cal72, _ = cal["72hr"]
    caladm, _ = cal["admission"]

    log("B. DCA")
    run_dca(Y, cal72, caladm)

    log("C. risk groups")
    run_risk_groups(Y, cal72)

    log("D. RQ3")
    run_rq3(X, Y, groups)

    log("E. Table 1")
    run_table1(X, y_raw)

    log("F. EPV")
    run_epv(y, fs)

    log("G. imputation probe")
    run_imputation_probe(X, Y)

    log("H. literature comparison")
    run_literature()

    log("UTILITY DONE")


if __name__ == "__main__":
    main()
