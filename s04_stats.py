"""s04_stats.py - Central analysis engine.

Reads the OOF prediction tensors saved by s02/s03 and produces every
numeric result of the study as tidy CSVs in results/tables/:

  raw_metrics.csv      point estimates: model x stage x outcome (AUROC, AUPRC)
  overall_metrics.csv  macro/micro summaries per model x stage (+ bootstrap CI)
  ci_metrics.csv       bootstrap 95% CI (primary model per outcome x stage)
  cls_metrics.csv      classification metrics under the threshold policies
  thresholds.csv       selected thresholds per outcome
  temporal_tests.csv   DeLong + paired bootstrap for stage comparisons (RQ2/RQ4)
  model_tests.csv      paired tests between models at 72h (incl. OvR vs Chain)
  calibration.csv      Brier / ECE / ICI / slope-intercept, uncalibrated vs
                       sigmoid vs isotonic vs best (admission & 72h)
  reliability_bins.csv binned data for reliability-curve figures
  subgroups.csv        age/sex subgroup metrics with CI (primary model, 72h)

Conventions (documented in README):
- Point estimates use OOF probabilities averaged over the 5 CV repeats.
- 95% CIs: patient-level percentile bootstrap (B=2000, seed=10).
- AUROC comparisons: DeLong test (paired, same patients & folds).
- AUPRC comparisons: paired bootstrap.
- p-values across the 12 outcomes of one comparison are Holm-adjusted.
- Thresholds: F1-maximising on OOF for standard outcomes; for high-risk
  outcomes (high conditional lethality) the lowest threshold achieving
  >= 90% recall is used (clinical sensitivity policy).
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (OOF_DIR, TABLES_DIR, OUTCOMES, OUTCOME_NAMES, STAGES,
                    STAGE_LABELS, SEED, N_REPEATS, HIGH_RISK_OUTCOMES,
                    CLINICAL_MIN_RECALL, N_BOOTSTRAP)
from common import (load_raw, auroc, auprc, classification_metrics,
                    expected_calibration_error, integrated_calibration_index,
                    brier, delong_pvalue, paired_bootstrap_delta, log)
from sklearn.metrics import f1_score

MODELS = ["LR", "RF", "XGB", "LGBM", "CHAIN"]
PRIMARY = "XGB"
B_FULL = 1000       # bootstrap replicates for the wide grid
B_KEY = 2000        # for primary-model CIs


def load_oof(model: str, stage: str) -> np.ndarray:
    return np.load(OOF_DIR / f"{model}_{stage}.npz")["prob"].mean(axis=0)


def load_cal(method: str, stage: str) -> np.ndarray:
    return np.load(OOF_DIR / f"CAL_{method}_{stage}.npz")["prob"].mean(axis=0)


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        prev = max(prev, min(val, 1.0))
        adj[idx] = prev
    return adj.tolist()


def boot_ci_metric(y_true, score, metric, B=B_KEY, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(idx):
            vals[b] = np.nan
        else:
            vals[b] = metric(y_true[idx], score[idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def select_threshold(y_true, prob, outcome):
    if outcome in HIGH_RISK_OUTCOMES:
        pos = prob[y_true == 1]
        if len(pos):
            thr = float(np.quantile(pos, 1 - CLINICAL_MIN_RECALL,
                                    method="lower"))
            if (pos >= thr).mean() >= CLINICAL_MIN_RECALL:
                return thr, "clinical_recall90"
    grid = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 199)))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, "f1_max"


def main():
    X, y, variables, y_raw = load_raw()
    Y = y.values.astype(int)
    n = Y.shape[0]
    age = X["AGE"].values
    sex = X["SEX"].values

    raw_rows, overall_rows, ci_rows, cls_rows, thr_rows = [], [], [], [], []

    log("=== point estimates + overall metrics ===")
    oof_store = {}
    for m in MODELS:
        for s in STAGES:
            P = load_oof(m, s)
            oof_store[(m, s)] = P
            aurocs, auprcs = [], []
            for j, o in enumerate(OUTCOMES):
                a = auroc(Y[:, j], P[:, j]); p = auprc(Y[:, j], P[:, j])
                aurocs.append(a); auprcs.append(p)
                raw_rows.append(dict(model=m, stage=s, outcome=o,
                                     AUROC=a, AUPRC=p, n_pos=int(Y[:, j].sum())))
            pred = np.zeros_like(P, dtype=int)
            for j, o in enumerate(OUTCOMES):
                t, pol = select_threshold(Y[:, j], P[:, j], o)
                pred[:, j] = (P[:, j] >= t).astype(int)
                thr_rows.append(dict(model=m, stage=s, outcome=o, policy=pol,
                                     threshold=t))
            micro_f1 = f1_score(Y, pred, average="micro", zero_division=0)
            macro_f1 = f1_score(Y, pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(Y, pred, average="weighted", zero_division=0)
            overall_rows.append(dict(
                model=m, stage=s,
                macro_AUROC=float(np.nanmean(aurocs)),
                macro_AUPRC=float(np.nanmean(auprcs)),
                micro_F1=float(micro_f1), macro_F1=float(macro_f1),
                weighted_F1=float(weighted_f1)))
            for j, o in enumerate(OUTCOMES):
                mm = classification_metrics(Y[:, j], pred[:, j])
                cls_rows.append(dict(model=m, stage=s, outcome=o, **mm))
    pd.DataFrame(raw_rows).to_csv(TABLES_DIR / "raw_metrics.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(TABLES_DIR / "thresholds.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(TABLES_DIR / "cls_metrics.csv", index=False)

    log("=== bootstrap CIs: primary model (all outcomes x stages) ===")
    for s in STAGES:
        P = oof_store[(PRIMARY, s)]
        for j, o in enumerate(OUTCOMES):
            for mname, met in [("AUROC", auroc), ("AUPRC", auprc)]:
                lo, hi = boot_ci_metric(Y[:, j], P[:, j], met)
                ci_rows.append(dict(model=PRIMARY, stage=s, outcome=o,
                                    metric=mname, lo=lo, hi=hi))
        log(f"  CI {s} done")

    log("=== bootstrap CIs: macro metrics, all models x stages ===")
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(B_FULL)]
    for m in MODELS:
        for s in STAGES:
            P = oof_store[(m, s)]
            macro_a, macro_p = [], []
            for idx in boot_idx:
                aa = [auroc(Y[idx, j], P[idx, j]) for j in range(12)]
                pp = [auprc(Y[idx, j], P[idx, j]) for j in range(12)]
                macro_a.append(np.nanmean(aa)); macro_p.append(np.nanmean(pp))
            lo_a, hi_a = np.nanpercentile(macro_a, [2.5, 97.5])
            lo_p, hi_p = np.nanpercentile(macro_p, [2.5, 97.5])
            ci_rows.append(dict(model=m, stage=s, outcome="MACRO",
                                metric="AUROC", lo=float(lo_a), hi=float(hi_a)))
            ci_rows.append(dict(model=m, stage=s, outcome="MACRO",
                                metric="AUPRC", lo=float(lo_p), hi=float(hi_p)))
        log(f"  macro CI {m} done")
    pd.DataFrame(overall_rows).to_csv(TABLES_DIR / "overall_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(TABLES_DIR / "ci_metrics.csv", index=False)

    log("=== temporal incremental value (DeLong + paired bootstrap) ===")
    temp_rows = []
    for j, o in enumerate(OUTCOMES):
        base = oof_store[(PRIMARY, "admission")][:, j]
        for s in STAGES[1:]:
            cur = oof_store[(PRIMARY, s)][:, j]
            d_auc, p_del = delong_pvalue(Y[:, j], base, cur)
            d_ap, lo, hi, p_bt = paired_bootstrap_delta(
                Y[:, j], base, cur, metric=auprc, n_boot=B_FULL)
            temp_rows.append(dict(outcome=o, comparison=f"{STAGE_LABELS[s]} vs Admission",
                                  stage=s, AUROC_base=auroc(Y[:, j], base),
                                  AUROC_new=auroc(Y[:, j], cur),
                                  delta_AUROC=d_auc, p_DeLong=p_del,
                                  AUPRC_base=auprc(Y[:, j], base),
                                  AUPRC_new=auprc(Y[:, j], cur),
                                  delta_AUPRC=d_ap, dAUPRC_lo=lo, dAUPRC_hi=hi,
                                  p_boot_AUPRC=p_bt))
        # consecutive-step comparisons (24 vs adm, 48 vs 24, 72 vs 48)
    for j, o in enumerate(OUTCOMES):
        for a, b in [("admission", "24hr"), ("24hr", "48hr"), ("48hr", "72hr")]:
            sa = oof_store[(PRIMARY, a)][:, j]
            sb = oof_store[(PRIMARY, b)][:, j]
            d_auc, p_del = delong_pvalue(Y[:, j], sa, sb)
            temp_rows.append(dict(outcome=o,
                                  comparison=f"{STAGE_LABELS[b]} vs {STAGE_LABELS[a]}",
                                  stage=f"{a}->{b}",
                                  AUROC_base=auroc(Y[:, j], sa),
                                  AUROC_new=auroc(Y[:, j], sb),
                                  delta_AUROC=d_auc, p_DeLong=p_del,
                                  AUPRC_base=np.nan, AUPRC_new=np.nan,
                                  delta_AUPRC=np.nan, dAUPRC_lo=np.nan,
                                  dAUPRC_hi=np.nan, p_boot_AUPRC=np.nan))
    tdf = pd.DataFrame(temp_rows)
    # Holm adjustment within each comparison across the 12 outcomes
    tdf["p_DeLong_holm"] = np.nan
    for comp, g in tdf.groupby("comparison"):
        tdf.loc[g.index, "p_DeLong_holm"] = holm(g["p_DeLong"].tolist())
    tdf.to_csv(TABLES_DIR / "temporal_tests.csv", index=False)
    log("  temporal tests done")

    log("=== model comparison tests at 72h (vs XGB) ===")
    mrows = []
    P_ref = oof_store[(PRIMARY, "72hr")]
    for m in MODELS:
        if m == PRIMARY:
            continue
        P = oof_store[(m, "72hr")]
        for j, o in enumerate(OUTCOMES):
            d_auc, p_del = delong_pvalue(Y[:, j], P_ref[:, j], P[:, j])
            d_ap, lo, hi, p_bt = paired_bootstrap_delta(
                Y[:, j], P_ref[:, j], P[:, j], metric=auprc, n_boot=B_FULL)
            mrows.append(dict(model=m, vs=PRIMARY, outcome=o,
                              delta_AUROC=d_auc, p_DeLong=p_del,
                              delta_AUPRC=d_ap, p_boot_AUPRC=p_bt))
    mdf = pd.DataFrame(mrows)
    mdf["p_DeLong_holm"] = np.nan
    for mm, g in mdf.groupby("model"):
        mdf.loc[g.index, "p_DeLong_holm"] = holm(g["p_DeLong"].tolist())
    mdf.to_csv(TABLES_DIR / "model_tests.csv", index=False)
    log("  model tests done")

    log("=== calibration ===")
    cal_rows, rel_rows = [], []
    for s in ["admission", "72hr"]:
        P_unc = oof_store[(PRIMARY, s)]
        variants = {"uncalibrated": P_unc}
        for meth in ["sigmoid", "isotonic"]:
            f = OOF_DIR / f"CAL_{meth}_{s}.npz"
            if f.exists():
                variants[meth] = load_cal(meth, s)
        # per-outcome 'best' by Brier
        bestP = P_unc.copy()
        for j, o in enumerate(OUTCOMES):
            cands = {k: brier(Y[:, j], v[:, j]) for k, v in variants.items()
                     if not np.isnan(v[:, j]).any()}
            bm = min(cands, key=cands.get)
            bestP[:, j] = variants[bm][:, j]
            cal_rows.append(dict(stage=s, outcome=o, best_method=bm))
        variants["best"] = bestP
        for vname, P in variants.items():
            for j, o in enumerate(OUTCOMES):
                if np.isnan(P[:, j]).all():
                    continue
                b = brier(Y[:, j], P[:, j])
                ece = expected_calibration_error(Y[:, j], P[:, j], n_bins=10)
                ici = integrated_calibration_index(Y[:, j], P[:, j])
                # Cox calibration slope/intercept (logit scale)
                pp = np.clip(P[:, j], 1e-6, 1 - 1e-6)
                lp = np.log(pp / (1 - pp)).reshape(-1, 1)
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=2000, random_state=SEED)
                lr.fit(lp, Y[:, j])
                slope = float(lr.coef_[0, 0]); inter = float(lr.intercept_[0])
                cal_rows.append(dict(stage=s, outcome=o, variant=vname,
                                     Brier=b, ECE=ece, ICI=ici,
                                     cal_slope=slope, cal_intercept=inter,
                                     mean_pred=float(pp.mean()),
                                     obs_rate=float(Y[:, j].mean())))
                # reliability bins
                edges = np.linspace(0, 1, 11)
                for b_i in range(10):
                    lo_e, hi_e = edges[b_i], edges[b_i + 1]
                    mask = (P[:, j] >= lo_e) & (P[:, j] < hi_e) if b_i < 9 \
                        else (P[:, j] >= lo_e) & (P[:, j] <= hi_e)
                    if mask.sum() == 0:
                        continue
                    rel_rows.append(dict(stage=s, outcome=o, variant=vname,
                                         bin_lo=lo_e, bin_hi=hi_e,
                                         bin_mid=(lo_e + hi_e) / 2,
                                         n=int(mask.sum()),
                                         observed=float(Y[mask, j].mean()),
                                         predicted=float(P[mask, j].mean())))
        log(f"  calibration {s} done")
    pd.DataFrame(cal_rows).to_csv(TABLES_DIR / "calibration.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(TABLES_DIR / "reliability_bins.csv", index=False)

    log("=== subgroups (primary model, 72h, OOF) ===")
    P = oof_store[(PRIMARY, "72hr")]
    srows = []
    sub_defs = [("Age", "<65", age < 65), ("Age", ">=65", age >= 65),
                ("Sex", "Female", sex == 0), ("Sex", "Male", sex == 1)]
    for sname, gname, mask in sub_defs:
        microF1 = None
        for j, o in enumerate(OUTCOMES):
            yt = Y[mask, j]
            a = auroc(yt, P[mask, j]); p = auprc(yt, P[mask, j])
            lo_a = hi_a = lo_p = hi_p = np.nan
            if yt.sum() >= 5 and len(np.unique(yt)) == 2:
                lo_a, hi_a = boot_ci_metric(yt, P[mask, j], auroc, B=B_FULL)
                lo_p, hi_p = boot_ci_metric(yt, P[mask, j], auprc, B=B_FULL)
            srows.append(dict(subgroup=sname, group=gname, outcome=o,
                              N=int(mask.sum()), n_pos=int(yt.sum()),
                              AUROC=a, AUROC_lo=lo_a, AUROC_hi=hi_a,
                              AUPRC=p, AUPRC_lo=lo_p, AUPRC_hi=hi_p))
        # overall macro for subgroup
        aa = [auroc(Y[mask, j], P[mask, j]) for j in range(12)]
        pp_ = [auprc(Y[mask, j], P[mask, j]) for j in range(12)]
        srows.append(dict(subgroup=sname, group=gname, outcome="MACRO",
                          N=int(mask.sum()), n_pos=int(Y[mask].sum()),
                          AUROC=float(np.nanmean(aa)), AUROC_lo=np.nan,
                          AUROC_hi=np.nan, AUPRC=float(np.nanmean(pp_)),
                          AUPRC_lo=np.nan, AUPRC_hi=np.nan))
    pd.DataFrame(srows).to_csv(TABLES_DIR / "subgroups.csv", index=False)

    # save mean OOF matrices for downstream scripts (figures/utility)
    np.savez_compressed(OOF_DIR / "oof_mean_all.npz",
                        **{f"{m}__{s}": oof_store[(m, s)]
                           for m in MODELS for s in STAGES})
    log("ALL STATS DONE")


if __name__ == "__main__":
    main()
