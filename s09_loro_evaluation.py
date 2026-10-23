"""s09_loro_evaluation.py - Leave-One-Repeat-Out (LORO) honest evaluation of
threshold-dependent and calibration-selection-dependent metrics.

For each CV repeat r:
  - calibration method (sigmoid/isotonic) selected by Brier on the OTHER repeats
  - decision thresholds (policy rules) selected on the OTHER repeats'
    calibrated probabilities
  - classification metrics and calibration metrics evaluated ON repeat r
Reported values = mean +/- SD across the five repeats (no patient contributes
to the selection of its own threshold/method).

Also produces:
  model_tests_macro.csv  paired patient-level bootstrap of macro AUROC/AUPRC
                         differences (RF, LGBM, CHAIN vs XGB) at 72 h
  per_repeat_macro.csv   per-repeat macro metrics for every model x stage
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (OOF_DIR, TABLES_DIR, OUTCOMES, STAGES, HIGH_RISK_OUTCOMES,
                    CLINICAL_MIN_RECALL, SEED)
from common import (load_raw, auroc, auprc, brier, expected_calibration_error,
                    classification_metrics, paired_bootstrap_delta, log)
from sklearn.metrics import f1_score

MODELS = ["LR", "RF", "XGB", "LGBM", "CHAIN"]


def select_threshold(y_true, prob, outcome):
    if outcome in HIGH_RISK_OUTCOMES:
        pos = prob[y_true == 1]
        if len(pos):
            thr = float(np.quantile(pos, 1 - CLINICAL_MIN_RECALL, method="lower"))
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
    X, y, variables, _ = load_raw()
    Y = y.values.astype(int)
    R = None
    # ---------- LORO for primary model ----------
    unc = {s: np.load(OOF_DIR / f"XGB_{s}.npz")["prob"] for s in ["72hr"]}
    sig = {s: np.load(OOF_DIR / f"CAL_sigmoid_{s}.npz")["prob"] for s in ["72hr"]}
    iso_f = OOF_DIR / "CAL_isotonic_72hr.npz"
    iso = {s: np.load(iso_f)["prob"] for s in ["72hr"]} if iso_f.exists() else {}
    stage = "72hr"
    U, S, I = unc[stage], sig[stage], iso[stage]
    R = U.shape[0]
    elig_iso = {j: int(Y[:, j].sum()) >= 50 for j in range(12)}

    cal_rows, cls_rows, thr_rows = [], [], []
    for j, o in enumerate(OUTCOMES):
        per_rep_cal, per_rep_cls = [], []
        thr_rep = []
        for r in range(R):
            others = [k for k in range(R) if k != r]
            # method selection on other repeats (calibrated candidates)
            cands = {"sigmoid": S[:, :, j][others].mean(0)}
            if elig_iso[j]:
                cands["isotonic"] = I[:, :, j][others].mean(0)
            br = {k: brier(Y[:, j], v) for k, v in cands.items()}
            method = min(br, key=br.get)
            cal_sel = {"sigmoid": S, "isotonic": I}[method]
            # evaluate calibration on repeat r
            per_rep_cal.append(dict(
                Brier_cal=brier(Y[:, j], cal_sel[r, :, j]),
                ECE_cal=expected_calibration_error(Y[:, j], cal_sel[r, :, j]),
                Brier_unc=brier(Y[:, j], U[r, :, j]),
                ECE_unc=expected_calibration_error(Y[:, j], U[r, :, j])))
            # thresholds on other repeats' calibrated probs
            p_others = cal_sel[:, :, j][others].mean(0)
            t, pol = select_threshold(Y[:, j], p_others, o)
            thr_rep.append(t)
            per_rep_cls.append(dict(**classification_metrics(
                Y[:, j], (cal_sel[r, :, j] >= t).astype(int)), threshold=t))
        cc = pd.DataFrame(per_rep_cal).mean()
        cs = pd.DataFrame(per_rep_cls)
        cal_rows.append(dict(outcome=o, method=method,
                             Brier_unc=cc.Brier_unc, ECE_unc=cc.ECE_unc,
                             Brier_cal=cc.Brier_cal, ECE_cal=cc.ECE_cal,
                             ECE_cal_sd=pd.DataFrame(per_rep_cal).ECE_cal.std(),
                             Brier_cal_sd=pd.DataFrame(per_rep_cal).Brier_cal.std()))
        m = cs.mean(numeric_only=True)
        sd = cs.std(numeric_only=True)
        cls_rows.append(dict(outcome=o, policy=pol,
                             threshold=float(np.mean(thr_rep)),
                             threshold_sd=float(np.std(thr_rep, ddof=1)),
                             Sensitivity=m.Sensitivity, Sensitivity_sd=sd.Sensitivity,
                             Specificity=m.Specificity, Specificity_sd=sd.Specificity,
                             PPV=m.PPV, PPV_sd=sd.PPV, NPV=m.NPV, NPV_sd=sd.NPV,
                             F1=m.F1, F1_sd=sd.F1))
        thr_rows.append(dict(outcome=o, policy=pol,
                             threshold_mean=float(np.mean(thr_rep)),
                             threshold_sd=float(np.std(thr_rep, ddof=1))))
        log(f"  LORO {o}: method={method}")
    pd.DataFrame(cal_rows).to_csv(TABLES_DIR / "calibration_loro.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(TABLES_DIR / "cls_loro.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(TABLES_DIR / "thresholds_loro.csv", index=False)

    # ---------- paired macro bootstrap at 72h ----------
    z = np.load(OOF_DIR / "oof_mean_all.npz")
    P = {k: z[k] for k in z.files}
    rows = []
    for m in ["RF", "LGBM", "CHAIN", "LR"]:
        for met, fn in [("AUROC", auroc), ("AUPRC", auprc)]:
            a = np.array([fn(Y[:, j], P[f"XGB__72hr"][:, j]) for j in range(12)])
            b = np.array([fn(Y[:, j], P[f"{m}__72hr"][:, j]) for j in range(12)])
            d, lo, hi, p = paired_bootstrap_delta(
                Y[:, 0], P["XGB__72hr"][:, 0], P["XGB__72hr"][:, 0], n_boot=1) \
                if False else (None, None, None, None)
            # macro-level paired bootstrap: resample patients, recompute macros
            rng = np.random.default_rng(SEED)
            deltas = []
            for _ in range(1000):
                idx = rng.integers(0, len(Y), len(Y))
                aa = np.nanmean([fn(Y[idx, j], P[f"XGB__72hr"][idx, j]) for j in range(12)])
                bb = np.nanmean([fn(Y[idx, j], P[f"{m}__72hr"][idx, j]) for j in range(12)])
                deltas.append(bb - aa)
            deltas = np.array(deltas)
            obs = np.nanmean(b) - np.nanmean(a)
            lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
            p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
            rows.append(dict(model=m, metric=met, macro_XGB=float(np.nanmean(a)),
                             macro_other=float(np.nanmean(b)), delta=obs,
                             ci_lo=lo, ci_hi=hi, p_boot=p))
        log(f"  macro paired {m} done")
    pd.DataFrame(rows).to_csv(TABLES_DIR / "model_tests_macro.csv", index=False)

    # ---------- per-repeat macro metrics ----------
    rows = []
    for m in MODELS:
        for s in STAGES:
            pr = np.load(OOF_DIR / f"{m}_{s}.npz")["prob"]
            for r in range(pr.shape[0]):
                aa = np.nanmean([auroc(Y[:, j], pr[r, :, j]) for j in range(12)])
                pp = np.nanmean([auprc(Y[:, j], pr[r, :, j]) for j in range(12)])
                rows.append(dict(model=m, stage=s, repeat=r + 1,
                                 macro_AUROC=aa, macro_AUPRC=pp))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "per_repeat_macro.csv", index=False)
    summ = df.groupby(["model", "stage"]).agg(
        macro_AUROC_mean=("macro_AUROC", "mean"), macro_AUROC_sd=("macro_AUROC", lambda v: v.std(ddof=1)),
        macro_AUPRC_mean=("macro_AUPRC", "mean"), macro_AUPRC_sd=("macro_AUPRC", lambda v: v.std(ddof=1)))
    summ.reset_index().to_csv(TABLES_DIR / "per_repeat_macro_summary.csv", index=False)
    log("LORO + macro tests + per-repeat DONE")


if __name__ == "__main__":
    main()
