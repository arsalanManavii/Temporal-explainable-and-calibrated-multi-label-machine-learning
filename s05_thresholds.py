"""s05_thresholds.py - recompute threshold policy outputs with the
data-driven high-risk outcome set (conditional lethality >= 25%).

Regenerates: thresholds.csv, cls_metrics.csv, overall_metrics.csv,
and saves conditional_lethality.csv (justification table).
Fast: no model refits, operates on saved OOF matrices.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (OOF_DIR, TABLES_DIR, OUTCOMES, STAGES, HIGH_RISK_OUTCOMES,
                    CLINICAL_MIN_RECALL)
from common import load_raw, auroc, auprc, classification_metrics, log
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
    X, y, variables, y_raw = load_raw()
    Y = y.values.astype(int)

    # conditional lethality table
    death = (y_raw["LET_IS"] > 0).values
    leth = []
    for o in OUTCOMES[:-1]:
        c = y[o].values == 1
        leth.append(dict(outcome=o, n=int(c.sum()),
                         conditional_lethality_pct=float(death[c].mean() * 100) if c.sum() else np.nan,
                         risk_tier="high (>=25%)" if o in HIGH_RISK_OUTCOMES else "standard"))
    pd.DataFrame(leth).to_csv(TABLES_DIR / "conditional_lethality.csv", index=False)
    log("saved conditional_lethality.csv")

    z = np.load(OOF_DIR / "oof_mean_all.npz")
    P = {k: z[k] for k in z.files}

    thr_rows, cls_rows, overall_rows = [], [], []
    for m in MODELS:
        for s in STAGES:
            Pm = P[f"{m}__{s}"]
            pred = np.zeros_like(Pm, dtype=int)
            aurocs, auprcs = [], []
            for j, o in enumerate(OUTCOMES):
                t, pol = select_threshold(Y[:, j], Pm[:, j], o)
                pred[:, j] = (Pm[:, j] >= t).astype(int)
                thr_rows.append(dict(model=m, stage=s, outcome=o, policy=pol,
                                     threshold=t))
                mm = classification_metrics(Y[:, j], pred[:, j])
                cls_rows.append(dict(model=m, stage=s, outcome=o, **mm))
                aurocs.append(auroc(Y[:, j], Pm[:, j]))
                auprcs.append(auprc(Y[:, j], Pm[:, j]))
            overall_rows.append(dict(
                model=m, stage=s,
                macro_AUROC=float(np.nanmean(aurocs)),
                macro_AUPRC=float(np.nanmean(auprcs)),
                micro_F1=float(f1_score(Y, pred, average="micro", zero_division=0)),
                macro_F1=float(f1_score(Y, pred, average="macro", zero_division=0)),
                weighted_F1=float(f1_score(Y, pred, average="weighted", zero_division=0))))
        log(f"  {m} done")

    pd.DataFrame(thr_rows).to_csv(TABLES_DIR / "thresholds.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(TABLES_DIR / "cls_metrics.csv", index=False)

    # merge macro AUROC/AUPRC CIs back into overall (from ci_metrics.csv)
    ci = pd.read_csv(TABLES_DIR / "ci_metrics.csv")
    macro_ci = ci[ci.outcome == "MACRO"].copy()
    wide = {}
    for met in ["AUROC", "AUPRC"]:
        sub = macro_ci[macro_ci.metric == met].set_index(["model", "stage"])
        wide[f"{met}_lo"] = sub["lo"]
        wide[f"{met}_hi"] = sub["hi"]
    macro_wide = pd.DataFrame(wide).reset_index()
    ov = pd.DataFrame(overall_rows)
    ov = ov.merge(macro_wide, on=["model", "stage"], how="left")
    ov.to_csv(TABLES_DIR / "overall_metrics.csv", index=False)
    log("saved thresholds.csv, cls_metrics.csv, overall_metrics.csv")


if __name__ == "__main__":
    main()
