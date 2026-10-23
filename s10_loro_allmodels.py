"""s15_loro_allmodels.py - LORO classification metrics for EVERY model x stage
(honest held-out classification metrics for every model and stage; Supplementary Table S4)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OOF_DIR, TABLES_DIR, OUTCOMES, STAGES, HIGH_RISK_OUTCOMES, CLINICAL_MIN_RECALL
from common import load_raw, classification_metrics, log
from sklearn.metrics import f1_score

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
    rows = []
    for m in ["LR", "RF", "XGB", "LGBM", "CHAIN"]:
        for s in STAGES:
            P = np.load(OOF_DIR / f"{m}_{s}.npz")["prob"]
            R = P.shape[0]
            for j, o in enumerate(OUTCOMES):
                per = []
                thrs = []
                for r in range(R):
                    others = [k for k in range(R) if k != r]
                    t, pol = select_threshold(Y[:, j], P[:, :, j][others].mean(0), o)
                    thrs.append(t)
                    per.append(classification_metrics(Y[:, j], (P[r, :, j] >= t).astype(int)))
                df = pd.DataFrame(per)
                rows.append(dict(model=m, stage=s, outcome=o, policy=pol,
                                 threshold_mean=df.index.size and float(np.mean(thrs)),
                                 threshold_sd=float(np.std(thrs, ddof=1)),
                                 **{f"{k}_mean": df[k].mean() for k in
                                    ["Sensitivity", "Specificity", "PPV", "NPV", "F1"]},
                                 **{f"{k}_sd": df[k].std(ddof=1) for k in
                                    ["Sensitivity", "Specificity", "PPV", "NPV", "F1"]}))
        log(f"  LORO-all {m} done")
    pd.DataFrame(rows).to_csv(TABLES_DIR / "cls_loro_all.csv", index=False)
    log("cls_loro_all.csv saved")

if __name__ == "__main__":
    main()
