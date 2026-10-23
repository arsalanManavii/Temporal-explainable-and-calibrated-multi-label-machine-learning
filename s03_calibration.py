"""s03_calibration.py - Out-of-fold calibrated predictions for the tuned
XGBoost model (sigmoid and isotonic), at the admission and 72h stages.

Protocol (pre-specified):
- Same CV folds as s02 (deterministic, seed=10).
- Inside every training fold, CalibratedClassifierCV(cv=3) refits the base
  XGBoost and learns the calibration mapping using TRAINING data only;
  calibrated probabilities are produced for the held-out fold.
- Isotonic is only fitted for outcomes with >= 50 positive cases
  (isotonic is unstable on very rare events).
- Method selection rule: per outcome, the method with the lower OOF Brier
  score is chosen ('best'); all variants are reported.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OOF_DIR, OUTCOMES, SEED, N_SPLITS, N_REPEATS, XGB_PARAMS
from common import (load_raw, temporal_feature_sets, feature_type_groups,
                    build_preprocessor, repeated_multilabel_cv, log)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

CAL_STAGES = ["admission", "72hr"]
MIN_POS_ISOTONIC = 50


def main():
    X, y, variables, _ = load_raw()
    fs = temporal_feature_sets(X)
    groups = feature_type_groups(variables)
    Y = y.values.astype(np.int8)
    n = Y.shape[0]

    eligible_iso = {j: int(Y[:, j].sum()) >= MIN_POS_ISOTONIC
                    for j in range(len(OUTCOMES))}
    log("isotonic-eligible outcomes:",
        [OUTCOMES[j] for j in range(12) if eligible_iso[j]])

    for stage in CAL_STAGES:
        X_stage = X[fs[stage]]
        for method in ["sigmoid", "isotonic"]:
            tag = f"CAL_{method}_{stage}"
            out = OOF_DIR / f"{tag}.npz"
            ckpt = OOF_DIR / f"{tag}_ckpt.npz"
            if out.exists():
                log(f"skip existing {out.name}")
                continue
            if ckpt.exists():
                z = np.load(ckpt)
                prob = z["prob"]; done = int(z["done_repeats"])
                log(f"resume {tag} from repeat {done}")
            else:
                prob = np.zeros((N_REPEATS, n, len(OUTCOMES)), dtype=np.float32)
                done = 0
            t0 = time.time()
            for r, f, tr, te in repeated_multilabel_cv(Y, N_SPLITS, N_REPEATS, SEED):
                if r < done:
                    continue
                pre = build_preprocessor(X_stage.iloc[tr], groups, scale=False)
                pre.fit(X_stage.iloc[tr])
                Xtr = pre.transform(X_stage.iloc[tr])
                Xte = pre.transform(X_stage.iloc[te])
                inner = StratifiedKFold(n_splits=3, shuffle=True,
                                        random_state=SEED)
                for j in range(len(OUTCOMES)):
                    if method == "isotonic" and not eligible_iso[j]:
                        continue
                    pos = int(Y[tr, j].sum())
                    spw = (len(tr) - pos) / pos if pos > 0 else 1.0
                    base = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
                    cal = CalibratedClassifierCV(estimator=base, method=method,
                                                 cv=inner)
                    cal.fit(Xtr, Y[tr, j])
                    prob[r, te, j] = cal.predict_proba(Xte)[:, 1].astype(np.float32)
                log(f"  [{tag}] repeat {r} fold {f} ({time.time()-t0:.0f}s)")
                if f == N_SPLITS - 1:
                    np.savez_compressed(ckpt, prob=prob, done_repeats=r + 1)
            np.savez_compressed(out, prob=prob)
            log(f"saved {out.name} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
