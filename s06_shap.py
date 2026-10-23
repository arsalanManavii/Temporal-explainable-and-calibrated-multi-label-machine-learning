"""s06_shap.py - SHAP explanations for the primary model (tuned XGBoost,
per-outcome scale_pos_weight) at the full-information stage (72h).

Uses the pre-specified fixed 80/20 split (seed=10) for interpretable
local explanations (global model-level SHAP on the held-out 20%),
consistent with the original analysis. Quantitative performance claims
of the paper rely on the cross-validated OOF results (s02-s04), not on
this split.

Saves:
  results/oof/shap_values.npz     (12 x 340 x n_features SHAP values)
  results/oof/shap_X.npz          (feature matrix + names for plots)
  results/tables/shap_top5.csv    (rank 1-5 per outcome, mean |SHAP|)
  results/tables/shap_top15.csv   (rank 1-15 per outcome)
  results/tables/shap_global_rank.csv (mean |SHAP| rank per outcome)
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (OOF_DIR, TABLES_DIR, OUTCOMES, SEED, FIXED_TEST_SIZE,
                    XGB_PARAMS)
from common import (load_raw, temporal_feature_sets, feature_type_groups,
                    build_preprocessor, log)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap


def main():
    X, y, variables, _ = load_raw()
    fs = temporal_feature_sets(X)
    groups = feature_type_groups(variables)
    Y = y.values.astype(int)
    X72 = X[fs["72hr"]]

    idx = np.arange(len(X72))
    tr, te = train_test_split(idx, test_size=FIXED_TEST_SIZE, random_state=SEED)
    log(f"fixed split: train={len(tr)} test={len(te)}")

    pre = build_preprocessor(X72.iloc[tr], groups, scale=False)
    pre.fit(X72.iloc[tr])
    Xtr = pre.transform(X72.iloc[tr])
    Xte = pre.transform(X72.iloc[te])
    names = pre.get_feature_names_out()
    log(f"processed: {Xtr.shape} / {Xte.shape}")

    shap_vals = np.zeros((len(OUTCOMES), len(te), Xte.shape[1]),
                         dtype=np.float32)
    top5, top15, globalrank = [], [], []
    t0 = time.time()
    for j, o in enumerate(OUTCOMES):
        pos = int(Y[tr, j].sum()); neg = len(tr) - pos
        spw = neg / pos if pos > 0 else 1.0
        model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.fit(Xtr, Y[tr, j])
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(Xte).astype(np.float32)
        shap_vals[j] = sv
        mabs = np.abs(sv).mean(axis=0)
        order = np.argsort(mabs)[::-1]
        for rank, ix in enumerate(order[:15], 1):
            row = dict(outcome=o, rank=rank, feature=names[ix],
                       mean_abs_shap=float(mabs[ix]))
            top15.append(row)
            if rank <= 5:
                top5.append(row)
        for rank, ix in enumerate(order, 1):
            globalrank.append(dict(outcome=o, rank=rank, feature=names[ix],
                                   mean_abs_shap=float(mabs[ix])))
        log(f"  SHAP {o} done ({time.time()-t0:.0f}s)")

    np.savez_compressed(OOF_DIR / "shap_values.npz", values=shap_vals)
    np.savez_compressed(OOF_DIR / "shap_X.npz", X=Xte.astype(np.float32),
                        names=np.array(names, dtype=object),
                        test_idx=te, train_idx=tr,
                        allow_pickle=True)
    pd.DataFrame(top5).to_csv(TABLES_DIR / "shap_top5.csv", index=False)
    pd.DataFrame(top15).to_csv(TABLES_DIR / "shap_top15.csv", index=False)
    pd.DataFrame(globalrank).to_csv(TABLES_DIR / "shap_global_rank.csv",
                                    index=False)
    log("SHAP DONE")


if __name__ == "__main__":
    main()
