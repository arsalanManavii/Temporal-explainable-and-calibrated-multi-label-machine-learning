"""s01_tuning.py - reproducible reproduction of the XGBoost hyper-parameter
search (RandomizedSearchCV-equivalent, 40 candidates, 3-fold CV, macro-F1,
admission-stage training portion of the pre-specified 80/20 split).
Confirms the hyper-parameters frozen in config.XGB_PARAMS."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TABLES_DIR, OUTCOMES, SEED, XGB_PARAMS
from common import (load_raw, temporal_feature_sets, feature_type_groups,
                    build_preprocessor, log)
from sklearn.model_selection import train_test_split, KFold, ParameterSampler
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

GRID = dict(
    n_estimators=[100, 200, 300, 400],
    max_depth=[3, 4, 5, 6, 7],
    learning_rate=[0.01, 0.05, 0.1, 0.15],
    subsample=[0.7, 0.8, 0.9, 1.0],
    colsample_bytree=[0.7, 0.8, 0.9, 1.0],
    min_child_weight=[1, 3, 5],
    gamma=[0, 0.1, 0.2],
    scale_pos_weight=[1, 3, 5, 8, 10],
)

def main():
    X, y, variables, _ = load_raw()
    fs = temporal_feature_sets(X)
    groups = feature_type_groups(variables)
    Y = y.values.astype(int)
    Xad = X[fs["admission"]]
    idx = np.arange(len(Xad))
    tr, _ = train_test_split(idx, test_size=0.2, random_state=SEED)
    pre = build_preprocessor(Xad.iloc[tr], groups, scale=False)
    pre.fit(Xad.iloc[tr])
    Xtr = pre.transform(Xad.iloc[tr]); Ytr = Y[tr]
    skf = list(KFold(3).split(Xtr))   # identical to RandomizedSearchCV(cv=3) on multi-label y
    rows = []
    t0 = time.time()
    for it, params in enumerate(ParameterSampler(GRID, n_iter=40, random_state=SEED)):
        params = dict(params)
        preds = np.zeros_like(Ytr)
        for ftr, fte in skf:
            for j in range(12):
                m = XGBClassifier(**{**params, "random_state": SEED,
                                     "eval_metric": "logloss", "n_jobs": 2,
                                     "tree_method": "hist"})
                m.fit(Xtr[ftr], Ytr[ftr, j])
                preds[fte, j] = m.predict(Xtr[fte])
        score = f1_score(Ytr, preds, average="macro", zero_division=0)
        rows.append({**params, "macro_F1_CV": score})
        log(f"  iter {it}: macro-F1 {score:.4f} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows).sort_values("macro_F1_CV", ascending=False)
    df.to_csv(TABLES_DIR / "tuning_log.csv", index=False)
    best = df.iloc[0]
    log(f"BEST: {best.to_dict()}")
    # evaluate the FROZEN config parameters under the identical audit protocol
    preds = np.zeros_like(Ytr)
    for ftr, fte in skf:
        for j in range(12):
            m = XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": 10})
            m.fit(Xtr[ftr], Ytr[ftr, j])
            preds[fte, j] = m.predict(Xtr[fte])
    cfg_score = f1_score(Ytr, preds, average="macro", zero_division=0)
    rows.append({**{k: XGB_PARAMS[k] for k in GRID if k != "scale_pos_weight"},
                 "scale_pos_weight": 10, "macro_F1_CV": cfg_score,
                 "tag": "config_frozen"})
    df = pd.DataFrame(rows).sort_values("macro_F1_CV", ascending=False)
    df.to_csv(TABLES_DIR / "tuning_log.csv", index=False)
    log(f"config_frozen macro-F1 under audit protocol: {cfg_score:.4f} vs best {best.macro_F1_CV:.4f}")

if __name__ == "__main__":
    main()
