"""s02_models.py - Main CV engine: out-of-fold predictions for every
model x temporal stage combination.

Usage:
    python s02_models.py --models XGB,LR        [--stages admission,24hr,48hr,72hr]
    python s02_models.py --models CHAIN --stages admission,24hr

Saves per (model, stage):
    results/oof/{model}_{stage}.npz with
      prob:  (n_repeats, n_patients, 12) float32  - out-of-fold probabilities
      meta:  json-ish dict saved alongside

Protocol notes (documented in README/Methods):
- Repeated iterative-stratified multi-label CV (5 folds x 5 repeats, seed=10).
- Identical folds for every model/stage (deterministic splitter) so that
  paired comparisons (DeLong / paired bootstrap) are valid.
- Preprocessing (median impute + missing indicator [+ scaler for LR],
  one-hot for ZSN_A) is fitted INSIDE each training fold only.
- Class imbalance: per-outcome scale_pos_weight = n_neg/n_pos from the
  training fold (XGB/LGBM); class_weight='balanced' (LR);
  'balanced_subsample' (RF). CHAIN uses a single global weight
  (median of per-outcome ratios) because sklearn chains share one base
  estimator across links.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (OOF_DIR, OUTCOMES, STAGES, STAGE_LABELS, SEED, N_SPLITS,
                    N_REPEATS, XGB_PARAMS, LGBM_PARAMS, RF_PARAMS, LR_PARAMS,
                    N_CHAIN_ORDERS)
from common import (load_raw, temporal_feature_sets, feature_type_groups,
                    build_preprocessor, repeated_multilabel_cv, log)

MODEL_REGISTRY = ["LR", "RF", "XGB", "LGBM", "CHAIN"]


def make_estimator(model: str, spw: float | None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    if model == "LR":
        return LogisticRegression(**LR_PARAMS)
    if model == "RF":
        return RandomForestClassifier(**RF_PARAMS)
    if model == "XGB":
        return XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
    if model == "LGBM":
        return LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw)
    raise ValueError(model)


def run_ovr(model: str, stage: str, Xs, Y, groups):
    X_stage = Xs[stage]
    n = X_stage.shape[0]
    oof = np.zeros((N_REPEATS, n, len(OUTCOMES)), dtype=np.float32)
    t_start = time.time()
    for r, f, tr, te in repeated_multilabel_cv(Y, N_SPLITS, N_REPEATS, SEED):
        scale = (model == "LR")
        pre = build_preprocessor(X_stage.iloc[tr], groups, scale=scale)
        pre.fit(X_stage.iloc[tr])
        Xtr = pre.transform(X_stage.iloc[tr])
        Xte = pre.transform(X_stage.iloc[te])
        Ytr = Y[tr]
        for j, out in enumerate(OUTCOMES):
            pos = int(Ytr[:, j].sum()); neg = len(tr) - pos
            spw = (neg / pos) if pos > 0 else 1.0
            est = make_estimator(model, spw)
            est.fit(Xtr, Ytr[:, j])
            oof[r, te, j] = est.predict_proba(Xte)[:, 1].astype(np.float32)
        log(f"  [{model}|{stage}] repeat {r} fold {f} done "
            f"({time.time()-t_start:.0f}s)")
    return oof


def run_chain(stage: str, Xs, Y, groups, n_orders: int = N_CHAIN_ORDERS):
    """ClassifierChain + tuned XGB, averaged over `n_orders` random chain
    orders. Global scale_pos_weight = median of per-outcome imbalance ratios
    (computed on full data; chains cannot vary weights per link).

    Checkpoints after every repeat so an interrupted run can resume."""
    from sklearn.multioutput import ClassifierChain
    from xgboost import XGBClassifier
    X_stage = Xs[stage]
    n = X_stage.shape[0]
    ratios = []
    for j in range(len(OUTCOMES)):
        pos = int(Y[:, j].sum())
        ratios.append((n - pos) / pos if pos > 0 else 1.0)
    global_spw = float(np.median(ratios))

    ckpt = OOF_DIR / f"CHAIN_{stage}_ckpt.npz"
    if ckpt.exists():
        z = np.load(ckpt)
        acc = z["acc"]; done_repeats = int(z["done_repeats"])
        log(f"  [CHAIN|{stage}] resuming from checkpoint "
            f"({done_repeats}/{N_REPEATS} repeats done)")
    else:
        acc = np.zeros((n_orders, N_REPEATS, n, len(OUTCOMES)), dtype=np.float32)
        done_repeats = 0

    t_start = time.time()
    for r, f, tr, te in repeated_multilabel_cv(Y, N_SPLITS, N_REPEATS, SEED):
        if r < done_repeats:
            continue
        pre = build_preprocessor(X_stage.iloc[tr], groups, scale=False)
        pre.fit(X_stage.iloc[tr])
        Xtr = pre.transform(X_stage.iloc[tr])
        Xte = pre.transform(X_stage.iloc[te])
        for o in range(n_orders):
            base = XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": global_spw})
            chain = ClassifierChain(estimator=base, order="random",
                                    random_state=SEED + o)
            chain.fit(Xtr, Y[tr])
            acc[o, r, te, :] = chain.predict_proba(Xte).astype(np.float32)
        log(f"  [CHAIN|{stage}] repeat {r} fold {f} done "
            f"({time.time()-t_start:.0f}s)")
        if f == N_SPLITS - 1:   # end of repeat -> checkpoint
            np.savez_compressed(ckpt, acc=acc, done_repeats=r + 1)
    oof = acc.mean(axis=0)          # average across chain orders
    np.savez_compressed(OOF_DIR / f"CHAIN_{stage}_orders.npz", orders=acc)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--chain-orders", type=int, default=N_CHAIN_ORDERS)
    args = ap.parse_args()
    models = [m.strip().upper() for m in args.models.split(",")]
    stages = [s.strip() for s in args.stages.split(",")]

    X, y, variables, _ = load_raw()
    fs = temporal_feature_sets(X)
    Xs = {s: X[fs[s]] for s in STAGES}
    groups = feature_type_groups(variables)
    Y = y.values.astype(np.int8)

    for model in models:
        assert model in MODEL_REGISTRY, model
        for stage in stages:
            out_file = OOF_DIR / f"{model}_{stage}.npz"
            if out_file.exists():
                log(f"skip existing {out_file.name}")
                continue
            t0 = time.time()
            log(f"=== {model} | {STAGE_LABELS[stage]} ===")
            if model == "CHAIN":
                oof = run_chain(stage, Xs, Y, groups, args.chain_orders)
            else:
                oof = run_ovr(model, stage, Xs, Y, groups)
            np.savez_compressed(out_file, prob=oof)
            with open(OOF_DIR / f"{model}_{stage}_meta.json", "w") as fh:
                json.dump({"model": model, "stage": stage,
                           "n_repeats": N_REPEATS, "n_splits": N_SPLITS,
                           "seed": SEED,
                           "n_features": len(fs[stage]),
                           "runtime_sec": round(time.time() - t0, 1)}, fh)
            log(f"saved {out_file.name} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
