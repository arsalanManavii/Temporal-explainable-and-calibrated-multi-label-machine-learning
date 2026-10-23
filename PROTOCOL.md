# Pre-specification record (analysis protocol)

Frozen design decisions, recorded before outcome-level results were inspected.
Analyses added during peer-review revision (sensitivity analyses A-C, LORO
threshold/calibration evaluation) are explicitly labelled post-hoc in the
manuscript and supplementary material.

1. Data: UCI MI Complications (ID 579), 1,700 records, no exclusions.
2. Outcomes: 11 binary complications + in-hospital mortality (LET_IS > 0); multi-label.
3. Information stages: admission (102 predictors), +24 h (105), +48 h (108), +72 h (111),
   per UCI column exclusions [93-95,100-105] / [94,95,101,102,104,105] / [95,102,105] / none.
4. Missing data: median imputation + missing indicators (chosen by pre-specified probe).
5. Validation: 5-fold x 5-repeat iterative multi-label stratified CV, identical folds for
   all models/stages; out-of-fold probabilities averaged over repeats for point estimates.
6. Models: binary relevance LR / RF / XGBoost / LightGBM + classifier chain (10 orders).
   Primary model pre-specified: tuned XGBoost (per-outcome scale_pos_weight; TreeExplainer
   on log-odds; continuity with originating protocol). Reference learners use defaults.
7. Hyper-parameters: single RandomizedSearchCV (40 candidates, 3-fold, macro-F1) on the
   training portion of the pre-specified 80/20 split; frozen across stages (audit: s01_tuning.py).
8. Calibration: fold-internal sigmoid/isotonic (isotonic only if >=50 events); method chosen
   by leave-one-repeat-out Brier; metrics evaluated on the held-out repeat.
9. Thresholds: F1-max (standard outcomes); lowest cutoff with >=90% recall for high-lethality
   outcomes (conditional mortality >=25%, computed from outcome labels only); derived from
   calibrated probabilities; honest metrics via leave-one-repeat-out.
10. Metrics: per-outcome AUROC/AUPRC (AUPRC primary for imbalance); macro averages;
    micro/macro/weighted F1 at policy thresholds.
11. Inference: patient-level percentile bootstrap (B=2000 primary, 1000 elsewhere);
    paired DeLong for AUROC; paired bootstrap for AUPRC and macro metrics;
    Holm adjustment across the 12 outcomes per comparison; alpha=0.05 two-sided.
12. Explainability: SHAP TreeExplainer per outcome on the held-out 20% of the pre-specified
    split (illustrative, not inferential).
13. Sensitivity (post-hoc, revision): A no day-2/3 missing indicators; B no ICU treatment
    variables; C unweighted model; all with identical folds.
14. Seeds: global seed 10; chain order seeds 11-20.
