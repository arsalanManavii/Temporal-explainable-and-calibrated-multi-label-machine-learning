"""s14_supplementary_package.py - build the FINAL supplementary deliverables:

  1) supplementary/Supplementary_Appendix.docx
       Supplementary Text S1 + Tables S1-S8 + Figures S1-S5 (embedded, captions)
  2) supplementary/Supplementary_Tables_machine_readable.xlsx
       sheets: README, T1-T9, S1-S8, features, engineered, outcomes,
       lethality, stages
  3) (TRIPOD checklist refreshed separately by s10_supplementary.py)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (SUPP_DIR, TABLES_DIR, FIGURES_DIR, OUTCOMES, OUTCOME_NAMES,
                    STAGES, STAGE_LABELS, TIMEPOINT_EXCLUSION, SEED, N_SPLITS,
                    N_REPEATS, N_BOOTSTRAP)
from common import load_raw, temporal_feature_sets, log

FINAL = TABLES_DIR / "final"

SUPP_FIGS = [
    ("S1", "F04_roc_grid", "Per-outcome receiver operating characteristic curves of the primary model at the 72 h stage (out-of-fold predictions). Dashed line: chance."),
    ("S2", "F05_pr_grid", "Per-outcome precision-recall curves of the primary model at the 72 h stage (out-of-fold predictions). Dashed line: outcome prevalence."),
    ("S3", "F09_subgroup_forest", "Subgroup discrimination of the primary model (72 h) by age and sex with bootstrap 95% confidence intervals."),
    ("S4", "F11_model_comparison", "Macro AUROC and macro AUPRC of all models at the 72 h stage with bootstrap 95% confidence intervals."),
    ("S5", "F12_ovr_vs_chain", "Per-outcome comparison of binary relevance versus classifier chain (mean of 10 random orders) at the 72 h stage."),
]

SUPP_TABLE_NOTES = {
    "S1": "Logistic-regression probe, 5-fold iterative-stratified cross-validation, 72 h feature set; macro metrics averaged over the twelve outcomes.",
    "S2": "Conditional in-hospital mortality given each complication; basis of the high-lethality threshold tier (>=25%).",
    "S3": "Phi/Pearson correlations between outcome labels in the full cohort.",
    "S4": "Complete out-of-fold results for every model, stage and outcome, including classification metrics at the pre-specified thresholds.",
    "S5": "Macro AUROC per random chain order at the 72 h stage (10 orders).",
    "S6": "Observed rates with Wilson 95% CI per quintile of calibrated predicted risk (72 h).",
    "S7": "Out-of-fold subgroup performance of the primary model (72 h); CIs by patient-level bootstrap where >=5 events.",
    "S8": "Effect of adding engineered clinical-course summaries to the raw 72 h feature set (identical CV protocol).",
}


def build_appendix():
    from docx import Document
    from docx.shared import Pt, Inches
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(10)
    h = doc.add_heading("Supplementary Appendix", 0)
    for r in h.runs: r.font.name = "Arial"
    p = doc.add_paragraph("Temporal, explainable and calibrated multi-label machine learning for "
                          "outcome-specific risk stratification after acute myocardial infarction")
    p.runs[0].italic = True

    # ---------------- Supplementary Text S1 ----------------
    doc.add_heading("Supplementary Text S1. Extended protocol details", level=1)
    paras = [
        "Cross-validation diagnostics. Five folds x five repeats were constructed by iterative "
        "multi-label stratification on the twelve outcomes. The maximum absolute deviation of any "
        "outcome's prevalence between a fold and the full cohort was 0.24 percentage points "
        "(test-fold sizes 340 patients). Identical fold assignments were reused for every model, "
        "stage, calibration run and statistical comparison, so that all between-stage and "
        "between-model tests are paired on the same patients.",
        "Missing-data strategy probe. Four handling strategies were compared with a logistic-"
        "regression probe (balanced class weights) under 5-fold cross-validation on the 72 h "
        "feature set: median imputation (macro AUROC 0.669, macro AUPRC 0.179), median imputation "
        "with missing indicators (0.672, 0.199), iterative imputation (0.662, 0.178) and iterative "
        "imputation with indicators (0.666, 0.197). Because the indicator-augmented median strategy "
        "matched the best alternative at ~1% of its runtime (5 s vs 431 s per fold set), it was "
        "pre-specified for all models (Supplementary Table S1).",
        "Calibration method selection. For each outcome, sigmoid and isotonic recalibration were "
        "fitted inside training folds (internal 3-fold). Method selection and all reported "
        "calibration metrics used a leave-one-repeat-out (LORO) scheme: the variant with the lower "
        "Brier score on four repeats was evaluated on the held-out repeat, removing "
        "selection-on-evaluation optimism. At the 72 h stage isotonic regression was selected for "
        "eight of twelve outcomes; sigmoid was selected for supraventricular tachycardia, "
        "ventricular tachycardia, Dressler syndrome and post-infarction angina. Isotonic fitting "
        "was restricted to outcomes with >=50 events.",
        "Threshold derivation. Thresholds were derived from CALIBRATED probabilities. F1-maximising "
        "thresholds were obtained by scanning the 1st-99th percentiles; high-lethality thresholds "
        "were the lowest cutoffs achieving >=90% recall. For honest classification metrics the same "
        "LORO scheme was applied: thresholds selected on four repeats were applied to the held-out "
        "repeat, and Table 7 reports mean +/- SD across repeats (held-out recall of the mortality "
        "policy 0.905 +/- 0.015).",
        "Decision-curve and chain-order settings. Net benefit was evaluated on a 0.5-60% threshold "
        "grid in 0.5% steps. Classifier-chain predictions were averaged over ten random chain "
        "orders (seed offsets 10-19); macro AUROC across orders was 0.7138 +/- 0.0032 (SD, ddof=1) "
        "(range 0.7087-0.7188; Supplementary Table S5).",
        "Sensitivity analyses. (A) Removing the missing indicators of the day-2/day-3 clinical-course "
        "variables - whose missingness marks patients who died before those days (mortality 93.6-96.3% "
        "among such patients) - abolished the temporal gain for mortality (72 h AUROC 0.906 vs 0.935 "
        "primary) while gains for pulmonary edema, chronic heart failure and recurrent infarction "
        "persisted (Supplementary Table S9). (B) Removing ICU treatment variables left mortality and "
        "heart-failure discrimination essentially unchanged but reduced pulmonary edema and "
        "ventricular fibrillation AUROC by 0.07-0.09, confirming that part of their signal encodes "
        "treatment response rather than baseline risk (Supplementary Table S9). (C) An unweighted "
        "XGBoost reference achieved comparable discrimination (macro AUROC 0.719) and was already "
        "well calibrated (mean ECE 0.015), showing that the calibration improvement of the primary "
        "model corrects the intentional probability inflation induced by class weighting "
        "(Supplementary Table S9b).",
        "Software and reproducibility. Python 3.11; scikit-learn 1.9.1; XGBoost 3.2.0; LightGBM "
        "4.7.0; SHAP 0.51.0; pandas 3.0.6; numpy 2.4.6. All stochastic operations used seed 10. "
        "Out-of-fold prediction tensors, analysis tables and the full pipeline are released in the "
        "public repository (code, results and out-of-fold tensors), allowing every reported number "
        "to be recomputed.",
    ]
    for t in paras:
        pp = doc.add_paragraph(t); pp.paragraph_format.line_spacing = 1.4

    # ---------------- Supplementary Tables ----------------
    for key in ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S9b","S10","S10b"]:
        i = int(key.lstrip('S').rstrip('b')) if key[1:].rstrip('b').isdigit() else key
        df = pd.read_csv(FINAL / f"{key}.csv")
        doc.add_heading(f"Supplementary Table S{i}.", level=1).runs[0].font.size = Pt(11)
        cap = {
            "S1": "Imputation-strategy probe (logistic-regression probe, 5-fold CV, 72 h features).",
            "S2": "Conditional lethality of each complication (basis of the threshold policy).",
            "S3": "Label correlation matrix between the twelve outcomes.",
            "S4": "Complete per-outcome results for all models and information stages.",
            "S5": "Classifier-chain order variability at the 72 h stage (10 random orders).",
            "S6": "Risk-group stratification by quintiles of calibrated predicted risk (72 h).",
            "S7": "Subgroup performance of the primary model (72 h) by age and sex.",
            "S8": "Clinical-course engineered features added to the raw 72 h set (RQ3).",
            "S9": "Sensitivity analyses A/B: censoring-linked missing indicators and ICU treatment variables.",
            "S9b": "Sensitivity analysis C: unweighted XGBoost discrimination and calibration.",
            "S10": "Paired macro-metric bootstrap comparisons of all models vs the primary model (72 h).",
            "S10b": "Per-repeat macro metrics (mean +/- SD over 5 repeats) for all models and stages.",
        }[key]
        cp = doc.add_paragraph(cap); cp.runs[0].italic = True; cp.runs[0].font.size = Pt(9)
        t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
        for k, cname in enumerate(df.columns):
            t.rows[0].cells[k].text = str(cname)
            for par in t.rows[0].cells[k].paragraphs:
                for r in par.runs: r.bold = True; r.font.size = Pt(7.5); r.font.name = "Arial"
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for k, v in enumerate(row):
                cells[k].text = "" if pd.isna(v) else str(v)
                for par in cells[k].paragraphs:
                    for r in par.runs: r.font.size = Pt(7.5); r.font.name = "Arial"
        note = SUPP_TABLE_NOTES.get(key, "")
        if note:
            np_ = doc.add_paragraph(note)
            np_.runs[0].italic = True; np_.runs[0].font.size = Pt(8)

    # ---------------- Supplementary Figures ----------------
    for tag, fname, cap in SUPP_FIGS:
        doc.add_page_break()
        doc.add_picture(str(FIGURES_DIR / f"{fname}.png"), width=Inches(6.4))
        cp = doc.add_paragraph(f"Supplementary Figure {tag}. {cap}")
        cp.runs[0].italic = True; cp.runs[0].font.size = Pt(9)

    out = SUPP_DIR / "Supplementary_Appendix.docx"
    doc.save(out)
    log(f"saved {out.name}")


def build_workbook():
    X, y, variables, y_raw = load_raw()
    fs = temporal_feature_sets(X)
    sheets = {}

    readme = pd.DataFrame({"item": [
        "Workbook contents", "T1_stages", "T2_cohort", "T2b_prevalence", "T3_overall",
        "T4_temporal", "T5_per_outcome", "T6_calibration", "T7_thresholds_DCA",
        "T8_SHAP", "T9_literature", "S1..S8", "features / engineered / outcomes / lethality",
        "Provenance"],
        "description": [
        "Machine-readable versions of all manuscript and supplementary tables",
        "Temporal information stages and withheld UCI columns (manuscript Table 1)",
        "Baseline cohort characteristics (manuscript Table 2, panel A)",
        "Outcome prevalence and events-per-variable (manuscript Table 2, panel B)",
        "Overall model performance by stage (manuscript Table 3)",
        "Incremental value 72 h vs admission with paired tests (manuscript Table 4)",
        "Per-outcome primary-model performance (manuscript Table 5)",
        "Calibration before/after post-hoc calibration (manuscript Table 6)",
        "Thresholds, classification metrics and DCA net benefit (manuscript Table 7)",
        "SHAP top-5 predictors per outcome (manuscript Table 8)",
        "Literature positioning (manuscript Table 9)",
        "Supplementary tables S1-S8",
        "Feature dictionary sheets",
        f"Generated by the released pipeline; seed {SEED}; 5x{N_SPLITS}-fold CV, "
        f"{N_REPEATS} repeats; bootstrap B={N_BOOTSTRAP}"]})
    sheets["README"] = readme

    stages = pd.DataFrame({
        "stage": [STAGE_LABELS[s] for s in STAGES],
        "n_features": [len(fs[s]) for s in STAGES],
        "withheld_uci_columns": [", ".join(map(str, TIMEPOINT_EXCLUSION[s])) or "none"
                                  for s in STAGES]})
    sheets["T1_stages"] = stages
    sheets["T2_cohort"] = pd.read_csv(TABLES_DIR / "table1_cohort.csv")
    sheets["T2b_prevalence"] = pd.read_csv(TABLES_DIR / "epv.csv")
    sheets["T3_overall"] = pd.read_csv(TABLES_DIR / "overall_metrics.csv")
    ci = pd.read_csv(TABLES_DIR / "ci_metrics.csv")
    mac = ci[ci.outcome == "MACRO"].copy()
    wide = {}
    for met in ["AUROC", "AUPRC"]:
        sub = mac[mac.metric == met].set_index(["model", "stage"])
        wide[f"{met}_lo"] = sub["lo"]; wide[f"{met}_hi"] = sub["hi"]
    mac_w = pd.DataFrame(wide).reset_index()
    sheets["T3_overall"] = sheets["T3_overall"].merge(
        mac_w, on=["model", "stage"], how="left")
    tt = pd.read_csv(TABLES_DIR / "temporal_tests.csv")
    sheets["T4_temporal"] = tt
    raw = pd.read_csv(TABLES_DIR / "raw_metrics.csv")
    cls = pd.read_csv(TABLES_DIR / "cls_metrics.csv")
    thr = pd.read_csv(TABLES_DIR / "thresholds.csv")
    t5 = raw[(raw.model == "XGB")][["stage", "outcome", "AUROC", "AUPRC"]].merge(
        ci[(ci.model == "XGB") & (ci.metric == "AUROC")][["stage", "outcome", "lo", "hi"]],
        on=["stage", "outcome"], how="left", suffixes=("", "_auc_ci"))
    t5 = t5.merge(ci[(ci.model == "XGB") & (ci.metric == "AUPRC")][["stage", "outcome", "lo", "hi"]],
                  on=["stage", "outcome"], how="left", suffixes=("_auc_ci", "_auprc_ci"))
    sheets["T5_per_outcome"] = t5.round(4)
    loro = pd.read_csv(TABLES_DIR / "calibration_loro.csv")
    fullc = pd.read_csv(TABLES_DIR / "calibration.csv")
    sl = {}
    for meth in ["sigmoid", "isotonic"]:
        sub = fullc[(fullc.stage == "72hr") & (fullc.variant == meth)].set_index("outcome")
        sl[meth] = sub[["cal_slope", "cal_intercept"]]
    loro = loro.copy()
    loro["slope"] = [sl[m].loc[o, "cal_slope"] for m, o in zip(loro.method, loro.outcome)]
    loro["intercept"] = [sl[m].loc[o, "cal_intercept"] for m, o in zip(loro.method, loro.outcome)]
    sheets["T6_calibration"] = loro.round(4)
    dca = pd.read_csv(TABLES_DIR / "dca_summary.csv")
    thr_l = pd.read_csv(TABLES_DIR / "thresholds_loro.csv")
    cls_l = pd.read_csv(TABLES_DIR / "cls_loro.csv")
    sheets["T7_thresholds"] = thr_l.merge(cls_l, on="outcome", how="left").round(4)
    sheets["T7b_DCA"] = dca
    sheets["T8_SHAP"] = pd.read_csv(TABLES_DIR / "shap_top5.csv")
    sheets["T9_literature"] = pd.read_csv(TABLES_DIR / "literature_comparison.csv")
    for nm in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
               "S9", "S9b", "S10", "S10b"]:
        sheets[nm] = pd.read_csv(FINAL / f"{nm}.csv")
    feat = pd.read_excel(SUPP_DIR / "feature_dictionary.xlsx", sheet_name=None)
    for name, df in feat.items():
        sheets[f"dict_{name}"[:31]] = df

    out = SUPP_DIR / "Supplementary_Tables_machine_readable.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    log(f"saved {out.name} with {len(sheets)} sheets")


if __name__ == "__main__":
    build_appendix()
    build_workbook()
    log("SUPPLEMENTARY PACKAGE DONE")
