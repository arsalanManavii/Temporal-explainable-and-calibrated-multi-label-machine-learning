"""s13_supplementary_docs.py - supplementary deliverables:
  supplementary/TRIPOD_AI_checklist.docx
  supplementary/feature_dictionary.xlsx
  supplementary/statements.md  (ethics / data / code availability / funding)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (SUPP_DIR, DATA_DIR, OUTCOMES, OUTCOME_NAMES, STAGES,
                    STAGE_LABELS, SEED, N_SPLITS, N_REPEATS, XGB_PARAMS)
from common import load_raw, temporal_feature_sets, feature_type_groups, log

TRIPOD_ITEMS = [
    # (section, item, where addressed)
    ("Title", "1 Identification of the study as using artificial intelligence and study design", "Title/Abstract: 'temporal, explainable, calibrated multi-label machine learning'; Methods: repeated cross-validation prediction-model study"),
    ("Abstract", "2 Structured abstract following TRIPOD+AI", "Abstract: structured (Background, Methods, Results, Conclusions), 284 words; manuscript file 06"),
    ("Introduction", "3-4 Rationale and objectives (including AI-specific rationale)", "Introduction: clinical rationale for outcome-specific, stage-wise risk stratification; RQ1-RQ7"),
    ("Methods", "5-6 Study design, setting, data sources", "Methods: secondary analysis of the public UCI MI Complications dataset (Golovenkin et al. 2020); single-centre Krasnoyarsk cohort 1992-1995"),
    ("Methods", "7-8 Eligibility criteria, participant flow, sample size", "Methods 2.1-2.2; Results 3.1: all 1700 records, no exclusions (no missing outcomes); EPV in Table 2 panel B addresses effective sample size"),
    ("Methods", "9-10 Outcome definition and timing", "Methods: 11 binary complications + in-hospital mortality (LET_IS>0); outcomes ascertained during index hospitalisation"),
    ("Methods", "11-12 Predictor definitions and timing (temporal availability)", "Methods 2.2 + Table 1 + supplementary workbook sheets dict_features_111/dict_temporal_stages: 4 cumulative information stages (102/105/108/111 features) per UCI documentation"),
    ("Methods", "13-14 Missing data: extent and handling", "Methods: 8.47% overall missingness; median imputation + missing indicators inside each CV fold; imputation-strategy probe (Suppl. Table S1); missingness patterns analysed"),
    ("Methods", "15-16 Data preprocessing and feature engineering", "Methods + code: fold-internal preprocessing; engineered clinical-course summaries (pain relapse, opioid, NSAID) analysed in RQ3"),
    ("Methods", "17-18 Model specifications and hyperparameters", "Methods + config.py: LR, RF, tuned XGBoost, LightGBM (binary relevance), classifier chain (10 random orders); tuned hyperparameters reported"),
    ("Methods", "19-20 Class imbalance handling", "Methods: per-outcome scale_pos_weight (n_neg/n_pos) from training folds; class_weight='balanced' (LR), 'balanced_subsample' (RF); NO synthetic resampling (justified vs prior work)"),
    ("Methods", "21-22 Validation strategy", "Methods: repeated 5x5-fold iterative multi-label stratified CV, out-of-fold predictions; identical folds across models/stages for paired tests; pre-specified fixed split used only for SHAP illustrations"),
    ("Methods", "23-24 Performance measures and comparisons", "Methods: AUROC, AUPRC, F1/sensitivity/specificity/PPV/NPV at pre-specified threshold policies; macro/micro summaries; DeLong and paired-bootstrap tests with Holm adjustment"),
    ("Methods", "25 Calibration assessment and adjustment", "Methods: post-hoc sigmoid/isotonic calibration fitted fold-internally; Brier, ECE(10 bins), ICI, calibration slope/intercept; per-outcome method chosen by OOF Brier (pre-specified)"),
    ("Methods", "26-27 Clinical utility assessment", "Methods: decision-curve analysis; risk-quintile stratification; threshold policies (F1-max; recall>=90% for high-lethality outcomes defined by conditional mortality >=25%)"),
    ("Methods", "28 Explainability plan", "Methods: SHAP TreeExplainer per outcome (global top predictors; local explanations); engineered-feature SHAP ranks in RQ3"),
    ("Methods", "29-30 Implementation, software, code availability", "Code: version-pinned requirements.txt; full pipeline released; seed=10 everywhere"),
    ("Results", "31-33 Study population, flow, descriptive statistics", "Results: Table 2 (panels A-B), Figure 2"),
    ("Results", "34-35 Model performance (discrimination) with uncertainty", "Results: Tables 3-5, Figures 3, S1-S2, S4; bootstrap 95% CIs"),
    ("Results", "36 Calibration results", "Results: Table 6, Figure 4"),
    ("Results", "37 Clinical utility results", "Results: Table 7 panel B, Figures 5-6, Supplementary Table S6"),
    ("Results", "38 Explainability results", "Results: Table 8, Figure 7"),
    ("Results", "39 Subgroup/fairness results", "Results: Supplementary Table S7, Supplementary Figure S3"),
    ("Discussion", "40-43 Interpretation, limitations, generalisability", "Discussion: single-centre 1990s cohort; rare outcomes underpowered (EPV<1); no external validation; secondary analysis of public data; honest reporting of non-improving outcomes"),
    ("Other", "44-46 Funding, competing interests, ethics", "Statements file (supplementary/statements.md)"),
    ("Other", "47-50 Registration, protocol, data/code availability", "Data: UCI repository DOI 10.24432/C53P5M; code released with manuscript; no prospective registration (retrospective prediction-model development)"),
]


def build_tripod():
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(9)
    h = doc.add_heading("TRIPOD+AI checklist - completed for the present study", 0)
    for r in h.runs:
        r.font.name = "Arial"
    t = doc.add_table(rows=1, cols=3)
    t.style = "Light Grid Accent 1"
    for i, cname in enumerate(["Section", "Checklist item (abridged)", "Where addressed in this study"]):
        c = t.rows[0].cells[i]; c.text = cname
        for p in c.paragraphs:
            for r in p.runs:
                r.bold = True; r.font.size = Pt(8); r.font.name = "Arial"
    for sec, item, where in TRIPOD_ITEMS:
        cells = t.add_row().cells
        for i, v in enumerate([sec, item, where]):
            cells[i].text = v
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8); r.font.name = "Arial"
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("Note: page/line numbers will be added at journal formatting; all analytic items are "
                  "fully implemented and traceable to the released code modules (s01-s14 and verify_results.py) and to the "
                  "tables/figures listed above. Items added during revision (sensitivity analyses, "
                  "leave-one-repeat-out evaluation) are marked post-hoc in the manuscript.")
    r.italic = True; r.font.size = Pt(8); r.font.name = "Arial"
    doc.save(SUPP_DIR / "TRIPOD_AI_checklist.docx")
    log("saved TRIPOD_AI_checklist.docx")


def build_feature_dictionary():
    X, y, variables, y_raw = load_raw()
    fs = temporal_feature_sets(X)
    groups = feature_type_groups(variables)

    df = variables[variables.role == "Feature"].copy()
    df = df[df.name.isin(X.columns)]
    df["group"] = ""
    for g, lst in groups.items():
        df.loc[df.name.isin(lst), "group"] = g
    for s in STAGES:
        df[f"available_{s}"] = df.name.isin(fs[s]).map({True: "yes", False: "no"})
    df["missing_pct"] = [round(X[c].isna().mean() * 100, 2) if c in X else np.nan
                         for c in df.name]
    df = df[["name", "type", "group", "description", "units",
             "available_admission", "available_24hr", "available_48hr",
             "available_72hr", "missing_pct"]]
    df.columns = ["Feature", "UCI type", "Model group", "Description", "Units",
                  "Admission", "+24h", "+48h", "+72h", "Missing (%)"]

    eng = pd.DataFrame([
        ["pain_relapse_total", "engineered (ordinal-like)", "Sum of anginal attack counts, days 1-3 (R_AB_1..3_n)"],
        ["pain_relapse_active_days", "engineered", "Number of days (1-3) with >=1 anginal attack"],
        ["pain_relapse_max", "engineered", "Maximum daily anginal attack count, days 1-3"],
        ["pain_relapse_late", "engineered (binary)", "Any anginal attack on day 2 or 3"],
        ["opioid_use_total", "engineered", "Sum of opioid doses, days 1-3 (NA_R_1..3_n)"],
        ["opioid_use_active_days", "engineered", "Days with >=1 opioid dose"],
        ["opioid_use_max", "engineered", "Maximum daily opioid doses"],
        ["nsaid_use_total", "engineered", "Sum of NSAID doses, days 1-3 (NOT_NA_1..3_n)"],
        ["nsaid_use_active_days", "engineered", "Days with >=1 NSAID dose"],
        ["nsaid_use_max", "engineered", "Maximum daily NSAID doses"],
    ], columns=["Feature", "UCI type", "Description"])
    eng["Model group"] = "engineered (RQ3 analysis only)"
    eng["Units"] = ""
    for s in STAGES:
        eng[f"available_{s}"] = np.where([True]*len(eng), "72h only" if s == "72hr" else "no", "")
    eng["Missing (%)"] = ""

    outcomes = pd.DataFrame({
        "Outcome code": OUTCOMES,
        "Clinical name": [OUTCOME_NAMES[o] for o in OUTCOMES],
        "Definition": ["binary complication indicator (UCI)" if o != "DEATH"
                       else "in-hospital mortality = LET_IS > 0 (any lethal outcome cause)"
                       for o in OUTCOMES],
        "Positive n": [int(y[o].sum()) for o in OUTCOMES],
        "Prevalence %": [round(y[o].mean() * 100, 2) for o in OUTCOMES],
    })
    lethal = pd.read_csv(Path(__file__).resolve().parents[1] / "results" /
                         "tables" / "conditional_lethality.csv")
    stages = pd.DataFrame({
        "Stage": [STAGE_LABELS[s] for s in STAGES],
        "Features available": [len(fs[s]) for s in STAGES],
        "Excluded UCI columns": [", ".join(map(str, {
            "admission": [93, 94, 95, 100, 101, 102, 103, 104, 105],
            "24hr": [94, 95, 101, 102, 104, 105],
            "48hr": [95, 102, 105], "72hr": []}[s])) or "none" for s in STAGES],
    })

    out = SUPP_DIR / "feature_dictionary.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="features_111", index=False)
        eng.to_excel(w, sheet_name="engineered_rq3", index=False)
        outcomes.to_excel(w, sheet_name="outcomes_12", index=False)
        lethal.to_excel(w, sheet_name="conditional_lethality", index=False)
        stages.to_excel(w, sheet_name="temporal_stages", index=False)
    log(f"saved {out.name}")


def build_statements():
    txt = """# Statements (for manuscript submission)

## Ethics statement
This study is a secondary analysis of the fully anonymised, publicly available
"Myocardial Infarction Complications" dataset (UCI Machine Learning Repository,
ID 579; Golovenkin S.E. et al., 2020; DOI 10.24432/C53P5M), collected at the
Krasnoyarsk Interdistrict Clinical Hospital No. 20 (1992-1995) with original
ethical approval and informed consent waived for anonymised retrospective
records as reported by the dataset creators. Because no identifiable individual
data were used and no new human subjects were involved, the present analysis is
exempt from institutional review board approval; this exemption and its basis
should be confirmed with the submitting institution.

## Data availability
The dataset analysed in this study is openly available from the UCI Machine
Learning Repository (https://archive.ics.uci.edu/dataset/579/myocardial+
infarction+complications). All derived analysis outputs (out-of-fold
predictions, tables, figures) are provided with the article.

## Code availability
The complete analysis pipeline (data preparation, cross-validation, models,
calibration, statistical tests, SHAP, utility analyses, tables and figures) is
released as a version-pinned, seed-controlled repository (see README.md and
requirements.txt); every number in the manuscript is regenerable from the
released code and data.

## Author contributions / funding / competing interests
[To be completed by the authors at submission.]

## Acknowledgements of prior work
We cite the dataset creators and the two prior methodological studies on the
same dataset (Guo et al., Sci Rep 2026; Makhmudov et al., Bioengineering 2025)
as reference comparators.
"""
    (SUPP_DIR / "statements.md").write_text(txt)
    log("saved statements.md")


if __name__ == "__main__":
    build_tripod()
    build_feature_dictionary()
    build_statements()
    log("SUPPLEMENTARY DONE")
