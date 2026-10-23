"""s11_tables.py - build manuscript-ready tables.

Reads all analysis CSVs and produces:
  results/tables/final/T01..T13.csv        (clean, rounded, publication form)
  results/Manuscript_Tables.docx           (Tables 1-13 with titles/footnotes)
  results/Supplementary_Tables.docx        (S1-S6)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (TABLES_DIR, OUTCOMES, OUTCOME_NAMES, STAGES, STAGE_LABELS,
                    HIGH_RISK_OUTCOMES, SEED, N_SPLITS, N_REPEATS)
from common import load_raw, auroc, auprc, log

FINAL = TABLES_DIR / "final"
FINAL.mkdir(exist_ok=True)

MODELS = ["LR", "RF", "XGB", "LGBM", "CHAIN"]
MINUS = "\u2212"
MODEL_NAMES = {"LR": "Logistic regression", "RF": "Random forest",
               "XGB": "XGBoost (tuned)", "LGBM": "LightGBM",
               "CHAIN": "Classifier chain + XGBoost"}


def f3(x):
    return "" if pd.isna(x) else f"{x:.3f}"


def fp(p):
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ci(v, lo, hi, nd=3):
    if pd.isna(v):
        return ""
    if pd.isna(lo) or pd.isna(hi):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} ({lo:.{nd}f}\u2013{hi:.{nd}f})"


# ---------------------------------------------------------------------------
def t01_cohort():
    df = pd.read_csv(TABLES_DIR / "table1_cohort.csv")
    df = df[["variable", "overall", "survivors", "non_survivors",
             "missing_pct", "p_value"]]
    df.columns = ["Characteristic", f"All patients (N=1700)",
                  "Survivors (n=1429)", "Non-survivors (n=271)",
                  "Missing (%)", "p-value"]
    return df


def t02_prevalence_epv():
    epv = pd.read_csv(TABLES_DIR / "epv.csv")
    epv["Outcome"] = epv.outcome.map(OUTCOME_NAMES)
    out = epv[["outcome", "Outcome", "events", "prevalence_pct",
               "n_features_admission", "EPV_admission", "n_features_72hr",
               "EPV_72hr"]].copy()
    out.columns = ["Code", "Outcome", "Events (n)", "Prevalence (%)",
                   "Features at admission (n)", "EPV admission",
                   "Features at 72 h (n)", "EPV 72 h"]
    return out


def t03_overall():
    om = pd.read_csv(TABLES_DIR / "overall_metrics.csv")
    rows = []
    for m in MODELS:
        for s in STAGES:
            r = om[(om.model == m) & (om.stage == s)].iloc[0]
            rows.append({
                "Model": MODEL_NAMES[m], "Information stage": STAGE_LABELS[s],
                "Macro AUROC": ci(r.macro_AUROC,
                                  r.get("AUROC_lo", np.nan), r.get("AUROC_hi", np.nan)),
                "Macro AUPRC": ci(r.macro_AUPRC,
                                  r.get("AUPRC_lo", np.nan), r.get("AUPRC_hi", np.nan)),
                "Micro F1": f3(r.micro_F1), "Macro F1": f3(r.macro_F1),
                "Weighted F1": f3(r.weighted_F1)})
    return pd.DataFrame(rows)


def t04_primary_per_outcome():
    raw = pd.read_csv(TABLES_DIR / "raw_metrics.csv")
    cim = pd.read_csv(TABLES_DIR / "ci_metrics.csv")
    rows = []
    for s_ in STAGES:
        for o in OUTCOMES:
            r = raw[(raw.model == "XGB") & (raw.stage == s_) & (raw.outcome == o)].iloc[0]
            ca = cim[(cim.model == "XGB") & (cim.stage == s_) & (cim.outcome == o) & (cim.metric == "AUROC")]
            cp = cim[(cim.model == "XGB") & (cim.stage == s_) & (cim.outcome == o) & (cim.metric == "AUPRC")]
            rows.append({
                "Stage": STAGE_LABELS[s_], "Outcome": OUTCOME_NAMES[o],
                "AUROC": ci(r.AUROC, ca.lo.iloc[0] if len(ca) else np.nan,
                            ca.hi.iloc[0] if len(ca) else np.nan),
                "AUPRC": ci(r.AUPRC, cp.lo.iloc[0] if len(cp) else np.nan,
                            cp.hi.iloc[0] if len(cp) else np.nan)})
    return pd.DataFrame(rows)


def t05_temporal():
    tt = pd.read_csv(TABLES_DIR / "temporal_tests.csv")
    rows = []
    for comp in ["Admission+24h vs Admission", "Admission+24h+48h vs Admission",
                 "Admission+24h+48h+72h vs Admission"]:
        g = tt[tt.comparison == comp]
        for _, r in g.iterrows():
            rows.append({
                "Comparison": comp, "Outcome": OUTCOME_NAMES[r.outcome],
                "AUROC earlier": f3(r.AUROC_base), "AUROC later": f3(r.AUROC_new),
                "\u0394AUROC": f3(r.delta_AUROC), "p (DeLong)": fp(r.p_DeLong),
                "p (Holm)": fp(r.p_DeLong_holm),
                "\u0394AUPRC": f3(r.delta_AUPRC),
                "\u0394AUPRC 95% CI": "" if pd.isna(r.dAUPRC_lo) else
                    f"{r.dAUPRC_lo:.3f}\u2013{r.dAUPRC_hi:.3f}",
                "p (AUPRC, Holm)": fp(r.p_AUPRC_holm)})
    return pd.DataFrame(rows)


def t06_model_tests():
    om = pd.read_csv(TABLES_DIR / "overall_metrics.csv")
    o72 = om[om.stage == "72hr"].set_index("model")
    mt = pd.read_csv(TABLES_DIR / "model_tests.csv")
    rows = []
    for m in MODELS:
        r = o72.loc[m]
        n_sig = mt[(mt.model == m) & (mt.p_DeLong_holm < 0.05)]
        rows.append({
            "Model": MODEL_NAMES[m],
            "Macro AUROC (95% CI)": ci(r.macro_AUROC, r.get("AUROC_lo", np.nan), r.get("AUROC_hi", np.nan)),
            "Macro AUPRC (95% CI)": ci(r.macro_AUPRC, r.get("AUPRC_lo", np.nan), r.get("AUPRC_hi", np.nan)),
            "Micro F1": f3(r.micro_F1), "Macro F1": f3(r.macro_F1),
            "Outcomes with AUROC \u2260 XGBoost (Holm p<0.05)":
                ", ".join("{} ({})".format(
                    OUTCOME_NAMES[o],
                    "+" if mt[(mt.model == m) & (mt.outcome == o)].delta_AUROC.iloc[0] > 0 else MINUS)
                    for o in n_sig.outcome.tolist()) or "none"})
    return pd.DataFrame(rows)


def t07_calibration():
    cal = pd.read_csv(TABLES_DIR / "calibration_loro.csv")
    full = pd.read_csv(TABLES_DIR / "calibration.csv")
    full72 = full[full.stage == "72hr"].pivot_table(
        index="outcome", columns="variant",
        values=["cal_slope", "cal_intercept"])
    loro = cal.set_index("outcome")
    rows = []
    for o in OUTCOMES:
        lr_ = loro.loc[o]
        meth = str(lr_["method"])
        sl = full72.loc[o, ("cal_slope", meth)] if o in full72.index else np.nan
        ic = full72.loc[o, ("cal_intercept", meth)] if o in full72.index else np.nan
        rows.append({
            "Outcome": OUTCOME_NAMES[o],
            "Brier uncal.": f3(lr_.Brier_unc), "Brier cal.": f3(lr_.Brier_cal),
            "ECE uncal.": f3(lr_.ECE_unc),
            "ECE cal. (SD)": f"{lr_.ECE_cal:.3f} ({lr_.ECE_cal_sd:.3f})",
            "Slope cal.": f3(sl), "Intercept cal.": f3(ic),
            "Method": meth.capitalize()})
    return pd.DataFrame(rows)


def t08_thresholds():
    thr = pd.read_csv(TABLES_DIR / "thresholds_loro.csv")
    cls = pd.read_csv(TABLES_DIR / "cls_loro.csv")
    rows = []
    for o in OUTCOMES:
        t = thr[thr.outcome == o].iloc[0]
        c = cls[cls.outcome == o].iloc[0]
        rows.append({
            "Outcome": OUTCOME_NAMES[o],
            "Risk tier": "High-lethality" if o in HIGH_RISK_OUTCOMES else "Standard",
            "Policy": "Recall\u226590%" if t.policy == "clinical_recall90" else "F1-max",
            "Threshold": f"{t.threshold_mean:.3f} ({t.threshold_sd:.3f})",
            "Sensitivity": f"{c.Sensitivity:.3f} ({c.Sensitivity_sd:.3f})",
            "Specificity": f"{c.Specificity:.3f} ({c.Specificity_sd:.3f})",
            "PPV": f"{c.PPV:.3f} ({c.PPV_sd:.3f})",
            "NPV": f"{c.NPV:.3f} ({c.NPV_sd:.3f})",
            "F1": f"{c.F1:.3f} ({c.F1_sd:.3f})"})
    return pd.DataFrame(rows)


def clean_feat(nm):
    for p in ["continuous__", "binary__", "ordinal__", "partially_ordered__"]:
        nm = nm.replace(p, "")
    return nm.replace("missingindicator_", "missing:")


CLINICAL_NAMES = {
    "AGE": "Age (years)", "L_BLOOD": "Leukocyte count", "ROE": "ESR",
    "K_BLOOD": "Serum potassium", "NA_BLOOD": "Serum sodium",
    "AST_BLOOD": "AST", "ALT_BLOOD": "ALT", "NITR_S": "IV nitrates (ICU)",
    "LID_S_n": "Lidocaine (ICU)", "GEPAR_S_n": "Heparin (ICU)",
    "ANT_CA_S_n": "Anticoagulants (ICU)", "ASP_S_n": "Aspirin (ICU)",
    "B_BLOK_S_n": "Beta-blockers (ICU)", "TRENT_S_n": "Pentoxifylline (ICU)",
    "ZSN_A_1.0": "Baseline heart-failure class >=1", "FK_STENOK": "Angina functional class",
    "STENOK_AN": "Exertional angina (anamnesis)", "INF_ANAM": "Prior MI count",
    "IBS_POST": "CHD in recent weeks", "DLIT_AG": "Angina duration",
    "TIME_B_S": "Onset-to-admission time (code)", "S_AD_ORIT": "Systolic BP at admission",
    "D_AD_ORIT": "Diastolic BP at admission", "NA_R_1_n": "Opioid doses, day 1",
    "NA_R_2_n": "Opioid doses, day 2", "NA_R_3_n": "Opioid doses, day 3",
    "R_AB_2_n": "Anginal attacks, day 2", "R_AB_3_n": "Anginal attacks, day 3",
    "NOT_NA_2_n": "NSAID doses, day 2", "NOT_NA_3_n": "NSAID doses, day 3",
    "ant_im": "Anterior MI location", "lat_im": "Lateral MI location",
    "inf_im": "Inferior MI location",
}


def clinical_feat(nm):
    miss = nm.startswith("missing:")
    core = nm[8:] if miss else nm
    name = CLINICAL_NAMES.get(core, core)
    return f"Missing record: {name}" if miss else name


def t09_shap():
    s = pd.read_csv(TABLES_DIR / "shap_top5.csv")
    s = s.copy()
    s["feature"] = s.feature.apply(lambda n: clinical_feat(clean_feat(n)))
    rows = []
    for o in OUTCOMES:
        g = s[s.outcome == o].sort_values("rank")
        row = {"Outcome": OUTCOME_NAMES[o]}
        for _, r in g.iterrows():
            row[f"#{int(r['rank'])}"] = f"{r['feature']} ({r['mean_abs_shap']:.3f})"
        rows.append(row)
    return pd.DataFrame(rows)


def t10_subgroups():
    sg = pd.read_csv(TABLES_DIR / "subgroups.csv")
    rows = []
    for _, r in sg.iterrows():
        rows.append({
            "Subgroup": r.subgroup, "Group": r.group,
            "Outcome": OUTCOME_NAMES[r.outcome] if r.outcome != "MACRO" else "Macro (12 outcomes)",
            "N": int(r.N), "Events": int(r.n_pos),
            "AUROC (95% CI)": ci(r.AUROC, r.AUROC_lo, r.AUROC_hi),
            "AUPRC (95% CI)": ci(r.AUPRC, r.AUPRC_lo, r.AUPRC_hi)})
    return pd.DataFrame(rows)


def t11_utility():
    d = pd.read_csv(TABLES_DIR / "dca_summary.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append({"Outcome": OUTCOME_NAMES[r.outcome], "Threshold": f3(r.threshold),
                     "Net benefit (72 h model)": f3(r.calibrated_72h),
                     "Net benefit (admission model)": f3(r.calibrated_admission),
                     "Net benefit (treat all)": f3(r.treat_all)})
    return pd.DataFrame(rows)


def t12_rq3():
    r = pd.read_csv(TABLES_DIR / "rq3_comparison.csv")
    rows = []
    for _, x in r.iterrows():
        rows.append({
            "Outcome": OUTCOME_NAMES[x.outcome] if x.outcome != "MACRO" else "Macro (12 outcomes)",
            "AUROC base": f3(x.AUROC_base), "AUROC +course features": f3(x.AUROC_rq3),
            "\u0394AUROC": f3(x.delta_AUROC), "p (DeLong)": fp(x.p_DeLong),
            "\u0394AUPRC": f3(x.delta_AUPRC)})
    return pd.DataFrame(rows)


def t13_literature():
    df = pd.read_csv(TABLES_DIR / "literature_comparison.csv")
    df = df.rename(columns={"study": "Study", "model": "Model", "task": "Task",
                            "dataset": "Dataset", "validation": "Validation",
                            "metric": "Headline metric", "value": "Value",
                            "per_outcome": "Per-outcome", "calibration": "Calibration",
                            "explainability": "Explainability",
                            "temporal_stages": "Temporal staging"})
    return df


SUPP_BUILDERS = {}


def s_imputation():
    df = pd.read_csv(TABLES_DIR / "imputation_probe.csv")
    df.columns = ["Strategy", "Macro AUROC", "Macro AUPRC", "Runtime (s)"]
    return df


def s_lethality():
    df = pd.read_csv(TABLES_DIR / "conditional_lethality.csv")
    df["Outcome"] = df.outcome.map(OUTCOME_NAMES)
    return df[["outcome", "Outcome", "n", "conditional_lethality_pct", "risk_tier"]]


def s_labelcorr():
    X, y, _, _ = load_raw()
    c = y.corr().round(3)
    c.index = [OUTCOME_NAMES[i] for i in c.index]
    c.columns = [i for i in OUTCOMES]
    return c.reset_index().rename(columns={"index": "Outcome"})


def s_full_grid():
    raw = pd.read_csv(TABLES_DIR / "raw_metrics.csv")
    cls = pd.read_csv(TABLES_DIR / "cls_loro_all.csv")
    cls = cls.rename(columns={c: c.replace("_mean", "") for c in cls.columns
                              if c.endswith("_mean")})
    m = raw.merge(cls, on=["model", "stage", "outcome"])
    m["Model"] = m.model.map(MODEL_NAMES)
    m["Stage"] = m.stage.map(STAGE_LABELS)
    m["Outcome"] = m.outcome.map(OUTCOME_NAMES)
    keep = ["Model", "Stage", "Outcome", "AUROC", "AUPRC", "F1",
            "Sensitivity", "Specificity", "PPV", "NPV"]
    return m[keep].round(4)


def s_chain_orders():
    """Chain-order variability at 72h (macro AUROC across the 10 orders)."""
    z = np.load(Path(TABLES_DIR).parent / "oof" / "CHAIN_72hr_orders.npz")
    orders = z["orders"]      # (n_orders, R, n, 12)
    X, y, _, _ = load_raw()
    Y = y.values.astype(int)
    rows = []
    for oi in range(orders.shape[0]):
        P = orders[oi].mean(axis=0)
        aurocs = [auroc(Y[:, j], P[:, j]) for j in range(12)]
        rows.append({"chain_order": oi + 1, "macro_AUROC": float(np.nanmean(aurocs))})
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "chain_order": "Mean \u00b1 SD (10 orders)",
        "macro_AUROC": f"{df.macro_AUROC.mean():.4f} \u00b1 {df.macro_AUROC.std():.4f}"}])
    return pd.concat([df, summary], ignore_index=True)


def s_sensitivity_ab():
    a = pd.read_csv(TABLES_DIR / "sens_noIND_vs_primary.csv")
    b = pd.read_csv(TABLES_DIR / "sens_noTRT_vs_primary.csv")
    a = a.assign(variant="A: no day-2/3 course missing indicators")
    b = b.assign(variant="B: no ICU treatment variables")
    ab = pd.concat([a, b], ignore_index=True)
    ab["Outcome"] = ab.outcome.map(lambda o: OUTCOME_NAMES.get(o, "Macro (12 outcomes)"))
    return ab[["variant", "stage", "Outcome", "AUROC_primary", "AUROC_variant",
               "delta", "p_DeLong"]].round(4)


def s_sensitivity_c():
    u = pd.read_csv(TABLES_DIR / "sens_unW.csv")
    uc = pd.read_csv(TABLES_DIR / "sens_unW_calibration.csv")
    u = u.copy(); u["Outcome"] = u.outcome.map(OUTCOME_NAMES)
    u = u[["outcome", "Outcome", "AUROC", "AUPRC"]].round(4)
    uc = uc.copy(); uc["Outcome"] = uc.outcome.map(OUTCOME_NAMES)
    return u.merge(uc[["Outcome", "ECE_unweighted_uncal", "ECE_unweighted_cal"]],
                   on="Outcome", how="left").round(4)


def s_model_macro():
    return pd.read_csv(TABLES_DIR / "model_tests_macro.csv").round(4)


def s_per_repeat():
    return pd.read_csv(TABLES_DIR / "per_repeat_macro_summary.csv").round(4)


def s_riskgroups():
    df = pd.read_csv(TABLES_DIR / "risk_groups.csv")
    df["Outcome"] = df.outcome.map(OUTCOME_NAMES)
    return df[["Outcome", "quintile", "n", "events", "observed_rate",
               "obs_lo", "obs_hi", "mean_predicted", "pred_range"]].round(4)


# ---------------------------------------------------------------------------
def write_docx(tables: dict, path: Path, title: str):
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(9)
    h = doc.add_heading(title, level=0)
    for run in h.runs:
        run.font.name = "Arial"
    for (num, caption, df, footnote) in tables:
        p = doc.add_paragraph()
        r = p.add_run(f"Table {num}. {caption}")
        r.bold = True
        r.font.name = "Arial"
        r.font.size = Pt(9)
        t = doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Light Grid Accent 1"
        hdr = t.rows[0].cells
        for i, cname in enumerate(df.columns):
            hdr[i].text = str(cname)
            for par in hdr[i].paragraphs:
                for run in par.runs:
                    run.bold = True
                    run.font.size = Pt(8)
                    run.font.name = "Arial"
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = "" if pd.isna(v) else str(v)
                for par in cells[i].paragraphs:
                    for run in par.runs:
                        run.font.size = Pt(8)
                        run.font.name = "Arial"
        if footnote:
            fp_ = doc.add_paragraph()
            fr = fp_.add_run(footnote)
            fr.italic = True
            fr.font.size = Pt(8)
            fr.font.name = "Arial"
        doc.add_paragraph()
    doc.save(path)
    log(f"saved {path.name}")


def main():
    builders = [
        ("1", "Baseline characteristics of the study cohort (N = 1700).", t01_cohort,
         "Continuous variables: median [IQR]; categorical: n (%). P-values compare survivors vs non-survivors (Mann-Whitney U or chi-square). In-hospital mortality defined as LET_IS > 0."),
        ("2", "Outcome prevalence and events-per-variable (EPV) by information stage.", t02_prevalence_epv,
         "EPV = events / candidate predictors at the given stage. Outcomes with EPV < 1 (supraventricular tachycardia, ventricular tachycardia) are severely underpowered; results for these outcomes are reported for completeness but flagged as exploratory."),
        ("3", "Overall performance of all models across cumulative information stages (out-of-fold, 5\u00d75-fold iterative-stratified cross-validation).", t03_overall,
         "Values are mean out-of-fold metrics over 5 repeats; 95% CI by patient-level percentile bootstrap (B = 1000). Class imbalance handled per outcome (scale_pos_weight = n_neg/n_pos for boosting; class_weight='balanced' otherwise)."),
        ("4", "Per-outcome performance of the primary model (tuned XGBoost) at each information stage.", t04_primary_per_outcome,
         "95% CI: patient-level percentile bootstrap (B = 2000). Classification metrics at pre-specified decision thresholds (Table 8)."),
        ("5", "Incremental predictive value of accumulating in-hospital information (primary model, paired tests on identical cross-validation folds).", t05_temporal,
         "\u0394AUROC tested by DeLong's test; \u0394AUPRC by paired patient-level bootstrap (B = 1000). Holm adjustment across the 12 outcomes within each comparison."),
        ("6", "Model comparison at the full-information stage (72 h).", t06_model_tests,
         "Pairwise DeLong tests vs the primary XGBoost model per outcome, Holm-adjusted; '+'/'\u2212' indicates the direction of the difference."),
        ("7", "Calibration of the primary model before and after post-hoc calibration (out-of-fold).", t07_calibration,
         "Post-hoc calibration fitted inside each training fold (CalibratedClassifierCV, inner 3-fold). Per outcome the method with lower OOF Brier score was selected (pre-specified rule). ECE: 10 equal-width bins; ICI: integrated calibration index (spline logistic smoothing)."),
        ("8", "Decision thresholds and classification performance of the primary model at 72 h.", t08_thresholds,
         "High-lethality outcomes (conditional mortality \u2265 25%: myocardial rupture, third-degree AV block, ventricular fibrillation, recurrent MI, pulmonary edema, death) use a clinical-sensitivity policy (lowest threshold achieving \u2265 90% recall); standard outcomes use the F1-maximising threshold. Thresholds were derived from out-of-fold predictions."),
        ("9", "Top-5 SHAP predictors per outcome (primary model, full-information stage, held-out 20% of the pre-specified split).", t09_shap,
         "Mean |SHAP| in log-odds units. Feature prefixes denote preprocessing blocks (continuous__, binary__, ordinal__)."),
        ("10", "Subgroup performance of the primary model (72 h) by age and sex.", t10_subgroups,
         "Out-of-fold predictions; 95% CI by patient-level bootstrap (B = 1000) where \u2265 5 events in the subgroup. Small event counts warrant cautious interpretation."),
        ("11", "Decision-curve analysis: net benefit of the calibrated model vs treat-all.", t11_utility,
         "Net benefit computed from out-of-fold calibrated probabilities (best method per outcome, 72 h stage vs admission stage)."),
        ("12", "Effect of engineered clinical-course features (pain-relapse, opioid and NSAID exposure summaries) added to the raw 72 h feature set.", t12_rq3,
         "Repeated stratified cross-validation identical to the primary protocol; DeLong tests per outcome."),
        ("13", "Positioning of this study against published work on the same dataset.", t13_literature,
         "Values as reported by the respective publications; protocols differ (composite endpoints, splits without reported CI, aggressive resampling) - see Discussion."),
    ]
    tables = []
    for num, cap, fn, note in builders:
        log(f"building Table {num}")
        df = fn()
        df.to_csv(FINAL / f"T{int(num):02d}.csv", index=False)
        tables.append((num, cap, df, note))
    write_docx(tables, Path(TABLES_DIR).parent / "Manuscript_Tables.docx",
               "Manuscript Tables - Temporal, Explainable and Calibrated Multi-Label "
               "Machine Learning for Myocardial Infarction Complications")

    supp = [
        ("S1", "Imputation-strategy probe (logistic-regression probe model, 5-fold stratified CV, 72 h features).", s_imputation,
         "Median imputation with missing indicators achieved the best macro AUPRC at negligible computational cost and was therefore used throughout."),
        ("S2", "Conditional lethality of each complication (basis of the decision-threshold policy).", s_lethality, ""),
        ("S3", "Label correlation matrix between outcomes (Phi/Pearson).", s_labelcorr, ""),
        ("S4", "Complete per-outcome results for all models and stages.", s_full_grid, ""),
        ("S5", "Classifier-chain order variability at the 72 h stage (10 random chain orders).", s_chain_orders, ""),
        ("S6", "Risk-group stratification by quintiles of calibrated predicted risk (72 h).", s_riskgroups,
         "Observed event rates with Wilson 95% CI; mean predicted risk shown for calibration-in-the-large inspection."),
        ("S7", "Subgroup performance of the primary model (72 h) by age and sex.",
         lambda: pd.read_csv(TABLES_DIR / "subgroups.csv").round(4), ""),
        ("S8", "Clinical-course engineered features added to the raw 72 h set (RQ3).",
         lambda: pd.read_csv(TABLES_DIR / "rq3_comparison.csv").round(4), ""),
        ("S9", "Sensitivity analyses A/B: removal of censoring-linked missing indicators and of ICU treatment variables (primary model, identical folds).",
         s_sensitivity_ab,
         "DeLong p-values on identical out-of-fold patients vs the pre-specified primary model."),
        ("S9b", "Sensitivity analysis C: unweighted XGBoost (72 h) - discrimination and calibration.",
         s_sensitivity_c,
         "Shows that class weighting intentionally inflates probabilities (hence the calibration step) while leaving discrimination comparable."),
        ("S10", "Model comparison at 72 h: paired macro-metric bootstrap vs the primary model, and per-repeat stability of macro metrics.",
         s_model_macro,
         "Paired patient-level bootstrap (B=1000); per-repeat means/SDs in S10b."),
        ("S10b", "Per-repeat macro metrics (mean +/- SD over 5 repeats).",
         s_per_repeat,
         "Reconciles Figure 3 (per-repeat means) with Table 3 (repeat-averaged probabilities)."),
    ]
    stables = []
    for num, cap, fn, note in supp:
        log(f"building Table {num}")
        df = fn()
        df.to_csv(FINAL / f"{num}.csv", index=False)
        stables.append((num, cap, df, note))
    write_docx(stables, Path(TABLES_DIR).parent / "Supplementary_Tables.docx",
               "Supplementary Tables")
    log("TABLES DONE")


if __name__ == "__main__":
    main()
