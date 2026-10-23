"""s12_figures.py - all manuscript figures with a uniform journal theme.

Theme: Liberation Sans (metric-compatible with Arial), 300 dpi PNG + vector
PDF, consistent colour mapping (stages/models from config), no top/right
spines, lettered panels.

Figures:
  F01_study_design          schematic of the study pipeline
  F02_prevalence_corr       outcome prevalence + label correlation
  F03_temporal_curves       macro AUROC/AUPRC vs information stage per model
  F04_roc_grid              per-outcome ROC (primary model, 72 h)
  F05_pr_grid               per-outcome precision-recall (primary, 72 h)
  F06_calibration           reliability curves before/after + ECE bars
  F07_shap                  SHAP beeswarms, six key outcomes
  F08_dca                   decision curves, six key outcomes
  F09_subgroups             forest plot AUROC by age/sex
  F10_risk_groups           observed vs predicted by risk quintile
  F11_model_comparison      macro metrics at 72 h with CIs
  F12_ovr_vs_chain          per-outcome comparison at 72 h
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (FIGURES_DIR, FONT_DIR, TABLES_DIR, OOF_DIR, OUTCOMES,
                    OUTCOME_NAMES, STAGES, STAGE_LABELS, STAGE_COLORS,
                    MODEL_COLORS, PALETTE, FIG_DPI, SEED)
from common import load_raw, auroc, auprc, log
from sklearn.metrics import roc_curve, precision_recall_curve

# ---------------------------------------------------------------- theme ----
for _f in ["LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
           "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"]:
    p = Path(FONT_DIR) / _f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": FIG_DPI,
    "pdf.fonttype": 42,      # editable text in Illustrator
    "ps.fonttype": 42,
})

PRIMARY = "XGB"
MODELS = ["LR", "RF", "XGB", "LGBM", "CHAIN"]
MODEL_LABEL = {"LR": "Logistic regression", "RF": "Random forest",
               "XGB": "XGBoost (tuned)", "LGBM": "LightGBM",
               "CHAIN": "Classifier chain + XGB"}


def savefig(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=FIG_DPI,
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    log(f"  saved {name}")


def load_oof(model, stage):
    return np.load(OOF_DIR / f"{model}_{stage}.npz")["prob"]      # (R, n, 12)


def panel_label(ax, letter, x=-0.08, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")


# ===========================================================================
def f01_study_design():
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, fs=6.0):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                           fc=fc, ec="#444444", lw=0.7)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.45)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=8,
                                     lw=0.8, color="#444444"))

    W = 23.0
    xs = [0.5, 26.0, 51.5, 77.0]
    ws = [W, W, W, 22.5]
    # top row
    box(xs[0], 33, ws[0], 11, "UCI MI Complications\n1,700 patients\n111 features,\n12 outcomes", "#DCE6F1")
    box(xs[1], 33, ws[1], 11, "Temporal feature staging\nAdmission (102)\n\u2192 +24 h (105) \u2192 +48 h (108)\n\u2192 +72 h (111)", "#DCE6F1")
    box(xs[2], 33, ws[2], 11, "Fold-internal\npreprocessing\nmedian imputation\n+ missing indicators", "#DCE6F1")
    box(xs[3], 33, ws[3], 11, "Models\n(binary relevance)\nLR \u00b7 RF \u00b7 XGBoost \u00b7\nLightGBM \u00b7 chain", "#FDE9D9")
    arrow(xs[0]+ws[0]+0.2, 38.5, xs[1]-0.3, 38.5)
    arrow(xs[1]+ws[1]+0.2, 38.5, xs[2]-0.3, 38.5)
    arrow(xs[2]+ws[2]+0.2, 38.5, xs[3]-0.3, 38.5)
    # second row (right to left)
    box(xs[3], 17.5, ws[3], 11, "Repeated stratified CV\n5 folds \u00d7 5 repeats\n(iterative stratification)", "#EBF1DE")
    box(xs[2], 17.5, ws[2], 11, "Discrimination\nAUROC \u00b7 AUPRC \u00b7 F1\nDeLong \u00b7 paired bootstrap\nHolm adjustment", "#EBF1DE")
    box(xs[1], 17.5, ws[1], 11, "Calibration\npost-hoc sigmoid / isotonic\nBrier \u00b7 ECE \u00b7 ICI\nreliability curves", "#EBF1DE")
    box(xs[0], 17.5, ws[0], 11, "Clinical utility\nthreshold policies\ndecision-curve analysis\nrisk-quintile stratification", "#EBF1DE")
    arrow(xs[3]+ws[3]/2, 32.8, xs[3]+ws[3]/2, 28.8)
    arrow(xs[3]-0.2, 23, xs[2]+ws[2]+0.3, 23)
    arrow(xs[2]-0.2, 23, xs[1]+ws[1]+0.3, 23)
    arrow(xs[1]-0.2, 23, xs[0]+ws[0]+0.3, 23)
    # interpretability
    box(26.0, 1.5, 47.0, 10, "Interpretability: SHAP global top predictors\nper outcome and local explanations \u00b7\nsubgroup performance by age and sex", "#E4DFEC")
    arrow(62.5, 17.3, 60.0, 11.8)
    arrow(37.0, 17.3, 39.5, 11.8)
    savefig(fig, "F01_study_design")

# ===========================================================================
def f02_prevalence_corr(y):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    prev = y[OUTCOMES].mean().sort_values()
    names = [OUTCOME_NAMES[i] for i in prev.index]
    cols = ["#C44E52" if i == "DEATH" else "#4C72B0" for i in prev.index]
    a1.barh(range(len(prev)), prev.values * 100, color=cols, height=0.68)
    a1.set_yticks(range(len(prev))); a1.set_yticklabels(names)
    for i, v in enumerate(prev.values * 100):
        a1.text(v + 0.35, i, f"{v:.1f}%", va="center", fontsize=6.5)
    a1.set_xlabel("Prevalence (%)")
    a1.set_title("a  Outcome prevalence (N = 1700)")
    a1.set_xlim(0, 27)

    C = y[OUTCOMES].corr()
    im = a2.imshow(C.values, cmap="coolwarm", vmin=-0.15, vmax=0.30)
    a2.set_xticks(range(12)); a2.set_yticks(range(12))
    a2.set_xticklabels(OUTCOMES, rotation=90, fontsize=5.6)
    a2.set_yticklabels(OUTCOMES, fontsize=5.6)
    a2.grid(False)
    for i in range(12):
        for j in range(12):
            v = C.values[i, j]
            if abs(v) >= 0.08 and i != j:
                a2.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center",
                        va="center", fontsize=5,
                        color="black")
    a2.set_title("b  Label correlation")
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, "F02_prevalence_corr")


# ===========================================================================
def f03_temporal(Y):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for ax, (met, mfun, letter) in zip(axes, [
            ("AUROC", auroc, "a"), ("AUPRC", auprc, "b")]):
        for m in MODELS:
            means, sds = [], []
            for s in STAGES:
                pr = load_oof(m, s)          # (R,n,12)
                per_rep = []
                for r in range(pr.shape[0]):
                    vals = [mfun(Y[:, j], pr[r, :, j]) for j in range(12)]
                    per_rep.append(np.nanmean(vals))
                means.append(np.mean(per_rep)); sds.append(np.std(per_rep))
            means = np.array(means); sds = np.array(sds)
            xs = np.arange(4)
            lw = 1.8 if m == PRIMARY else 1.0
            ax.plot(xs, means, "-o", color=MODEL_COLORS[m], ms=3.2, lw=lw,
                    label=MODEL_LABEL[m],
                    zorder=3 if m == PRIMARY else 2)
            ax.fill_between(xs, means - sds, means + sds,
                            color=MODEL_COLORS[m], alpha=0.14, lw=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Admission", "+24 h", "+48 h", "+72 h"])
        ax.set_xlabel("Cumulative information stage")
        ax.set_ylabel(f"Macro {met} (12 outcomes)")
        ax.set_title(f"{letter}  Macro {met} across information stages")
        ax.set_xlim(-0.15, 3.15)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=MODEL_COLORS[m], lw=1.4, marker="o",
                      ms=3, label=MODEL_LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.055), columnspacing=1.4,
               handlelength=1.8)
    fig.tight_layout()
    savefig(fig, "F03_temporal_curves")


# ===========================================================================
def f04_f05_roc_pr(Y, kind="roc"):
    P = load_oof(PRIMARY, "72hr").mean(axis=0)
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6))
    for j, o in enumerate(OUTCOMES):
        ax = axes[j // 4, j % 4]
        if kind == "roc":
            fpr, tpr, _ = roc_curve(Y[:, j], P[:, j])
            ax.plot(fpr, tpr, color=STAGE_COLORS["72hr"], lw=1.4)
            ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#888888")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            a = auroc(Y[:, j], P[:, j])
            ax.set_title(f"{OUTCOME_NAMES[o]}\nAUROC = {a:.3f}", fontsize=7)
            ax.set_xlabel("1 \u2212 Specificity"); ax.set_ylabel("Sensitivity")
            ax.set_aspect("equal")
        else:
            rec, prec, _ = precision_recall_curve(Y[:, j], P[:, j])
            ax.plot(rec, prec, color="#C44E52", lw=1.4)
            prev = Y[:, j].mean()
            ax.axhline(prev, ls="--", lw=0.8, color="#888888")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ap = auprc(Y[:, j], P[:, j])
            ax.set_title(f"{OUTCOME_NAMES[o]}\nAUPRC = {ap:.3f} (prev {prev:.2f})",
                         fontsize=7)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.tick_params(labelsize=6)
    fig.suptitle("")
    fig.tight_layout()
    savefig(fig, f"F0{4 if kind=='roc' else 5}_{'roc' if kind=='roc' else 'pr'}_grid")


# ===========================================================================
def f06_calibration(Y):
    cal = pd.read_csv(TABLES_DIR / "calibration.csv")
    rel = pd.read_csv(TABLES_DIR / "reliability_bins.csv")
    outs = ["DEATH", "ZSN", "OTEK_LANC", "REC_IM", "FIBR_PREDS", "P_IM_STEN"]
    fig = plt.figure(figsize=(7.2, 5.3))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.95, wspace=0.35,
                           width_ratios=[1, 1, 1, 1.1])
    for k, o in enumerate(outs):
        r, c = k // 3, k % 3
        ax = fig.add_subplot(gs[r, c])
        for vname, col, mk in [("uncalibrated", "#C44E52", "o"),
                               ("best", "#4C72B0", "s")]:
            g = rel[(rel.outcome == o) & (rel.stage == "72hr") &
                    (rel.variant == vname)]
            ax.plot(g.predicted, g.observed, mk + "-", ms=3, lw=1.0,
                    color=col, label="Uncalibrated" if vname == "uncalibrated"
                    else "Post-hoc calibrated")
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#666666")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{OUTCOME_NAMES[o]}", fontsize=7.5)
        ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed frequency")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    from matplotlib.lines import Line2D
    hnd = [Line2D([0], [0], color="#C44E52", marker="o", ms=3.2, lw=1.0,
                  label="Uncalibrated"),
           Line2D([0], [0], color="#4C72B0", marker="s", ms=3.2, lw=1.0,
                  label="Post-hoc calibrated")]
    fig.legend(handles=hnd, loc="center left", bbox_to_anchor=(0.03, 0.505),
               ncol=1, frameon=False, fontsize=7, handlelength=2.2,
               labelspacing=0.6)
    # ECE bar panel (right column, spanning both rows) - honest LORO values
    ax = fig.add_subplot(gs[:, 3])
    loro = pd.read_csv(TABLES_DIR / "calibration_loro.csv").set_index("outcome")
    piv = loro.loc[OUTCOMES[::-1], ["ECE_unc", "ECE_cal"]]
    yy = np.arange(len(piv))
    ax.barh(yy + 0.2, piv["ECE_unc"], height=0.38, color="#C44E52",
            label="Uncalibrated")
    ax.barh(yy - 0.2, piv["ECE_cal"], height=0.38, color="#4C72B0",
            label="Calibrated (held-out repeat)")
    ax.set_yticks(yy)
    ax.set_yticklabels([OUTCOME_NAMES[o] for o in piv.index], fontsize=6)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlabel("Expected calibration error")
    ax.set_title("ECE, all 12 outcomes (72 h)", fontsize=7.5)
    savefig(fig, "F06_calibration")


# ===========================================================================
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
    return f"Missing: {name}" if miss else name


def f07_shap():
    z = np.load(OOF_DIR / "shap_values.npz")
    vals = z["values"]
    zx = np.load(OOF_DIR / "shap_X.npz", allow_pickle=True)
    Xm = zx["X"]; names = [str(s) for s in zx["names"]]
    keys = ["DEATH", "ZSN", "OTEK_LANC", "FIBR_JELUD", "RAZRIV", "REC_IM"]
    cmap = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.8))
    rng = np.random.default_rng(SEED)
    for k, o in enumerate(keys):
        j = OUTCOMES.index(o)
        sv = vals[j]
        mabs = np.abs(sv).mean(axis=0)
        order = np.argsort(mabs)[::-1][:10]
        ax = axes[k // 3, k % 3]
        for yi, fi in enumerate(order[::-1]):
            v = sv[:, fi]
            feat = Xm[:, fi].astype(float)
            fmin, fmax = np.nanmin(feat), np.nanmax(feat)
            if fmax - fmin > 1e-12:
                cvals = (feat - fmin) / (fmax - fmin)
            else:
                cvals = np.full(len(feat), 0.5)
            cvals = np.nan_to_num(cvals, nan=0.5)
            jit = rng.normal(0, 0.11, len(v))
            ax.scatter(v, yi + jit, s=2.0, c=cvals, cmap=cmap, vmin=0,
                       vmax=1, linewidths=0, rasterized=True)
        ax.axvline(0, color="#888888", lw=0.6)
        ax.set_yticks(range(10))
        ax.set_yticklabels([clinical_feat(clean_feat(names[i])) for i in order[::-1]],
                           fontsize=5.6)
        ttl = OUTCOME_NAMES[o]
        if len(ttl) > 24:
            words = ttl.split()
            mid = int(np.ceil(len(words) / 2))
            ttl = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        ax.set_title(ttl, fontsize=7.5)
        ax.set_xlabel("SHAP value (log-odds)", fontsize=6.5)
        ax.grid(False)
        ax.tick_params(labelsize=6)
    fig.subplots_adjust(right=0.895, hspace=0.5, wspace=0.62)
    cax = fig.add_axes([0.912, 0.16, 0.012, 0.68])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"])
    cb.ax.tick_params(labelsize=6)
    cax.set_title("Feature\nvalue", fontsize=6.5)
    savefig(fig, "F07_shap_beeswarm")


# ===========================================================================
def f08_dca():
    d = pd.read_csv(TABLES_DIR / "dca_curves.csv")
    keys = ["DEATH", "ZSN", "OTEK_LANC", "REC_IM", "RAZRIV", "FIBR_PREDS"]
    ymax = {"DEATH": 0.16, "ZSN": 0.16, "OTEK_LANC": 0.07, "REC_IM": 0.06,
            "RAZRIV": 0.03, "FIBR_PREDS": 0.08}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for k, o in enumerate(keys):
        ax = axes[k // 3, k % 3]
        g = d[d.outcome == o]
        for mname, col, ls, lab in [
                ("calibrated_72h", "#4C72B0", "-", "Model (72 h, calibrated)"),
                ("calibrated_admission", "#55A868", "--", "Model (admission)"),
                ("treat_all", "#C44E52", ":", "Treat all")]:
            gg = g[g.model == mname].sort_values("threshold")
            ax.plot(gg.threshold, gg.net_benefit, ls, color=col, lw=1.3,
                    label=lab)
        ax.axhline(0, color="#666666", lw=0.8)
        ax.set_xlim(0, 0.6); ax.set_ylim(min(-0.02, -ymax[o] * 0.3), ymax[o])
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"{OUTCOME_NAMES[o]}", fontsize=7.5)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout()
    savefig(fig, "F08_decision_curves")


# ===========================================================================
def f09_subgroups():
    sg = pd.read_csv(TABLES_DIR / "subgroups.csv")
    keys = ["DEATH", "ZSN", "OTEK_LANC", "FIBR_JELUD", "REC_IM", "FIBR_PREDS"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    yy = 0; ticks = []; ticklabs = []
    colors = {"<65": "#4C72B0", ">=65": "#8172B2",
              "Female": "#C44E52", "Male": "#55A868"}
    for o in keys:
        for (sn, gn) in [("Age", "<65"), ("Age", ">=65"),
                         ("Sex", "Female"), ("Sex", "Male")]:
            r = sg[(sg.subgroup == sn) & (sg.group == gn) & (sg.outcome == o)]
            if not len(r) or pd.isna(r.AUROC.iloc[0]):
                continue
            r = r.iloc[0]
            lab = f"{gn}" if sn == "Age" else f"{gn}"
            ax.errorbar(r.AUROC, yy,
                        xerr=[[r.AUROC - r.AUROC_lo], [r.AUROC_hi - r.AUROC]],
                        fmt="o", ms=3.2, lw=1.0, capsize=2,
                        color=colors[gn])
            ticks.append(yy); ticklabs.append(f"{OUTCOME_NAMES[o]} \u2013 {gn}")
            yy += 1
        yy += 0.7
    ax.axvline(0.5, ls="--", lw=0.8, color="#888888")
    ax.set_yticks(ticks); ax.set_yticklabels(ticklabs, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC (95% CI), out-of-fold, 72 h model")
    ax.set_xlim(0.15, 1.02)
    ax.set_title("Subgroup discrimination by age and sex")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", ms=4, color=c, label=l)
               for l, c in [("Age < 65", "#4C72B0"), ("Age \u2265 65", "#8172B2"),
                            ("Female", "#C44E52"), ("Male", "#55A868")]]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    savefig(fig, "F09_subgroup_forest")


# ===========================================================================
def f10_risk_groups():
    rg = pd.read_csv(TABLES_DIR / "risk_groups.csv")
    keys = ["DEATH", "ZSN", "OTEK_LANC"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, o in zip(axes, keys):
        g = rg[rg.outcome == o]
        x = np.arange(1, 6)
        ax.bar(x - 0.19, g.observed_rate, width=0.38, color="#4C72B0",
               label="Observed rate")
        ax.errorbar(x - 0.19, g.observed_rate,
                    yerr=[g.observed_rate - g.obs_lo, g.obs_hi - g.observed_rate],
                    fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2)
        ax.bar(x + 0.19, g.mean_predicted, width=0.38, color="#DD8452",
               label="Mean predicted risk")
        ax.set_xticks(x); ax.set_xticklabels([f"Q{i}" for i in x])
        ax.set_xlabel("Quintile of calibrated predicted risk")
        ax.set_ylabel("Rate")
        ax.set_title(OUTCOME_NAMES[o], fontsize=7.5)
        ax.set_ylim(0, max(0.75, g.observed_rate.max() * 1.15))
        ax.tick_params(labelsize=6)
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    savefig(fig, "F10_risk_stratification")


# ===========================================================================
def f11_model_comparison():
    om = pd.read_csv(TABLES_DIR / "overall_metrics.csv")
    om = om[om.stage == "72hr"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, met, letter in [(axes[0], "AUROC", "a"), (axes[1], "AUPRC", "b")]:
        means = om.set_index("model")[f"macro_{met}"].reindex(MODELS)
        los = om.set_index("model")[f"{met}_lo"].reindex(MODELS)
        his = om.set_index("model")[f"{met}_hi"].reindex(MODELS)
        xs = np.arange(len(MODELS))
        ax.bar(xs, means.values, width=0.62,
               color=[MODEL_COLORS[m] for m in MODELS])
        ax.errorbar(xs, means.values,
                    yerr=[means.values - los.values, his.values - means.values],
                    fmt="none", ecolor="#222222", elinewidth=0.9, capsize=2.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_LABEL[m].replace(" (tuned)", "").replace(" + XGB", "+XGB")
                            for m in MODELS], rotation=18, ha="right", fontsize=6.5)
        ax.set_ylabel(f"Macro {met} (95% CI)")
        ax.set_title(f"{letter}  {met}, full-information stage (72 h)")
        lo_lim = 0.45 if met == "AUROC" else 0.12
        ax.set_ylim(lo_lim, (0.85 if met == "AUROC" else 0.32))
    fig.tight_layout()
    savefig(fig, "F11_model_comparison")


# ===========================================================================
def f12_ovr_vs_chain(Y):
    ovr = load_oof(PRIMARY, "72hr").mean(axis=0)
    ch = load_oof("CHAIN", "72hr").mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    xs = np.arange(12)
    for ax, mfun, met in [(axes[0], auroc, "AUROC"), (axes[1], auprc, "AUPRC")]:
        a = [mfun(Y[:, j], ovr[:, j]) for j in range(12)]
        b = [mfun(Y[:, j], ch[:, j]) for j in range(12)]
        ax.bar(xs - 0.19, a, width=0.38, color=MODEL_COLORS["XGB"],
               label="Binary relevance (OvR)")
        ax.bar(xs + 0.19, b, width=0.38, color=MODEL_COLORS["CHAIN"],
               label="Classifier chain (10 orders)")
        ax.set_xticks(xs)
        ax.set_xticklabels(OUTCOMES, rotation=60, ha="right", fontsize=5.6)
        ax.set_ylabel(f"Macro-per-outcome {met}")
        ax.set_title(f"{met} by outcome, 72 h", fontsize=8)
        ax.legend(frameon=False, fontsize=6)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    savefig(fig, "F12_ovr_vs_chain")


def main():
    X, y, variables, y_raw = load_raw()
    Y = y.values.astype(int)
    log("F01"); f01_study_design()
    log("F02"); f02_prevalence_corr(y)
    log("F03"); f03_temporal(Y)
    log("F04"); f04_f05_roc_pr(Y, "roc")
    log("F05"); f04_f05_roc_pr(Y, "pr")
    log("F06"); f06_calibration(Y)
    log("F07"); f07_shap()
    log("F08"); f08_dca()
    log("F09"); f09_subgroups()
    log("F10"); f10_risk_groups()
    log("F11"); f11_model_comparison()
    log("F12"); f12_ovr_vs_chain(Y)
    log("FIGURES DONE")


if __name__ == "__main__":
    main()
