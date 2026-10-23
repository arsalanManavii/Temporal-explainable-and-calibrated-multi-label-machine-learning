"""Global configuration for the MI complications prediction project.

All constants, paths, seeds and model hyper-parameters live here so that the
entire pipeline is reproducible from a single source of truth.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OOF_DIR = RESULTS_DIR / "oof"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
SUPP_DIR = PROJECT_ROOT / "supplementary"
FONT_DIR = PROJECT_ROOT.parent / "assets" / "fonts"

for _d in (OOF_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR, SUPP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 10                # same seed as the original analysis
N_SPLITS = 5             # CV folds
N_REPEATS = 5            # CV repeats
N_BOOTSTRAP = 2000       # bootstrap replicates for 95% CIs
ALPHA = 0.05

# Fixed legacy split (kept ONLY for SHAP explanations / continuity)
FIXED_TEST_SIZE = 0.20

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
N_PATIENTS = 1700

# UCI column numbers (1-based, ID = col 1) NOT available at each time point.
# Source: UCI dataset documentation (columns 2-112 are features).
TIMEPOINT_EXCLUSION = {
    "admission": [93, 94, 95, 100, 101, 102, 103, 104, 105],
    "24hr":      [94, 95, 101, 102, 104, 105],
    "48hr":      [95, 102, 105],
    "72hr":      [],
}
STAGES = ["admission", "24hr", "48hr", "72hr"]
STAGE_LABELS = {
    "admission": "Admission",
    "24hr": "Admission+24h",
    "48hr": "Admission+24h+48h",
    "72hr": "Admission+24h+48h+72h",
}

# 11 binary complications + derived in-hospital mortality
OUTCOMES = [
    "FIBR_PREDS", "PREDS_TAH", "JELUD_TAH", "FIBR_JELUD", "A_V_BLOK",
    "OTEK_LANC", "RAZRIV", "DRESSLER", "ZSN", "REC_IM", "P_IM_STEN",
    "DEATH",
]
OUTCOME_NAMES = {
    "FIBR_PREDS": "Atrial fibrillation",
    "PREDS_TAH": "Supraventricular tachycardia",
    "JELUD_TAH": "Ventricular tachycardia",
    "FIBR_JELUD": "Ventricular fibrillation",
    "A_V_BLOK": "Third-degree AV block",
    "OTEK_LANC": "Pulmonary edema",
    "RAZRIV": "Myocardial rupture",
    "DRESSLER": "Dressler syndrome",
    "ZSN": "Chronic heart failure",
    "REC_IM": "Recurrent myocardial infarction",
    "P_IM_STEN": "Post-infarction angina",
    "DEATH": "In-hospital mortality",
}
# Outcomes whose occurrence is strongly lethal (conditional P(death) high);
# for these a high-sensitivity decision rule is clinically appropriate.
# Data-driven (conditional lethality P(death|complication) >= 25%, computed
# in s04c): RAZRIV 100%, A_V_BLOK 29.8%, FIBR_JELUD 29.6%, REC_IM 28.9%,
# OTEK_LANC 25.2% - identical taxonomy to Guo et al. 2026 (Base 1).
HIGH_RISK_OUTCOMES = ["DEATH", "RAZRIV", "A_V_BLOK", "FIBR_JELUD", "REC_IM", "OTEK_LANC"]

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Tuned XGBoost hyper-parameters (RandomizedSearchCV on the admission stage,
# 3-fold CV, macro-F1 objective - identical search to the original analysis).
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=1.0,
    min_child_weight=3,
    gamma=0.1,
    random_state=SEED,
    eval_metric="logloss",
    n_jobs=2,
    tree_method="hist",
)
# NOTE: scale_pos_weight is set PER OUTCOME from train-fold counts
# (n_neg / n_pos) instead of a single global value of 10.

LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=SEED,
    n_jobs=2,
    verbose=-1,
)

RF_PARAMS = dict(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=2,
)

LR_PARAMS = dict(max_iter=5000, class_weight="balanced", random_state=SEED)

# Classifier chain: number of random chain orders (averaged)
N_CHAIN_ORDERS = 10

# Clinical threshold policy: minimum recall for high-risk outcomes
CLINICAL_MIN_RECALL = 0.90

# --------------------------------------------------------------------------
# Figures - uniform journal theme (Q1 medical journal style)
# --------------------------------------------------------------------------
FONT_FAMILY = "Liberation Sans"   # metric-compatible with Arial
FIG_DPI = 300

# Consistent colour mapping used across ALL figures
STAGE_COLORS = {
    "admission": "#4C72B0",
    "24hr": "#55A868",
    "48hr": "#C44E52",
    "72hr": "#8172B2",
}
MODEL_COLORS = {
    "LR": "#9C9C9C",
    "RF": "#55A868",
    "XGB": "#4C72B0",
    "LGBM": "#DD8452",
    "CHAIN": "#8172B2",
}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
