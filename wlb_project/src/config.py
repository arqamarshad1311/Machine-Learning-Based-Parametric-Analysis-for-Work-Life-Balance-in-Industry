"""Central configuration for the WLB project (single source of truth)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "WLB_PBL_DATASET.xlsx"      # sheet "WLB Dataset"
RAW_SHEET = "WLB Dataset"
CLEAN_PATH = ROOT / "data" / "processed" / "wlb_cleaned.csv"
OUT = ROOT / "outputs"
FIG_DIR, MODEL_DIR, REPORT_DIR, METRIC_DIR = (OUT / d for d in ("figures", "models", "reports", "metrics"))

RANDOM_STATE = 42
TARGET = "wlb_risk_category"
ID_COL = "employee_id"
CLASS_ORDER = ["Low Risk", "Medium Risk", "High Risk"]          # ordinal: 0, 1, 2
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}

CATEGORICAL = ["gender", "occupation_type"]
# Measurable behavioural parameters (low leakage risk)
BEHAVIOURAL_NUM = ["age", "avg_work_hours_per_day", "avg_rest_hours_per_day", "avg_sleep_hours_per_day",
                   "avg_exercise_hours_per_day", "commute_time_mins", "family_time_hours",
                   "social_time_hours"]
# Self-reported / outcome-like features (|r| ~ 0.4-0.8 with the label -> leakage risk)
SUBJECTIVE_NUM = ["stress_level", "job_satisfaction", "mental_health_score", "physical_health_score"]

FEATURE_SETS = {
    "behavioural_only": BEHAVIOURAL_NUM + CATEGORICAL,
    "all_features": BEHAVIOURAL_NUM + SUBJECTIVE_NUM + CATEGORICAL,
}
NUMERIC_ALL = BEHAVIOURAL_NUM + SUBJECTIVE_NUM
WORK, REST, SLEEP, EXERCISE = ("avg_work_hours_per_day", "avg_rest_hours_per_day",
                               "avg_sleep_hours_per_day", "avg_exercise_hours_per_day")
ALL_COLUMNS = [ID_COL, "gender", "occupation_type", *BEHAVIOURAL_NUM, *SUBJECTIVE_NUM, TARGET]  # 16

# Data-quality thresholds (from the Phase-II audit)
MAX_PLAUSIBLE_WORK_HOURS = 14
WORK_CAP = 14
DAY_HOURS = 24
