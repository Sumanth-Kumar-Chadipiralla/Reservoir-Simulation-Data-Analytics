from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "cases_0507.xlsx"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_PATH = MODEL_DIR / "oil_recovery_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
PREDICTIONS_PATH = REPORT_DIR / "predictions.csv"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"

TARGET = "oilRecovery"

FEATURE_COLUMNS = [
    "BHP_injectior",
    "bubblePointPressure",
    "injectionLayer2_status",
    "injection_layer3_status",
    "perm1",
    "perm2",
    "perm3",
    "perm4",
    "perm_ratio",
    "por1",
    "por2",
    "por3",
    "por4",
    "production1_rate",
    "production2_rate",
    "production3_rate",
]

CATEGORICAL_COLUMNS = ["injectionLayer2_status", "injection_layer3_status"]
NUMERIC_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]
