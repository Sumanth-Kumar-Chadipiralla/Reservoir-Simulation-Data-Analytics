from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURE_COLUMNS, MODEL_PATH
from .data import clean_status


def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run: python -m src.oil_recovery.train"
        )
    return joblib.load(path)


def predict_one(payload: dict, model_path: Path = MODEL_PATH) -> float:
    model = load_model(model_path)
    row = {col: payload[col] for col in FEATURE_COLUMNS}
    for col in ["injectionLayer2_status", "injection_layer3_status"]:
        row[col] = clean_status(row[col])
    X = pd.DataFrame([row])
    return float(model.predict(X)[0])


def predict_batch(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    model = load_model(model_path)
    X = df[FEATURE_COLUMNS].copy()
    for col in ["injectionLayer2_status", "injection_layer3_status"]:
        X[col] = X[col].apply(clean_status)
    out = df.copy()
    out["predicted_oilRecovery"] = model.predict(X)
    return out
