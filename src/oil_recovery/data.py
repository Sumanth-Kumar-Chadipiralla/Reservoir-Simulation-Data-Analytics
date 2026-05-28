import pandas as pd

from .config import DATA_PATH, FEATURE_COLUMNS, TARGET


def clean_status(value):
    """Normalize injector status values such as '"OPEN"' into 'OPEN'."""
    if pd.isna(value):
        return "UNKNOWN"
    return str(value).replace('"', '').strip().upper()


def load_raw_data(path=DATA_PATH) -> pd.DataFrame:
    """Load raw Excel simulation data."""
    return pd.read_excel(path)


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Filter valid rows and return X/y for oil recovery prediction."""
    missing = [col for col in FEATURE_COLUMNS + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.copy()
    for col in ["injectionLayer2_status", "injection_layer3_status"]:
        data[col] = data[col].apply(clean_status)

    data = data[data[TARGET].notna()]
    data = data[data[TARGET] > 0]

    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET].astype(float)
    return X, y
