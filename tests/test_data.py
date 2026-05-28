import pandas as pd

from src.oil_recovery.data import prepare_training_data
from src.oil_recovery.config import FEATURE_COLUMNS, TARGET


def test_prepare_training_data_filters_zero_recovery():
    row = {col: 1 for col in FEATURE_COLUMNS}
    row["injectionLayer2_status"] = '"OPEN"'
    row["injection_layer3_status"] = '"CLOSED"'
    df = pd.DataFrame([{**row, TARGET: 0}, {**row, TARGET: 55.0}])
    X, y = prepare_training_data(df)
    assert len(X) == 1
    assert y.iloc[0] == 55.0
    assert X.iloc[0]["injectionLayer2_status"] == "OPEN"
