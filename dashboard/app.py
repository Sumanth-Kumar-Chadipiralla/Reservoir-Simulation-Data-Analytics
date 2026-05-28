import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.oil_recovery.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
)
from src.oil_recovery.predict import predict_one, predict_batch

st.set_page_config(page_title="Oil Recovery ML Dashboard", layout="wide")

st.title("Oil Recovery Prediction Dashboard")
st.caption("Reservoir simulation ML project with training results, prediction API-ready inputs, and batch scoring.")

if not METRICS_PATH.exists():
    st.warning("Model artifacts are missing. Run `python -m src.oil_recovery.train` first.")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text())
best_model = metrics["best_model"]
best_test = metrics["models"][best_model]["test"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", best_model.replace("_", " ").title())
col2.metric("Test R²", f"{best_test['r2']:.3f}")
col3.metric("Test MAE", f"{best_test['mae']:.2f}")
col4.metric("Rows Used", metrics["rows_used_after_filter"])

st.subheader("Model comparison")
model_rows = []
for model_name, values in metrics["models"].items():
    row = {"model": model_name}
    for split in ["train", "test"]:
        for key, value in values[split].items():
            row[f"{split}_{key}"] = value
    model_rows.append(row)
model_df = pd.DataFrame(model_rows)
st.dataframe(model_df, use_container_width=True)
st.plotly_chart(px.bar(model_df, x="model", y="test_r2", title="Test R² by Model"), use_container_width=True)

if PREDICTIONS_PATH.exists():
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    st.subheader("Prediction diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.scatter(
                pred_df,
                x="actual_oilRecovery",
                y="predicted_oilRecovery",
                trendline="ols",
                title="Actual vs Predicted Oil Recovery",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.scatter(
                pred_df,
                x="predicted_oilRecovery",
                y="residual",
                title="Residuals vs Predicted Oil Recovery",
            ),
            use_container_width=True,
        )
    with st.expander("View test predictions"):
        st.dataframe(pred_df, use_container_width=True)

if FEATURE_IMPORTANCE_PATH.exists():
    fi = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    if not fi.empty:
        st.subheader("Top feature importance")
        st.plotly_chart(px.bar(fi.head(15), x="importance", y="feature", orientation="h"), use_container_width=True)

st.subheader("Single prediction")
with st.form("single_prediction"):
    c1, c2, c3 = st.columns(3)
    values = {}
    with c1:
        values["BHP_injectior"] = st.number_input("BHP injector", value=3750.0)
        values["bubblePointPressure"] = st.number_input("Bubble point pressure", value=5000.0)
        values["injectionLayer2_status"] = st.selectbox("Injection layer 2", ["OPEN", "CLOSED"])
        values["injection_layer3_status"] = st.selectbox("Injection layer 3", ["OPEN", "CLOSED"])
        values["perm_ratio"] = st.number_input("Permeability ratio", value=0.18, format="%.4f")
    with c2:
        values["perm1"] = st.number_input("Perm 1", value=300.0)
        values["perm2"] = st.number_input("Perm 2", value=250.0)
        values["perm3"] = st.number_input("Perm 3", value=350.0)
        values["perm4"] = st.number_input("Perm 4", value=400.0)
    with c3:
        values["por1"] = st.number_input("Porosity 1", value=0.22, format="%.4f")
        values["por2"] = st.number_input("Porosity 2", value=0.20, format="%.4f")
        values["por3"] = st.number_input("Porosity 3", value=0.25, format="%.4f")
        values["por4"] = st.number_input("Porosity 4", value=0.18, format="%.4f")
        values["production1_rate"] = st.number_input("Production 1 rate", value=6500.0)
        values["production2_rate"] = st.number_input("Production 2 rate", value=6000.0)
        values["production3_rate"] = st.number_input("Production 3 rate", value=7000.0)

    submitted = st.form_submit_button("Predict oil recovery")
    if submitted:
        prediction = predict_one(values)
        st.success(f"Predicted oil recovery: {prediction:.2f}%")

st.subheader("Batch prediction")
upload = st.file_uploader("Upload CSV or Excel with required feature columns", type=["csv", "xlsx", "xls"])
if upload:
    if upload.name.lower().endswith(".csv"):
        batch_df = pd.read_csv(upload)
    else:
        batch_df = pd.read_excel(upload)
    try:
        scored = predict_batch(batch_df)
        st.dataframe(scored, use_container_width=True)
        st.download_button(
            "Download scored predictions",
            scored.to_csv(index=False),
            file_name="oil_recovery_predictions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(str(exc))

with st.expander("Raw data preview"):
    st.dataframe(pd.read_excel(DATA_PATH).head(50), use_container_width=True)
