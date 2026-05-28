from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import io
import pandas as pd

from src.oil_recovery.config import FEATURE_COLUMNS, METRICS_PATH
from src.oil_recovery.predict import predict_one, predict_batch

app = FastAPI(
    title="Oil Recovery Prediction API",
    description="Predict oil recovery from reservoir simulation inputs.",
    version="1.0.0",
)


class OilRecoveryInput(BaseModel):
    BHP_injectior: float = Field(..., example=3750)
    bubblePointPressure: float = Field(..., example=5000)
    injectionLayer2_status: str = Field(..., example="OPEN")
    injection_layer3_status: str = Field(..., example="CLOSED")
    perm1: float = Field(..., example=300)
    perm2: float = Field(..., example=250)
    perm3: float = Field(..., example=350)
    perm4: float = Field(..., example=400)
    perm_ratio: float = Field(..., example=0.18)
    por1: float = Field(..., example=0.22)
    por2: float = Field(..., example=0.20)
    por3: float = Field(..., example=0.25)
    por4: float = Field(..., example=0.18)
    production1_rate: float = Field(..., example=6500)
    production2_rate: float = Field(..., example=6000)
    production3_rate: float = Field(..., example=7000)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Oil Recovery Prediction API is running"}


@app.get("/features")
def feature_schema():
    return {"required_features": FEATURE_COLUMNS}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    return METRICS_PATH.read_text()


@app.post("/predict")
def predict(payload: OilRecoveryInput):
    try:
        prediction = predict_one(payload.model_dump())
        return {"predicted_oilRecovery": prediction}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/predict-batch")
def predict_batch_file(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError("Upload a CSV or Excel file.")
        output = predict_batch(df)
        buffer = io.StringIO()
        output.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=oil_recovery_predictions.csv"},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
