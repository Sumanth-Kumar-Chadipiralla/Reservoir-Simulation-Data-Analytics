# Oil Recovery Prediction ML Dashboard

A deployable machine learning project for predicting **oil recovery (%)** from reservoir simulation case data. The project converts the original notebook workflow into a professional repository with reproducible training, a prediction API, and an interactive dashboard.

## Project goal

Use reservoir and production parameters such as BHP, bubble point pressure, permeability, porosity, injector layer status, and production rates to predict `oilRecovery`.

## Short repo structure

```text
oil-recovery-ml-dashboard/
├── data/
│   └── raw/cases_0507.xlsx
├── notebooks/
│   └── Sumanth_Lab_08_original.html
├── src/oil_recovery/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── api/main.py
├── dashboard/app.py
├── models/
├── reports/figures/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

## ML workflow

1. Load simulation cases from Excel.
2. Remove rows where `oilRecovery <= 0` because those are non-producing/no-recovery cases.
3. Select the same core features used in the notebook.
4. One-hot encode injector layer status columns.
5. Scale numeric variables for neural-network training.
6. Train two models:
   - Random Forest Regressor
   - MLP Regressor, similar to the original ANN notebook
7. Select the best model by test R².
8. Save model, metrics, predictions, and plots.
9. Serve predictions through FastAPI.
10. View metrics and predictions in Streamlit dashboard.

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

To allow PowerShell activation without changing system policy, run as administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Train the model

```bash
python -m src.oil_recovery.train
```

This creates:

```text
models/oil_recovery_model.joblib
reports/metrics.json
reports/predictions.csv
reports/feature_importance.csv
reports/figures/actual_vs_predicted.png
reports/figures/residuals.png
reports/figures/model_comparison.png
```

## Run prediction API

```bash
uvicorn src.oil_recovery.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Example API request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "BHP_injectior": 3750,
    "bubblePointPressure": 5000,
    "injectionLayer2_status": "OPEN",
    "injection_layer3_status": "CLOSED",
    "perm1": 300,
    "perm2": 250,
    "perm3": 350,
    "perm4": 400,
    "perm_ratio": 0.18,
    "por1": 0.22,
    "por2": 0.20,
    "por3": 0.25,
    "por4": 0.18,
    "production1_rate": 6500,
    "production2_rate": 6000,
    "production3_rate": 7000
  }'
```

## Run dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard includes:

- model metrics
- actual vs predicted plot
- residual plot
- model comparison
- input form for oil recovery prediction
- uploaded CSV/XLSX batch prediction

## Deploy options

### GitHub

```bash
git init
git add .
git commit -m "Initial oil recovery ML dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/oil-recovery-ml-dashboard.git
git push -u origin main
```

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud.
3. Select the GitHub repo.
4. Set main file path:

```text
dashboard/app.py
```

5. Add `requirements.txt` from the repo.
6. Deploy.

### Render/Fly.io/Railway API deployment

Use the `Dockerfile` for API deployment.

```bash
docker build -t oil-recovery-api .
docker run -p 8000:8000 oil-recovery-api
```

## Resume/GitHub project title

**Oil Recovery Prediction Dashboard using Reservoir Simulation Data**

Good one-line description:

> Built an end-to-end ML pipeline and dashboard to predict oil recovery from reservoir simulation parameters using Python, scikit-learn, FastAPI, and Streamlit.
