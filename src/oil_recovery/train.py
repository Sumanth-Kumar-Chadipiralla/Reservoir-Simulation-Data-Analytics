import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .config import (
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURE_DIR,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PREDICTIONS_PATH,
)
from .data import load_raw_data, prepare_training_data
from .features import build_preprocessor

RANDOM_STATE = 101


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def make_model_candidates():
    return {
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, min_samples_leaf=2)),
            ]
        ),
        "mlp_regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(20, 20),
                        activation="relu",
                        learning_rate="adaptive",
                        max_iter=2000,
                        early_stopping=True,
                        random_state=1,
                    ),
                ),
            ]
        ),
    }


def save_plots(y_test, y_pred, model_metrics):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    low = min(y_test.min(), y_pred.min())
    high = max(y_test.max(), y_pred.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual oil recovery (%)")
    plt.ylabel("Predicted oil recovery (%)")
    plt.title("Actual vs Predicted Oil Recovery")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close()

    residuals = y_pred - y_test
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted oil recovery (%)")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Values")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residuals.png", dpi=180)
    plt.close()

    names = list(model_metrics.keys())
    r2_scores = [model_metrics[name]["test"]["r2"] for name in names]
    plt.figure(figsize=(7, 5))
    plt.bar(names, r2_scores)
    plt.ylabel("Test R²")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison.png", dpi=180)
    plt.close()


def extract_feature_importance(best_pipeline, X_columns):
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array(X_columns)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coefs_"):
        # For MLP: approximate first-layer absolute connection strength.
        values = np.mean(np.abs(model.coefs_[0]), axis=1)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(DATA_PATH)
    X, y = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    results = {}
    trained = {}
    for name, pipeline in make_model_candidates().items():
        pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        results[name] = {
            "train": regression_metrics(y_train, train_pred),
            "test": regression_metrics(y_test, test_pred),
        }
        trained[name] = pipeline

    best_name = max(results, key=lambda name: results[name]["test"]["r2"])
    best_model = trained[best_name]
    y_pred = best_model.predict(X_test)

    metrics = {
        "best_model": best_name,
        "rows_raw": int(df.shape[0]),
        "rows_used_after_filter": int(len(X)),
        "feature_count": int(X.shape[1]),
        "target": "oilRecovery",
        "models": results,
    }

    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    predictions = X_test.copy()
    predictions["actual_oilRecovery"] = y_test.values
    predictions["predicted_oilRecovery"] = y_pred
    predictions["residual"] = predictions["predicted_oilRecovery"] - predictions["actual_oilRecovery"]
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    fi = extract_feature_importance(best_model, X.columns)
    fi.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    save_plots(y_test, y_pred, results)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
