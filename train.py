"""
train.py — Train a credit risk model and log everything to MLflow.

MLflow tracks:
  - hyperparameters
  - metrics (AUC, accuracy, F1)
  - the trained model artifact

Run this first. It creates an mlruns/ folder (the local MLflow store).
View the UI with:  mlflow ui  (then open http://localhost:5000)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, classification_report
)
from sklearn.pipeline import Pipeline
import joblib
import os

# ── Config ────────────────────────────────────────────────
DATA_PATH  = "data/reference.csv"
MODEL_PATH = "model/credit_risk_model.pkl"
EXPERIMENT = "credit-risk-monitoring"

HYPERPARAMS = {
    "C":           1.0,       # inverse regularisation strength
    "max_iter":    500,
    "solver":      "lbfgs",
    "random_state": 42,
}


def train():
    os.makedirs("model", exist_ok=True)

    # ── Load data ─────────────────────────────────────────
    df      = pd.read_csv(DATA_PATH)
    X       = df.drop(columns=["default"])
    y       = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── MLflow experiment ─────────────────────────────────
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="logistic_regression_v1"):

        # Log hyperparameters
        mlflow.log_params(HYPERPARAMS)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size",  len(X_test))
        mlflow.log_param("default_rate_train", round(y_train.mean(), 4))

        # Build pipeline: scaler + model
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(**HYPERPARAMS)),
        ])
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred      = pipeline.predict(X_test)
        y_prob      = pipeline.predict_proba(X_test)[:, 1]

        auc      = roc_auc_score(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metrics({
            "auc":      round(auc,      4),
            "accuracy": round(accuracy, 4),
            "f1":       round(f1,       4),
        })

        # Log model to MLflow registry
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="credit_risk_model",
            registered_model_name="CreditRiskModel",
        )

        # Also save locally for the API to load quickly
        joblib.dump(pipeline, MODEL_PATH)

        run_id = mlflow.active_run().info.run_id

    print(f"\n✓ Training complete")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  MLflow run ID : {run_id}")
    print(f"  Model saved   : {MODEL_PATH}")
    print(f"\n  View experiment: mlflow ui  →  http://localhost:5000")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    return pipeline


if __name__ == "__main__":
    train()
