"""
api.py — FastAPI model serving endpoint.

Routes:
  GET  /health        — liveness check + model metadata
  POST /predict       — score a single applicant
  POST /predict/batch — score a batch (list of applicants)
  GET  /drift/status  — latest drift check result
  POST /drift/run     — manually trigger drift check + optional retrain

Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
import os
from datetime import datetime

# ── Load model ────────────────────────────────────────────
MODEL_PATH   = "model/credit_risk_model.pkl"
RESULTS_LOG  = "monitoring/drift_results.json"

app = FastAPI(
    title="Credit Risk Model API",
    description="MLOps project: credit scoring with drift monitoring",
    version="1.0.0",
)

# Load model at startup (cached in memory)
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print("⚠️  Model not found. Run train.py first.")


# ── Schemas ───────────────────────────────────────────────

class Applicant(BaseModel):
    age:          int   = Field(..., ge=18, le=100, example=34)
    income:       int   = Field(..., ge=0,          example=52000)
    debt_ratio:   float = Field(..., ge=0, le=1,    example=0.35)
    credit_score: int   = Field(..., ge=300, le=850, example=670)
    num_accounts: int   = Field(..., ge=0,          example=4)

class PredictionResponse(BaseModel):
    default_probability: float
    risk_label:          str      # LOW / MEDIUM / HIGH
    timestamp:           str

class BatchRequest(BaseModel):
    applicants: list[Applicant]


# ── Helper ────────────────────────────────────────────────

def score(applicant_dict: dict) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    df   = pd.DataFrame([applicant_dict])
    prob = float(model.predict_proba(df)[0, 1])

    if prob < 0.2:
        label = "LOW"
    elif prob < 0.5:
        label = "MEDIUM"
    else:
        label = "HIGH"

    return PredictionResponse(
        default_probability=round(prob, 4),
        risk_label=label,
        timestamp=datetime.utcnow().isoformat(),
    )


# ── Routes ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok" if model is not None else "model_not_loaded",
        "model_path":   MODEL_PATH,
        "model_loaded": model is not None,
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: Applicant):
    return score(applicant.model_dump())


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    return [score(a.model_dump()) for a in request.applicants]


@app.get("/drift/status")
def drift_status():
    if not os.path.exists(RESULTS_LOG):
        return {"message": "No drift checks run yet. POST /drift/run to start."}
    with open(RESULTS_LOG) as f:
        logs = json.load(f)
    return {"latest": logs[-1], "total_checks": len(logs)}


@app.post("/drift/run")
def run_drift(auto_retrain: bool = False):
    """Manually trigger a drift check (and optional retrain)."""
    try:
        from monitor import monitor
        results = monitor(auto_retrain=auto_retrain)
        return {
            "drift_detected": results["dataset_drifted"],
            "drift_share":    results["drift_share"],
            "report_path":    results["report_path"],
            "retrain_triggered": auto_retrain and results["dataset_drifted"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
