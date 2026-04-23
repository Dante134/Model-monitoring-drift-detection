# MLOps Credit Risk Monitor

A production-style MLOps pipeline demonstrating model monitoring and drift
detection for a credit risk scoring model. Built with MLflow, Evidently AI,
and FastAPI — no cloud required, runs entirely locally.

---

## What it does

| Step | Tool | What happens |
|---|---|---|
| Data generation | NumPy | Synthetic reference + drifted production data |
| Training | scikit-learn + MLflow | Logistic regression trained, metrics + model logged |
| Drift detection | Evidently AI | KS-test per feature, auto-retrain if drift found |
| Serving | FastAPI | REST API with predict, health, drift endpoints |

---

## Architecture

```
data/
  reference.csv   ← model was trained on this
  production.csv  ← new data (simulates regime change / drift)
       │
       ▼
  train.py        ← LogisticRegression + StandardScaler pipeline
       │                 logs params + metrics + model to MLflow
       ▼
  monitor.py      ← Evidently runs KS-test on each feature
       │                 dataset_drifted? → trigger retrain
       ▼
  api.py          ← FastAPI: /predict  /health  /drift/status
       │
  mlruns/         ← MLflow experiment store (auto-created)
  monitoring/     ← HTML drift reports + JSON log
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline in one shot
python main.py

# 3. Start the API
uvicorn api:app --reload
# → http://localhost:8000/docs  (Swagger UI)

# 4. View MLflow experiments
mlflow ui
# → http://localhost:5000
```

---

## API endpoints

| Method | Route | Description |
|---|---|---|
| GET | /health | Model status + metadata |
| POST | /predict | Score a single applicant |
| POST | /predict/batch | Score a list of applicants |
| GET | /drift/status | Latest drift check result |
| POST | /drift/run | Trigger drift check manually |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":34,"income":52000,"debt_ratio":0.35,"credit_score":670,"num_accounts":4}'
```

**Example response:**
```json
{
  "default_probability": 0.2341,
  "risk_label": "MEDIUM",
  "timestamp": "2024-11-15T10:23:45"
}
```

---

## Key MLOps concepts demonstrated

**Data drift** — When production data distribution shifts away from training
data, model predictions become unreliable. The Kolmogorov-Smirnov test
measures if two distributions are statistically the same.

**MLflow tracking** — Every training run logs hyperparameters, metrics
(AUC, F1, accuracy), and the model artifact so you can compare runs and
roll back to previous versions.

**Auto-retrain** — When drift is detected across >50% of features, the
pipeline automatically triggers a fresh training run on the new data.

**Model serving** — FastAPI wraps the model as a REST API with input
validation via Pydantic, so it can be called by any downstream service.

---

## How to extend

- **Schedule monitoring**: Run `monitor.py` as a cron job or Airflow DAG daily
- **Add model registry stages**: Use `mlflow.register_model()` to move models
  through Staging → Production → Archived
- **Containerise**: Add a Dockerfile, run with `docker-compose up`
- **Add alerts**: Send a Slack/email alert when drift is detected instead of
  (or before) auto-retrain

---

## Project structure

```
mlops_monitor/
├── main.py               ← run everything in one shot
├── train.py              ← training + MLflow logging
├── monitor.py            ← Evidently drift detection + auto-retrain
├── api.py                ← FastAPI serving endpoint
├── requirements.txt
├── README.md
├── data/
│   ├── generate_data.py  ← synthetic data generator
│   ├── reference.csv     ← created on first run
│   └── production.csv    ← created on first run
├── model/
│   └── credit_risk_model.pkl   ← saved after training
└── monitoring/
    ├── drift_report_<ts>.html  ← Evidently HTML report
    └── drift_results.json      ← drift check history log
```
