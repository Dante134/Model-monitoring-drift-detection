"""
main.py — Run the full MLOps pipeline in one shot.

Steps:
  1. Generate synthetic data (reference + drifted production)
  2. Train model + log to MLflow
  3. Run drift detection (triggers retrain if drift found)
  4. Print instructions to start the API

This is the single entry point for the project demo.
"""

import os
import sys

def step(n: int, msg: str):
    print(f"\n{'─'*55}")
    print(f"  STEP {n}: {msg}")
    print(f"{'─'*55}")


def main():
    print("=" * 55)
    print("  MLOps Credit Risk Monitoring Pipeline")
    print("=" * 55)

    # ── Step 1: Generate data ─────────────────────────────
    step(1, "Generating synthetic data")
    from data.generate_data import make_batch
    import pandas as pd

    os.makedirs("data", exist_ok=True)
    ref  = make_batch(1000, drift=False)
    prod = make_batch(500,  drift=True)
    ref.to_csv("data/reference.csv",   index=False)
    prod.to_csv("data/production.csv", index=False)
    print(f"  Reference  : {len(ref)} rows | default rate: {ref['default'].mean():.2%}")
    print(f"  Production : {len(prod)} rows | default rate: {prod['default'].mean():.2%}")

    # ── Step 2: Train ─────────────────────────────────────
    step(2, "Training model + logging to MLflow")
    from train import train
    train()

    # ── Step 3: Drift detection ───────────────────────────
    step(3, "Running drift detection")
    from monitor import monitor
    results = monitor(auto_retrain=True)

    # ── Step 4: Instructions ──────────────────────────────
    step(4, "Pipeline complete — next steps")
    print("""
  Start the API:
    uvicorn api:app --reload
    → http://localhost:8000/docs

  View MLflow experiments:
    mlflow ui
    → http://localhost:5000

  Open the drift report:
    monitoring/drift_report_<timestamp>.html

  Test the API (example curl):
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"age":34,"income":52000,"debt_ratio":0.35,
           "credit_score":670,"num_accounts":4}'
    """)

    print("=" * 55)
    print(f"  Drift detected : {results['dataset_drifted']}")
    print(f"  Drift share    : {results['drift_share']:.1%} of features")
    print("=" * 55)


if __name__ == "__main__":
    main()
