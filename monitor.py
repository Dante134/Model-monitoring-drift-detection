"""
monitor.py — Data drift detection using Evidently AI.

What it does:
  1. Loads reference data (what the model was trained on)
  2. Loads production data (new incoming data)
  3. Runs Evidently's DataDriftPreset — computes statistical tests
     per feature (KS test for continuous, chi-squared for categorical)
  4. If drift is detected → triggers retrain automatically
  5. Saves an HTML drift report you can open in a browser

Evidently uses these tests under the hood:
  - Kolmogorov-Smirnov (KS) test for continuous features
  - Jensen-Shannon divergence as a secondary measure
  - Dataset-level drift = triggered when >50% of features drift

Run:  python monitor.py
"""

import os
import json
import pandas as pd
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric

# ── Config ────────────────────────────────────────────────
REFERENCE_PATH   = "data/reference.csv"
PRODUCTION_PATH  = "data/production.csv"
REPORT_DIR       = "monitoring"
DRIFT_THRESHOLD  = 0.5      # fraction of features that must drift to trigger retrain
RESULTS_LOG      = "monitoring/drift_results.json"


def run_drift_check() -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("📊 Running drift detection...")

    ref  = pd.read_csv(REFERENCE_PATH).drop(columns=["default"])
    prod = pd.read_csv(PRODUCTION_PATH).drop(columns=["default"])

    # ── Build Evidently report ────────────────────────────
    # DataDriftPreset runs a statistical test on each feature
    # DataQualityPreset checks for nulls, duplicates, out-of-range values
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(drift_share_threshold=DRIFT_THRESHOLD),
    ])

    report.run(reference_data=ref, current_data=prod)

    # Save HTML report (open this in a browser for a nice visual)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{REPORT_DIR}/drift_report_{timestamp}.html"
    report.save_html(report_path)
    print(f"  HTML report saved → {report_path}")

    # Extract results as dict for programmatic use
    result_dict = report.as_dict()

    # ── Parse drift summary ───────────────────────────────
    drift_results = {
        "timestamp":      timestamp,
        "report_path":    report_path,
        "features":       {},
        "dataset_drifted": False,
        "drift_share":    0.0,
    }

    # Walk through metrics to find per-feature drift and dataset-level result
    for metric in result_dict.get("metrics", []):
        metric_id = metric.get("metric", "")

        # Per-feature drift results
        if "ColumnDriftMetric" in metric_id:
            col    = metric["result"].get("column_name", "unknown")
            drifted = metric["result"].get("drift_detected", False)
            score  = metric["result"].get("stattest_threshold", None)
            drift_results["features"][col] = {
                "drifted": drifted,
                "p_value": score,
            }

        # Dataset-level drift result
        if "DatasetDriftMetric" in metric_id:
            drift_results["dataset_drifted"] = metric["result"].get("dataset_drift", False)
            drift_results["drift_share"]     = metric["result"].get("share_of_drifted_columns", 0.0)

    # Save JSON log
    logs = []
    if os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG) as f:
            logs = json.load(f)
    logs.append(drift_results)
    with open(RESULTS_LOG, "w") as f:
        json.dump(logs, f, indent=2)

    return drift_results


def print_drift_summary(results: dict):
    print("\n" + "═" * 55)
    print("  DRIFT DETECTION REPORT")
    print("═" * 55)
    print(f"  Timestamp     : {results['timestamp']}")
    print(f"  Drift share   : {results['drift_share']:.1%} of features")
    print(f"  Dataset drift : {'⚠️  YES' if results['dataset_drifted'] else '✓  NO'}")

    if results["features"]:
        print(f"\n  Feature-level results:")
        for feat, info in results["features"].items():
            status = "DRIFT" if info["drifted"] else "  ok "
            print(f"    [{status}] {feat}")
    print("═" * 55)


def trigger_retrain():
    """Auto-retrain when drift is detected."""
    print("\n🔄 Drift detected — triggering retrain...")
    import subprocess
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✓ Retrain complete.")
        print(result.stdout[-500:])     # last 500 chars of output
    else:
        print("  ✗ Retrain failed.")
        print(result.stderr[-300:])


def monitor(auto_retrain: bool = True):
    results = run_drift_check()
    print_drift_summary(results)

    if results["dataset_drifted"]:
        if auto_retrain:
            trigger_retrain()
        else:
            print("\n⚠️  Drift detected. Set auto_retrain=True to retrain automatically.")
    else:
        print("\n✓ No significant drift detected. Model is healthy.")

    return results


if __name__ == "__main__":
    monitor(auto_retrain=True)
