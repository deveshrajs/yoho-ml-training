"""
Standalone Inference Pipeline for Customer Churn Prediction.
Accepts raw batch CSV files, applies trained preprocessing/model artifacts,
and outputs scored customer risk files.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

# Ensure src directory is in python search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from features import TelecomFeatureEngineer


def run_batch_inference(
    input_csv_path: str,
    output_csv_path: str = "predictions.csv",
    model_artifact_path: str = "models/churn_model.joblib",
    threshold_override: float = None
) -> pd.DataFrame:
    """
    Executes end-to-end batch scoring for new customer records.
    """
    if not os.path.exists(model_artifact_path):
        # Fallback search path relative to script parent
        alt_path = os.path.join(os.path.dirname(SCRIPT_DIR), model_artifact_path)
        if os.path.exists(alt_path):
            model_artifact_path = alt_path
        else:
            raise FileNotFoundError(f"Trained model artifact not found at {model_artifact_path}")

    print(f"Loading trained ML artifact from: {model_artifact_path}")
    artifact = joblib.load(model_artifact_path)
    pipeline = artifact["model"]
    calibrated_threshold = threshold_override if threshold_override is not None else artifact.get("threshold", 0.50)

    print(f"Reading input records from: {input_csv_path}")
    raw_df = pd.read_csv(input_csv_path)
    total_records = len(raw_df)

    # Preserve customerID for CRM matching
    if "customerID" in raw_df.columns:
        customer_ids = raw_df["customerID"]
        inference_features = raw_df.drop(columns=["customerID"])
    else:
        customer_ids = pd.Series([f"CUST_{i:06d}" for i in range(1, total_records + 1)])
        inference_features = raw_df.copy()

    # Drop target column if present in input
    if "Churn" in inference_features.columns:
        inference_features = inference_features.drop(columns=["Churn"])

    print(f"Running model inference across {total_records} customer records...")
    churn_probabilities = pipeline.predict_proba(inference_features)[:, 1]
    binary_predictions = (churn_probabilities >= calibrated_threshold).astype(int)

    # Assign risk tier for CRM prioritization
    risk_tiers = np.where(
        churn_probabilities >= calibrated_threshold,
        "High Risk",
        np.where(churn_probabilities >= calibrated_threshold * 0.70, "Medium Risk", "Low Risk")
    )

    results_df = pd.DataFrame({
        "customerID": customer_ids,
        "churn_probability": np.round(churn_probabilities, 4),
        "churn_prediction": binary_predictions,
        "risk_tier": risk_tiers,
        "operating_threshold": np.round(calibrated_threshold, 4)
    })

    results_df.to_csv(output_csv_path, index=False)
    high_risk_count = (binary_predictions == 1).sum()
    print(f"Successfully generated {len(results_df)} predictions.")
    print(f"Saved results to: {output_csv_path}")
    print(f"Flagged {high_risk_count} customers ({high_risk_count/total_records*100:.1f}%) as High Risk (Threshold = {calibrated_threshold:.4f})")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom Churn Batch Scoring Engine")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(SCRIPT_DIR), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        help="Path to raw customer input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(SCRIPT_DIR), "predictions.csv"),
        help="Destination path for output predictions CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(os.path.dirname(SCRIPT_DIR), "models", "churn_model.joblib"),
        help="Path to serialized joblib model artifact"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional custom probability threshold override"
    )

    args = parser.parse_args()
    run_batch_inference(
        input_csv_path=args.input,
        output_csv_path=args.output,
        model_artifact_path=args.model,
        threshold_override=args.threshold
    )
