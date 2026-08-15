"""
predict.py
==========
Batch prediction script.

Usage (from project root):
    python src/predict.py --input data/new_customers.csv --output predictions.csv

The input CSV must have the same feature columns as train.csv
(target column is optional; if present it is ignored for prediction).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import replace_minus_one, drop_columns

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run  python src/train.py  first."
        )
    artifact = joblib.load(MODEL_PATH)
    print(f"[predict] Loaded model : {artifact['model_name']}")
    print(f"[predict] Val ROC-AUC  : {artifact['metrics']['roc_auc']}")
    print(f"[predict] Opt threshold: {artifact.get('opt_threshold', 0.5)}")
    return artifact


def predict(input_path: str, output_path: str = None, threshold: float = None):
    """
    Load input CSV, run predictions, return/save results DataFrame.
    """
    artifact  = load_model()
    model     = artifact["model"]
    if threshold is None:
        threshold = artifact.get("opt_threshold", 0.5)
        print(f"[predict] Using saved optimal threshold: {threshold}")

    print(f"[predict] Reading input: {input_path}")
    df = pd.read_csv(input_path)
    print(f"[predict] Input shape: {df.shape}")

    # Remove target if present (for convenience)
    if "target" in df.columns:
        y_true = df["target"].values
        df_feat = df.drop(columns=["target"])
    else:
        y_true  = None
        df_feat = df.copy()

    # Preprocess (same steps as training — model pipeline handles imputation)
    df_feat = replace_minus_one(df_feat)
    df_feat = drop_columns(df_feat)

    # Remove id if present
    ids = df_feat["id"].values if "id" in df_feat.columns else np.arange(len(df_feat))
    if "id" in df_feat.columns:
        df_feat = df_feat.drop(columns=["id"])

    # Predict
    y_proba = model.predict_proba(df_feat)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    results = pd.DataFrame({
        "id":              ids,
        "claim_probability": np.round(y_proba, 6),
        "predicted_label":   y_pred,
        "risk_segment":      pd.cut(
            y_proba,
            bins=[0, 0.03, 0.07, 0.15, 1.0],
            labels=["Low", "Medium", "High", "Very High"]
        )
    })

    if y_true is not None:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_proba)
        print(f"[predict] ROC-AUC on this dataset: {auc:.4f}")

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"[predict] Predictions saved -> {output_path}")

    print(f"[predict] Predicted positives: {y_pred.sum()} / {len(y_pred)} "
          f"({y_pred.mean()*100:.2f}%)")
    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Insurance Claim Prediction — Batch Predict"
    )
    parser.add_argument("--input",     required=True, help="Path to input CSV")
    parser.add_argument("--output",    default=None,  help="Path to save predictions CSV")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    args = parser.parse_args()
    predict(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
