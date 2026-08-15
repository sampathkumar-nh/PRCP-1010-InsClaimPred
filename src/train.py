"""
train.py
========
Full training pipeline for PRCP-1010 Insurance Claim Prediction.

Steps:
  1. Load and preprocess data (via preprocess.py)
  2. Train/Validation split (stratified, 80/20)
  3. Train 4 models with class-imbalance handling:
       - Logistic Regression (baseline)
       - Random Forest
       - Gradient Boosting (sklearn)
       - LightGBM
  4. Evaluate each model on validation set
  5. Select best model by ROC-AUC
  6. Save best model pipeline to models/best_model.pkl
  7. Save full comparison results to reports/
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

# Add src/ to path so imports work from project root
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_prepare, build_preprocessor, get_feature_names
from evaluate import evaluate_model, plot_roc_curves, plot_feature_importance

# -- Paths ---------------------------------------------------------------------
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH   = os.path.join(BASE_DIR, "data", "train.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20


def build_model_definitions(cat_cols, cont_cols):
    """
    Returns a list of (name, Pipeline) tuples.
    Each Pipeline contains:
      - 'preprocessor' : ColumnTransformer  (impute ± scale)
      - 'classifier'   : the model
    """
    # -- Imbalance ratio ? 26:1  -> use class_weight='balanced' or scale_pos_weight
    class_ratio = 26  # approx 573518 / 21694

    models = []

    # 1. Logistic Regression  (needs scaling)
    pre_scaled = build_preprocessor(cat_cols, cont_cols, scale=True)
    lr = Pipeline([
        ("preprocessor", pre_scaled),
        ("classifier",   LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            C=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])
    models.append(("Logistic Regression", lr))

    # 2. Random Forest
    pre_tree = build_preprocessor(cat_cols, cont_cols, scale=False)
    rf = Pipeline([
        ("preprocessor", pre_tree),
        ("classifier",   RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])
    models.append(("Random Forest", rf))

    # 3. Gradient Boosting (sklearn) — use subsample for speed
    pre_tree2 = build_preprocessor(cat_cols, cont_cols, scale=False)
    gb = Pipeline([
        ("preprocessor", pre_tree2),
        ("classifier",   GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
        ))
    ])
    models.append(("Gradient Boosting", gb))

    # 4. LightGBM — fastest, best on large datasets
    pre_tree3 = build_preprocessor(cat_cols, cont_cols, scale=False)
    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=class_ratio,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_pipe = Pipeline([
        ("preprocessor", pre_tree3),
        ("classifier",   lgbm_clf)
    ])
    models.append(("LightGBM", lgbm_pipe))

    return models


def train_and_evaluate():
    print("\n" + "="*70)
    print("  PRCP-1010 — Insurance Claim Prediction")
    print("  Training Pipeline")
    print("="*70)

    # -- 1. Load data ----------------------------------------------------------
    X, y, cat_cols, cont_cols = load_and_prepare(DATA_PATH)
    feature_names = get_feature_names(None, cat_cols, cont_cols)

    # -- 2. Train / Validation split -------------------------------------------
    print(f"\n[train] Splitting data: {int((1-TEST_SIZE)*100)}% train / "
          f"{int(TEST_SIZE*100)}% validation (stratified) ...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[train] Train size: {X_train.shape[0]:,}  |  "
          f"Val size: {X_val.shape[0]:,}")
    print(f"[train] Train positives: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"[train] Val   positives: {y_val.sum():,}  ({y_val.mean()*100:.2f}%)")

    # -- 3. Build model definitions --------------------------------------------
    model_defs = build_model_definitions(cat_cols, cont_cols)

    # -- 4. Train, predict, evaluate -------------------------------------------
    summary_metrics = []
    roc_data        = []
    trained_models  = {}

    for name, pipeline in model_defs:
        print(f"\n[train] -- Training: {name} --")
        t0 = time.time()

        pipeline.fit(X_train, y_train)

        elapsed = time.time() - t0
        print(f"[train] Fit completed in {elapsed:.1f}s")

        y_proba = pipeline.predict_proba(X_val)[:, 1]
        # Find optimal threshold via Youden's J statistic (maximises TPR - FPR)
        from sklearn.metrics import roc_curve
        fpr_arr, tpr_arr, thresholds_arr = roc_curve(y_train,
            pipeline.predict_proba(X_train)[:, 1])
        youden_idx = np.argmax(tpr_arr - fpr_arr)
        opt_threshold = float(thresholds_arr[youden_idx])
        print(f"[train] Optimal threshold (Youden's J): {opt_threshold:.4f}")
        y_pred  = (y_proba >= opt_threshold).astype(int)

        metrics = evaluate_model(
            model_name=name,
            y_true=y_val,
            y_pred=y_pred,
            y_proba=y_proba,
            reports_dir=REPORTS_DIR,
        )
        metrics["train_time_s"]   = round(elapsed, 1)
        metrics["opt_threshold"]  = round(opt_threshold, 4)
        summary_metrics.append(metrics)

        roc_data.append({
            "model":   name,
            "y_true":  y_val,
            "y_proba": y_proba,
        })
        trained_models[name] = pipeline

        # Feature importance (tree models only)
        if name != "Logistic Regression":
            plot_feature_importance(
                model=pipeline,
                feature_names=feature_names,
                model_name=name,
                top_n=20,
                reports_dir=REPORTS_DIR,
            )

    # -- 5. Plot ROC curves ----------------------------------------------------
    plot_roc_curves(roc_data, reports_dir=REPORTS_DIR)

    # -- 6. Select best model --------------------------------------------------
    best = max(summary_metrics, key=lambda d: d["roc_auc"])
    best_name  = best["model"]
    best_model = trained_models[best_name]
    print(f"\n[train] >> Best model: {best_name}  (ROC-AUC = {best['roc_auc']:.4f})")

    # -- 7. Save best model ----------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump({
        "model":          best_model,
        "model_name":     best_name,
        "cat_cols":       cat_cols,
        "cont_cols":      cont_cols,
        "feature_names":  feature_names,
        "metrics":        best,
        "opt_threshold":  best.get("opt_threshold", 0.5),
    }, model_path)
    print(f"[train] Best model saved -> {model_path}")

    # Save all models
    for name, pipeline in trained_models.items():
        safe = name.lower().replace(" ", "_")
        joblib.dump(pipeline, os.path.join(MODELS_DIR, f"{safe}.pkl"))

    # -- 8. Save metrics summary JSON ------------------------------------------
    summary_path = os.path.join(REPORTS_DIR, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"[train] Metrics summary saved -> {summary_path}")

    # -- 9. Print comparison table ---------------------------------------------
    print("\n" + "="*70)
    print("  MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<22} {'ROC-AUC':>8} {'Gini':>8} {'F1':>8} "
          f"{'Precision':>10} {'Recall':>8} {'Time(s)':>8}")
    print("-"*70)
    for m in sorted(summary_metrics, key=lambda d: d["roc_auc"], reverse=True):
        marker = " << BEST" if m["model"] == best_name else ""
        print(f"{m['model']:<22} {m['roc_auc']:>8.4f} {m['gini']:>8.4f} "
              f"{m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['train_time_s']:>8.1f}{marker}")
    print("="*70)

    return summary_metrics, best_name, best_model, cat_cols, cont_cols, feature_names


if __name__ == "__main__":
    train_and_evaluate()
