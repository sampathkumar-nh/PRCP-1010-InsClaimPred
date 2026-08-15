"""
evaluate.py
===========
Evaluation utilities shared across all models.
Computes and prints:
  - ROC-AUC
  - Normalized Gini Coefficient (2*AUC - 1)
  - F1-score (positive class)
  - Precision / Recall
  - Confusion Matrix
  - Classification Report
Saves ROC curve and feature importance plots to reports/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve,
    average_precision_score
)


def gini(y_true, y_pred_proba):
    """Normalized Gini = 2*AUC - 1."""
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1


def evaluate_model(model_name: str, y_true, y_pred, y_proba,
                   reports_dir: str = "reports"):
    """
    Print full evaluation table for one model.
    Returns a dict of scalar metrics.
    """
    auc   = roc_auc_score(y_true, y_proba)
    gini_ = gini(y_true, y_proba)
    f1    = f1_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred)
    ap    = average_precision_score(y_true, y_proba)
    cm    = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  ROC-AUC              : {auc:.4f}")
    print(f"  Normalized Gini      : {gini_:.4f}")
    print(f"  Average Precision    : {ap:.4f}")
    print(f"  F1 Score (class 1)   : {f1:.4f}")
    print(f"  Precision (class 1)  : {prec:.4f}")
    print(f"  Recall    (class 1)  : {rec:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_true, y_pred, zero_division=0)}")

    return {
        "model": model_name,
        "roc_auc": round(auc, 4),
        "gini": round(gini_, 4),
        "avg_precision": round(ap, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


def plot_roc_curves(results_list, reports_dir: str = "reports"):
    """
    Plot ROC curves for all models on the same axes.
    results_list: list of dicts with keys 'model', 'y_true', 'y_proba'
    """
    os.makedirs(reports_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results_list:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
        auc = roc_auc_score(r["y_true"], r["y_proba"])
        ax.plot(fpr, tpr, label=f"{r['model']}  (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    path = os.path.join(reports_dir, "roc_curves.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] ROC curve saved -> {path}")


def plot_feature_importance(model, feature_names, model_name: str,
                            top_n: int = 20, reports_dir: str = "reports"):
    """
    Plot top-N feature importances for tree-based models.
    Works with RandomForest, GradientBoosting, LightGBM pipelines.
    """
    os.makedirs(reports_dir, exist_ok=True)
    try:
        # For sklearn Pipeline, extract the classifier step
        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("classifier") or \
                  model.named_steps.get("clf") or \
                  list(model.named_steps.values())[-1]
        else:
            clf = model

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_values   = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(top_n), top_values[::-1], align="center",
                color="#3b82f6", alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Features — {model_name}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        path = os.path.join(reports_dir, f"feature_importance_{safe_name}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[evaluate] Feature importance saved -> {path}")
        return top_features, top_values
    except AttributeError:
        print(f"[evaluate] {model_name} does not expose feature_importances_; skipping plot.")
        return [], []
