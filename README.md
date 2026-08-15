# PRCP-1010 — Insurance Claim Prediction

## Overview

This project builds a **binary classification model** that predicts whether a
customer is likely to file an insurance claim (`target = 1`).  
The predictions help the insurance marketing team identify high-risk and
low-risk customers, enabling targeted communication and appropriate pricing.

**Domain:** Finance / Insurance  
**Task:** Binary classification (claim vs. no claim)  
**Dataset:** ~595,212 rows × 59 columns (anonymized features)

---

## Project Structure

```
PRCP-1010-InsClaimPred/
├── data/
│   └── train.csv               ← raw training data
├── src/
│   ├── preprocess.py           ← data loading & preprocessing
│   ├── train.py                ← model training & comparison
│   ├── evaluate.py             ← evaluation metrics & plots
│   └── predict.py              ← batch prediction script
├── models/
│   ├── best_model.pkl          ← saved best model pipeline
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── lightgbm.pkl
├── app/
│   └── ui.py                   ← interactive CLI prediction UI
├── reports/
│   ├── metrics_summary.json    ← all models' metrics
│   ├── model_comparison.md     ← final comparison report
│   ├── roc_curves.png          ← ROC curve plot
│   └── feature_importance_*.png
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train All Models

```bash
cd PRCP-1010-InsClaimPred
python src/train.py
```

This will:
- Preprocess the dataset
- Train 4 models (Logistic Regression, Random Forest, Gradient Boosting, LightGBM)
- Evaluate each on a held-out validation set
- Save the best model to `models/best_model.pkl`
- Save plots and metrics to `reports/`

### 3. Batch Predict on New Data

```bash
python src/predict.py --input data/new_customers.csv --output predictions.csv
```

The output CSV contains:
- `claim_probability` — predicted probability (0–1)
- `predicted_label` — 0 or 1
- `risk_segment` — Low / Medium / High / Very High

### 4. Interactive Single-Customer Prediction UI

```bash
python app/ui.py
```

Guides you step-by-step through entering customer details and returns the
prediction, risk segment, and a marketing recommendation.

---

## Dataset Details

| Property | Value |
|---|---|
| Total rows | 595,212 |
| Features | 59 (including `id` and `target`) |
| Target | `target` (0 = no claim, 1 = claim filed) |
| Class balance | 96.4% class-0 / 3.6% class-1 |
| Missing values | Encoded as **-1** in 13 columns |

### Feature Groups

| Prefix | Description |
|---|---|
| `ps_ind_*` | Individual / policyholder characteristics |
| `ps_reg_*` | Regional features |
| `ps_car_*` | Vehicle / car features |
| `ps_calc_*` | Calculated noise features (dropped) |

---

## Preprocessing

1. **Drop** `id`, `ps_car_03_cat` (69% missing), `ps_car_05_cat` (45% missing)
2. **Drop** all `ps_calc_*` columns (calculated noise — no causal link to claims)
3. **Replace** `-1` sentinel values with `NaN`
4. **Impute** categorical columns with mode; continuous with median
5. **Scale** continuous features only for Logistic Regression

---

## Models Trained

| Model | Class Imbalance Strategy |
|---|---|
| Logistic Regression | `class_weight='balanced'` + StandardScaler |
| Random Forest | `class_weight='balanced_subsample'` |
| Gradient Boosting | Subsample 0.8, `min_samples_leaf=50` |
| LightGBM | `scale_pos_weight=26` (≈ imbalance ratio) |

---

## Evaluation Metric

- **Primary**: ROC-AUC — standard for imbalanced insurance data
- **Secondary**: Normalized Gini = 2 × AUC − 1
- Also reported: F1, Precision, Recall, Average Precision

---

## Marketing Recommendations

See `reports/model_comparison.md` for the full recommendation section.

### Summary

| Risk Segment | Probability | Recommended Action |
|---|---|---|
| Low (< 3%) | < 3% | Standard coverage; loyalty incentives |
| Medium (3–7%) | 3–7% | Comprehensive coverage; value-add features |
| High (7–15%) | 7–15% | Risk-based pricing; telematics discounts |
| Very High (> 15%) | > 15% | Underwriting review; premium loading |

---

## Results

See `reports/model_comparison.md` for full metrics after running training.

---

## Requirements

```
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.4.2
lightgbm==4.3.0
matplotlib==3.8.4
seaborn==0.13.2
joblib==1.4.2
```
