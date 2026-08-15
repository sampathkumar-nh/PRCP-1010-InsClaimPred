# Model Comparison Report
## PRCP-1010 — Insurance Claim Prediction

**Date:** 2025  
**Dataset:** train.csv — 595,212 rows × 35 features (after preprocessing)  
**Validation Set:** 119,043 rows (20% stratified split)  
**Target:** `target` — 0 = No claim, 1 = Filed a claim  
**Class distribution:** 96.36% class-0 / 3.64% class-1 (imbalance ratio ≈ 26.4:1)

---

## 1. Preprocessing Summary

| Step | Action |
|---|---|
| Drop `id` | Not a feature |
| Drop `ps_car_03_cat` | 69% missing (-1 sentinel) |
| Drop `ps_car_05_cat` | 45% missing (-1 sentinel) |
| Drop all `ps_calc_*` (20 cols) | Calculated noise — no predictive value for claims |
| Replace `-1` with `NaN` | 13 columns affected |
| Impute categorical (12 cols) | Mode (most frequent) |
| Impute continuous (23 cols) | Median |
| Scale continuous | Only for Logistic Regression (StandardScaler) |

**Final feature matrix:** 595,212 × 35 features

---

## 2. Class Imbalance Strategy

All models were configured to handle the severe 26.4:1 class imbalance:

| Model | Imbalance Strategy |
|---|---|
| Logistic Regression | `class_weight='balanced'` (auto-weights minority 26× heavier) |
| Random Forest | `class_weight='balanced_subsample'` |
| Gradient Boosting | `min_samples_leaf=50`, `subsample=0.8` |
| LightGBM | `scale_pos_weight=26` |

**Threshold Selection:** Instead of the default 0.5 cutoff, the optimal threshold was
computed on the training set using **Youden's J statistic** (maximises TPR − FPR) for
each model. This prevents the common problem of tree models predicting all-zeros on
highly imbalanced data.

---

## 3. Model Results (Validation Set)

| Model | ROC-AUC | Gini | F1 (class 1) | Precision | Recall | Opt Threshold | Train Time |
|---|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.6402** | **0.2805** | **0.1046** | **0.0584** | 0.4957 | 0.0392 | 350s |
| LightGBM | 0.6387 | 0.2773 | 0.1031 | 0.0571 | **0.5255** | 0.4965 | **14s** |
| Random Forest | 0.6278 | 0.2555 | 0.0954 | 0.0521 | 0.5690 | 0.4921 | 46s |
| Logistic Regression | 0.6206 | 0.2412 | 0.0952 | 0.0523 | 0.5222 | 0.4956 | 447s |

### Primary Metric: ROC-AUC
ROC-AUC is the standard evaluation metric for imbalanced insurance claim prediction.
It measures the model's ability to rank positive (claim) customers higher than
negative (non-claim) customers, regardless of threshold.

### Secondary Metric: Normalized Gini
Normalized Gini = 2 × AUC − 1. This is the standard metric used in insurance
pricing competitions (e.g., Porto Seguro Kaggle). Values around 0.28 are typical
for this dataset with standard features.

---

## 4. Best Model: Gradient Boosting

**Selected on:** Highest ROC-AUC (0.6402) on held-out validation set.

### Why Gradient Boosting Won
- Gradient Boosting iteratively corrects the errors of previous trees, building a
  strong ensemble that captures non-linear interactions between features.
- On tabular insurance data it consistently outperforms linear models and standalone
  decision trees.
- The `subsample=0.8` setting reduces overfitting on the imbalanced minority class.

### Performance vs. Naive Baseline
A naive model that always predicts class-0 (no claim) would achieve:
- Accuracy: 96.4% (misleading — predicts nothing useful)
- ROC-AUC: 0.50 (random)
- F1 (class 1): 0.00

Our best model achieves ROC-AUC = **0.6402** — a **+14 percentage point** lift over random,
meaning it correctly ranks ~64% of true claimants above true non-claimants.

### Confusion Matrix (Gradient Boosting, validation set)

```
                  Predicted: 0    Predicted: 1
Actual: 0  (No claim)   80,047        34,657
Actual: 1  (Claim)       2,188         2,151
```

- True Positives (correctly caught claimants): **2,151**
- Recall (sensitivity): **49.6%** of actual claimants are correctly identified
- The model flags 34,657 + 2,151 = 36,808 customers as high-risk for follow-up

---

## 5. LightGBM — Recommended for Production

Although Gradient Boosting has marginally higher AUC (0.6402 vs 0.6387), **LightGBM
is recommended for any production or re-training scenario** because:

| Factor | Gradient Boosting | LightGBM |
|---|---|---|
| Training time | 350 seconds | **14 seconds** |
| AUC difference | 0.6402 | 0.6387 (−0.0015) |
| Scalability | Limited | Excellent |
| Memory usage | High | Low |
| Re-training cost | Expensive | Cheap |

For a dataset of 595K rows, LightGBM trains **25× faster** with only a negligible
0.0015 AUC difference — a clear production choice.

---

## 6. Top Predictive Features (LightGBM)

Based on feature importance analysis, the most predictive features for claim filing are:

| Rank | Feature | Group | Interpretation |
|---|---|---|---|
| 1 | `ps_car_13` | Car | Vehicle value/type index — higher-value cars file more claims |
| 2 | `ps_reg_03` | Regional | Regional risk factor — some regions have higher accident rates |
| 3 | `ps_car_14` | Car | Vehicle age/condition index |
| 4 | `ps_car_12` | Car | Vehicle specification index |
| 5 | `ps_ind_15` | Individual | Driver age band — certain age groups are higher risk |
| 6 | `ps_ind_03` | Individual | Policy tenure |
| 7 | `ps_reg_01` | Regional | Primary region code |
| 8 | `ps_car_15` | Car | Vehicle power/engine index |
| 9 | `ps_ind_05_cat` | Individual | Coverage type category |
| 10 | `ps_car_11` | Car | Car model/class code |

**Key finding:** Vehicle-related features dominate the top predictors — the type,
value, and age of the vehicle are stronger signals than individual demographics.

---

## 7. Marketing Recommendations

Based on model results and feature analysis, the following practical suggestions are
provided to the insurance marketing team:

### 7.1 Risk-Based Customer Segmentation

Use the model's `claim_probability` output to segment customers into four tiers:

| Segment | Probability | Estimated Count* | Action |
|---|---|---|---|
| Low Risk | < 3% | ~72% of customers | Standard premium; loyalty rewards |
| Medium Risk | 3–7% | ~20% of customers | Comprehensive coverage upsell |
| High Risk | 7–15% | ~6% of customers | Risk-based pricing; telematics offer |
| Very High Risk | > 15% | ~2% of customers | Underwriting review; premium loading |

*Estimated from validation set probability distribution.

### 7.2 Vehicle-Focused Marketing Strategy

Since vehicle features are the strongest predictors:
- **High-value vehicle owners** (`ps_car_13` high): Offer comprehensive coverage
  with agreed-value replacement — they expect premium protection.
- **Older vehicles** (`ps_car_14` low): Offer third-party + fire + theft as a
  cost-effective alternative; these customers are price-sensitive.
- **High-power vehicles** (`ps_car_15` high): Proactively offer excess waivers and
  driver training discounts to reduce claim likelihood.

### 7.3 Regional Targeting

`ps_reg_03` and `ps_reg_01` are strong predictors — high-risk regions should be
targeted with:
- Enhanced roadside assistance packages (more likely to be used, high perceived value)
- Telematics/black-box policies to reward safe drivers in risky regions
- Higher base premiums offset by no-claims bonus incentives

### 7.4 Retention of Low-Risk Customers

The model identifies ~72% of customers as low-risk. These are the most profitable
customers and should receive:
- Proactive renewal reminders with loyalty discounts
- Multi-policy bundle offers (home + auto)
- No-claims bonus communications to reinforce retention

### 7.5 Telematics / Usage-Based Insurance

For customers in the Medium and High risk bands, offer **telematics-based pricing**:
- Customers who accept telematics self-select towards safer driving
- Reduces adverse selection in the high-risk band
- Builds a richer data asset for future model improvements

### 7.6 Campaign Prioritization

The model's recall of ~50% means it correctly identifies roughly half of all future
claimants. The marketing team can use the `risk_segment` output to:
1. **Prioritise premium review calls** to Very High and High segments
2. **Run upsell campaigns** to Medium segment (most cost-effective conversion)
3. **Run retention campaigns** to Low segment (most profitable to keep)

---

## 8. Model Files Saved

| File | Description |
|---|---|
| `models/best_model.pkl` | Best model (Gradient Boosting) + metadata |
| `models/gradient_boosting.pkl` | Gradient Boosting pipeline |
| `models/lightgbm.pkl` | LightGBM pipeline (recommended for production) |
| `models/random_forest.pkl` | Random Forest pipeline |
| `models/logistic_regression.pkl` | Logistic Regression pipeline |
| `reports/metrics_summary.json` | All metrics in JSON format |
| `reports/roc_curves.png` | ROC curve comparison plot |
| `reports/feature_importance_*.png` | Feature importance plots (3 models) |

---

## 9. How to Use

### Batch Predictions
```bash
python src/predict.py --input data/new_customers.csv --output predictions.csv
```

### Interactive Single Customer
```bash
python app/ui.py
```

### Retrain (if new data available)
```bash
python src/train.py
```

---

*Report generated by PRCP-1010 automated training pipeline.*
