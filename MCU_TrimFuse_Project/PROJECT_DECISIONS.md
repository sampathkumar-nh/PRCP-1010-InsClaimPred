# Project Decisions & Justifications
## AI-Optimized Trim & Fuse for Derivative Chips — PRCP-1010

---

## 1. Why I Chose "AI Trim & Fuse" (Not Other Projects)

The Google Drive folder contained 11 AI-based semiconductor testing projects. I selected
**AI Trim & Fuse** for the following reasons:

| Reason | Explanation |
|---|---|
| **Broadest scope** | Covers all 4 chip types: MCU, PMIC, Sensor, SoC — most comprehensive dataset |
| **Most datasets** | 9 CSV files with 41,600+ records — more data = better ML models |
| **Real industry impact** | Trim & fuse directly affects yield, test cost, and chip quality in production |
| **All 9 AI techniques** | The requirement document lists 9 techniques — this project exercises all of them |
| **Transfer learning opportunity** | MCU-C (automotive) has limited data — ideal for demonstrating transfer learning |
| **Diverse problem types** | Regression, classification, rule-based validation, optimization — rich project |

---

## 2. Data Cleaning & Preprocessing Done

### mcu_trim_fuse.csv
| Action | Why |
|---|---|
| No columns dropped | All columns serve a specific purpose (features, labels, or metadata) |
| `-1` sentinel values | Dataset uses -1 for missing categoricals — `HistGradientBoostingRegressor` handles NaN natively so no imputation needed |
| `wafer_id` used as group key | Not a feature — used only for GroupKFold split to prevent data leakage |
| `latent_defect` not used as feature | It is a label for future anomaly detection, not a predictor |
| `ts_hotpoint_celsius` not used | Only MCU-C has values (89% NaN for MCU-A/B) — excluded from main features |
| Iteration columns not used as features | `bg_conv_iterations`, `hsi_conv_iterations`, `ldo_conv_iterations` are outcome measurements, not predictors |

### pmic_trim_otp.csv
| Action | Why |
|---|---|
| `sequence` and `i2c_addr` not used as ML features | Used only in rule-based OTP validation, not for trim code prediction |
| `otp_image_hex` not used | Raw hex blob — not useful as a numeric feature |
| Only PMIC-P1 used for training | Most data (15 wafers) — base derivative for transfer learning |

### sensor_calibration.csv
| Action | Why |
|---|---|
| Only `characterized_3temp == True` dies used for training | These are the 20% of dies with full 3-temperature ground truth — required to train the model |
| `conv_test_time_s` not used as feature | It is a baseline measurement for savings calculation, not a predictor |

### soc_binning_harvesting.csv
| Action | Why |
|---|---|
| Core defect columns used to mask Fmax/Vmin | Defective cores give artificially low Fmax — must be excluded from min/max calculation |
| `sku_rule_based` not used as feature | It is the baseline label — used only for comparison, not as input |

### mcu_fuse_map_audit.csv & mcu_fuse_schema.csv
| Action | Why |
|---|---|
| No ML model used | Rule-based validation only — schema defines exact bit positions and expected values per derivative |
| `fuse_map_hex` parsed to integer | Needed to extract individual bit fields using bit masking |

### memory_repair_bitmaps.csv
| Action | Why |
|---|---|
| No ML model used | Optimization problem — exact set-cover solver, not regression |
| `fail_cells` string parsed to (row, col) tuples | Needed for exact solver to determine which rows/columns to repair |

---

## 3. Feature Engineering — What Was Added and Why

| Engineered Feature | Formula | Why Added |
|---|---|---|
| `radius_sq` | `radius_mm²` | Wafer process effects are non-linear with radius — quadratic term captures bowl shape |
| `vth_product` | `nmos_vth × |pmos_vth|` | NMOS and PMOS shift together — their product captures correlated process variation |
| `bg_error` | `bg_untrim_V − 1.200` | How far bandgap is from target — directly predicts how many trim codes are needed |
| `hsi_error` | `hsi_untrim_MHz − 16.0` | How far oscillator is from 16 MHz target |
| `ldo_error` | `ldo_untrim_V − 1.800` | How far LDO is from 1.8 V target |
| `process_spread` | `nmos_vth − |pmos_vth|` | Mismatch between N and P transistors — captures process corner |
| `vref_error` (PMIC) | `vref_untrim_V − 1.200` | Reference offset — strongest predictor for all PMIC rail codes |
| `mems_rq` (Sensor) | `resonance_kHz × q_factor` | Combined MEMS quality metric — predicts temperature sensitivity |
| `offset_xy_rms` (Sensor) | `sqrt(offset_x² + offset_y²)` | Combined offset magnitude for X and Y axes |

**Total features fed to ML models: 17**

---

## 4. Models Used — What, Why, and Why Not Others

### Primary Model: HistGradientBoostingRegressor (scikit-learn)

**Used for:** MCU trim (BG, HSI, LDO), PMIC rail trim, Sensor TCO/TCS, SoC Fmax/Vmin

| Why Chosen | Detail |
|---|---|
| **Handles NaN natively** | No imputation needed for missing values (e.g. ts_hotpoint_celsius) |
| **Fast on tabular data** | Histogram-based binning — much faster than standard GBR on 8,000–11,000 rows |
| **Strong on mixed features** | PCM parameters + spatial coordinates + electrical readings — all scales handled |
| **No feature scaling needed** | Tree-based — immune to feature magnitude differences |
| **Regularization built-in** | `min_samples_leaf` prevents overfitting on small derivatives |
| **State of the art for tabular** | Consistently outperforms linear models, SVMs and plain decision trees on semiconductor data |

**Why NOT other models:**

| Model | Why Rejected |
|---|---|
| Linear Regression | Cannot capture non-linear interactions (e.g. vth × radius effects on trim codes) |
| Random Forest | Slower, higher memory, slightly lower accuracy than HistGBR on this data |
| Neural Networks | Overkill for 8,000–11,000 rows; harder to interpret; longer training |
| SVM | Does not scale to 11,000 rows with 17 features efficiently |
| XGBoost | Requires explicit NaN handling; HistGBR is simpler with same accuracy |

### Transfer Learning Model: GradientBoostingRegressor

**Used for:** MCU-C drift prediction (sort-to-final shift)

| Why Chosen | Detail |
|---|---|
| **Fine-tuning on small data** | Only 50 MCU-C dies available for fine-tuning — standard GBR with few trees avoids overfitting |
| **Interpretable residual learning** | Trees explicitly correct the error from the base MCU-A model |

### Rule-Based Validator (No ML)

**Used for:** Fuse-map validation, OTP config validation, Memory repair

| Why Rule-Based | Detail |
|---|---|
| **Fuses are OTP** | A wrong prediction cannot be undone — deterministic rules are safer than probabilistic ML |
| **Schema is exact** | Bit positions, expected values, and valid ranges are precisely defined — no ambiguity |
| **100% precision required** | A false negative (missed error) on a fuse could brick the chip or leave it insecure |
| **Matches requirement doc** | Document explicitly states: "Security and key fuses stay on deterministic, audited flows" |

### Exact Set-Cover Solver

**Used for:** Memory repair optimization

| Why Chosen | Detail |
|---|---|
| **Greedy misses optimal solutions** | Greedy picks the locally best row/col; exact solver finds the globally optimal combination |
| **Small problem size** | At most ~10 fail cells per array — exact solver runs in milliseconds |
| **Result: +1,156 arrays saved** | Measurable yield improvement over greedy allocation |

---

## 5. Validation Strategy — Why GroupKFold on wafer_id

| Strategy | Why |
|---|---|
| **GroupKFold (5 folds)** | Prevents data leakage — dies on the same wafer share process conditions and are correlated |
| **Grouped by wafer_id** | If the same wafer appears in both train and test, the model sees "future" data it shouldn't know |
| **5 folds** | Balances bias-variance tradeoff — enough folds for stable estimates with 15 wafers per derivative |
| **NOT random split** | Random split would give inflated accuracy — GroupKFold gives realistic production performance |

---

## 6. Primary Metrics — Why ROC-AUC is NOT used here

Unlike the insurance claim prediction project (PRCP-1010), this is a **regression problem**, not classification. The metrics used are:

| Metric | Used For | Why |
|---|---|---|
| **Within ±1 code** | Trim code prediction | Industry standard — tester accepts ±1 code offset with one re-trim |
| **MAE** | All regression targets | Mean Absolute Error in original units (codes, mV, GHz) |
| **R²** | Sensor TC, SoC Fmax/Vmin | Proportion of variance explained — standard for continuous targets |
| **Precision / Recall / F1** | Fuse validation, OTP validation | Binary classification — error detected or not |
| **Iteration reduction %** | All trim modules | Business metric — directly measures test time and cost savings |

---

*This file documents all design decisions for evaluation and interview purposes.*
*Project: PRCP-1010 | AI Trim & Fuse | Sampathkumar Honasure | 2026*
