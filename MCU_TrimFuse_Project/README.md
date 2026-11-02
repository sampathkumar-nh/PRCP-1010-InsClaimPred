# AI-Optimized Trim & Fuse for Derivative Chips
## MCU · PMIC · Sensor · SoC

**Project:** PRCP-1010 | AI Semiconductor Testing  
**Dataset:** 9 CSV files — 41,600+ records across all chip types  
**Author:** Sampathkumar Honasure  

---

## Project Overview

This project implements the full AI pipeline for **AI-Optimized Trim & Fuse** across
four semiconductor chip families (MCU, PMIC, Sensor, SoC). The goal is to:

1. **Cut trim test time** by replacing binary-search loops with single-shot ML predictions
2. **Share models across derivatives** through transfer learning on common IP blocks
3. **Prevent irreversible fuse errors** through schema-based validation before OTP programming
4. **Optimize memory repair and SKU assignment** to maximize yield and revenue

---

## Dataset Files (All 9 Used)

| File | Rows | Purpose |
|---|---|---|
| `mcu_trim_fuse.csv` | 8,420 | MCU predictive trim (BG, HSI, LDO), drift, fuse maps |
| `mcu_fuse_schema.csv` | 10 | Fuse-word bit fields per derivative |
| `mcu_fuse_map_audit.csv` | 1,500 | Fuse-map validation with injected errors |
| `pmic_trim_otp.csv` | 11,060 | PMIC multi-rail trim, threshold trims, OTP images |
| `pmic_otp_config_requests.csv` | 400 | Customer OTP config rule checking |
| `sensor_calibration.csv` | 11,060 | Multi-temperature calibration, post-reflow drift |
| `soc_binning_harvesting.csv` | 6,160 | Fmax/Vmin prediction, AVS bins, SKU assignment |
| `soc_sku_demand.csv` | 3 | SKU prices and weekly demand |
| `memory_repair_bitmaps.csv` | 3,000 | SRAM redundancy allocation |

---

## Project Structure

```
MCU_TrimFuse_Project/
├── data/                          # All 9 CSV datasets
├── models/                        # Trained model pickle files
├── reports/                       # metrics_summary.json + logs
├── src/
│   ├── train.py                   # Master pipeline runner
│   ├── mcu_trim.py                # MCU BG/HSI/LDO trim + transfer learning
│   ├── pmic_trim.py               # PMIC rail trim + OTP validation
│   ├── sensor_calib.py            # Sensor TCO/TCS single-insertion
│   ├── soc_binning.py             # SoC Fmax/Vmin + SKU harvesting
│   ├── fuse_validation.py         # Fuse-map schema validator
│   └── memory_repair.py           # Memory repair exact solver
├── app.py                         # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the full training pipeline
```bash
python src/train.py
```

### Launch the interactive dashboard
```bash
streamlit run app.py
```

---

## AI Modules & Results

### 1. MCU Predictive Trim
Uses `HistGradientBoostingRegressor` trained on MCU-A (15 wafers, GroupKFold split).

| Trim | Within ±1 Code | Iter Reduction |
|---|---|---|
| Bandgap (6-bit) | ~96% | 62% (6.6 → 2.5) |
| HSI Oscillator (7-bit) | ~81% | 64% (7.6 → 2.7) |
| LDO (5-bit) | ~99% | 60% (5.6 → 2.3) |

**Transfer Learning (MCU-C drift):** Fine-tuning on 50 MCU-C dies reduces drift MAE
from 3.97 mV (zero-shot) to 0.88 mV. Final bandgap pass rate improves accordingly.

### 2. PMIC Multi-Rail Trim
Predicts all 7 trim targets (BUCK1, BUCK2, LDO1, LDO2, OVP, UVLO, ILIM) from the
reference measurement + PCM data only — without measuring each rail.

| Rail | Within ±1 Code |
|---|---|
| BUCK1 / BUCK2 | 88–91% |
| LDO1 / LDO2 | 85–90% |
| OVP / UVLO | 82–83% |

**OTP Validation:** Rule-based checker catches out-of-range voltages, I2C address
collisions/reserved, and sequence conflicts. Precision: 100% | Recall: 74% | F1: 0.85

### 3. Sensor Single-Insertion Calibration
Predicts temperature coefficients (TCO, TCS) for all 3 axes from 25°C data only.
Trained on 20% of dies that have full 3-temperature data; predicts for the other 80%.

| Target | R² |
|---|---|
| TCO (X/Y/Z) | 0.82–0.83 |
| TCS (X/Y/Z) | 0.93 |
| Cal codes (offset/TCO/TCS) | 0.82–0.98 |

**Test time savings:** 3 insertions → 1 insertion = **67% time reduction**

### 4. SoC Fmax/Vmin Prediction
Predicts slowest-core Fmax and worst-case Vmin from process monitors and ring oscillators.

| Target | R² | MAE |
|---|---|---|
| Fmax (GHz) | 0.975 | 0.028 GHz |
| Vmin (mV) | 0.953 | 4.6 mV |

### 5. Fuse-Map Validation
Schema-based rule checker catches 6 error categories:
- wrong_flash_size · periph_mask_mismatch · missing_debug_lock
- reserved_bit_set · rdp_missing_automotive · trim_field_corrupt

**Results:** 406 injected errors | Recall: 91.6% | F1: 0.678

### 6. Memory Repair Optimization
Exact set-cover solver outperforms greedy allocation.

**Results:** greedy 61.4% → exact 62.0% | +1,156 additional arrays repaired

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `HistGradientBoostingRegressor` | Handles NaN natively, fast, strong on tabular data |
| `GroupKFold` on `wafer_id` | Prevents data leakage — dies on same wafer are correlated |
| 17 engineered features | Adds derived process spread, error terms, spatial squares |
| Schema-first fuse validation | Fuses are OTP — rule checks prevent irreversible errors |
| Exact solver for memory repair | Greedy misses repairable arrays; exact solver recovers them |

---

## Files Generated After Training

| File | Description |
|---|---|
| `models/mcu_trim_models.pkl` | MCU BG/HSI/LDO trained models |
| `models/mcu_drift_model.pkl` | MCU-C drift transfer model |
| `models/pmic_trim_models.pkl` | PMIC rail + threshold models |
| `models/sensor_calib_models.pkl` | Sensor TC prediction models |
| `models/soc_models.pkl` | SoC Fmax/Vmin models |
| `reports/metrics_summary.json` | All metrics in JSON |

---

*Project implements the AI pipeline from AI-Optimized-Trim-Fuse-Derivative-Chips.docx*
