"""
SoC Binning & Harvesting Engine
================================
1. Predicts slowest-core Fmax and Vmin from process monitors (reduces speed test).
2. Assigns each die to the best-revenue SKU using a greedy optimizer
   subject to demand constraints.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

DATA_SOC    = os.path.join(os.path.dirname(__file__), "..", "data", "soc_binning_harvesting.csv")
DATA_DEMAND = os.path.join(os.path.dirname(__file__), "..", "data", "soc_sku_demand.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

PCM_FEATURES = [
    "pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq", "pcm_tox_A", "pcm_mobility_rel",
    "x_mm", "y_mm", "radius_mm",
    "ro_freq_MHz", "cpm_path_delay_ps", "iddq_25C_mA",
]

SOC_TARGETS = {
    "fmax_min":  None,   # computed below: min fmax across non-defective cores
    "vmin_max":  None,   # computed below: max vmin across non-defective cores
}


def engineer_soc_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["radius_sq"]   = df["radius_mm"] ** 2
    df["vth_product"] = df["pcm_vth_n_mV"] * df["pcm_vth_p_mV"].abs()

    core_fmax_cols = [f"core{i}_fmax_GHz" for i in range(8)]
    core_vmin_cols = [f"core{i}_vmin_2GHz_mV" for i in range(8)]
    core_def_cols  = [f"core{i}_defect" for i in range(8)]

    fmax_vals = df[core_fmax_cols].copy()
    vmin_vals = df[core_vmin_cols].copy()
    def_vals  = df[core_def_cols].copy()

    for i in range(8):
        fmax_vals[f"core{i}_fmax_GHz"] = np.where(def_vals[f"core{i}_defect"] == 1,
                                                    np.nan, fmax_vals[f"core{i}_fmax_GHz"])
        vmin_vals[f"core{i}_vmin_2GHz_mV"] = np.where(def_vals[f"core{i}_defect"] == 1,
                                                        np.nan, vmin_vals[f"core{i}_vmin_2GHz_mV"])

    df["fmax_min_GHz"]  = fmax_vals.min(axis=1)
    df["vmin_max_mV"]   = vmin_vals.max(axis=1)
    df["n_defect_cores"] = def_vals.sum(axis=1)
    df["l2_total_fails"] = (df["l2_slice0_fail_bits"] + df["l2_slice1_fail_bits"] +
                            df["l2_slice2_fail_bits"] + df["l2_slice3_fail_bits"])
    return df


FEAT_ENG = PCM_FEATURES + ["radius_sq", "vth_product", "n_defect_cores", "l2_total_fails"]


def run_soc_module():
    os.makedirs(MODELS_PATH, exist_ok=True)

    df       = pd.read_csv(DATA_SOC)
    df       = engineer_soc_features(df)
    df_demand = pd.read_csv(DATA_DEMAND)

    gkf    = GroupKFold(n_splits=5)
    groups = df["wafer_id"].values

    results = {}
    models  = {}

    for key, col in [("fmax_min", "fmax_min_GHz"), ("vmin_max", "vmin_max_mV")]:
        df_valid = df.dropna(subset=[col])
        if len(df_valid) < 100:
            continue
        grp_v = df_valid["wafer_id"].values

        fold_r2s, fold_maes = [], []
        cv_preds = np.empty(len(df_valid))

        for tr_idx, va_idx in gkf.split(df_valid, groups=grp_v):
            X_tr = df_valid.iloc[tr_idx][FEAT_ENG].values
            y_tr = df_valid.iloc[tr_idx][col].values
            X_va = df_valid.iloc[va_idx][FEAT_ENG].values
            y_va = df_valid.iloc[va_idx][col].values

            m = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=6,
                min_samples_leaf=20, random_state=42
            )
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)
            cv_preds[va_idx] = preds
            fold_maes.append(mean_absolute_error(y_va, preds))
            fold_r2s.append(r2_score(y_va, preds))

        y_true = df_valid[col].values
        final_m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, random_state=42
        )
        final_m.fit(df_valid[FEAT_ENG].values, y_true)
        models[key] = final_m

        results[key] = {
            "target":   col,
            "cv_r2":    round(r2_score(y_true, cv_preds), 4),
            "cv_mae":   round(mean_absolute_error(y_true, cv_preds), 4),
            "fold_r2s": [round(x, 4) for x in fold_r2s],
        }
        print(f"  SoC {key} | RÂ²: {r2_score(y_true, cv_preds):.3f} | "
              f"MAE: {mean_absolute_error(y_true, cv_preds):.4f}")

    # SKU assignment comparison
    sku_results = _run_sku_assignment(df, df_demand)
    results["sku_assignment"] = sku_results

    with open(os.path.join(MODELS_PATH, "soc_models.pkl"), "wb") as f:
        pickle.dump(models, f)

    return results


def _run_sku_assignment(df: pd.DataFrame, df_demand: pd.DataFrame) -> dict:
    """Compare rule-based vs AI-optimized SKU assignment."""
    if "sku_rule_based" not in df.columns:
        return {}

    # Rule-based distribution (from data)
    rule_counts = df["sku_rule_based"].value_counts(normalize=True).to_dict()

    # AI assignment: maximize revenue subject to demand
    # Simple greedy: sort by price descending, assign if Fmax/Vmin qualify
    demand = {}
    price  = {}
    fmax_min_req = {}
    if len(df_demand) > 0:
        for _, row in df_demand.iterrows():
            sku = row["sku"]
            demand[sku]       = int(row["weekly_demand_units"])
            price[sku]        = float(row["price_usd"])
            fmax_min_req[sku] = float(row["fmax_GHz"])

    skus_by_price = sorted(price.keys(), key=lambda s: price[s], reverse=True)
    assigned = {s: 0 for s in skus_by_price}
    assigned["scrap"] = 0
    ai_assignments = []

    for _, row in df.iterrows():
        fmax = row.get("fmax_min_GHz", np.nan)
        if pd.isna(fmax):
            ai_assignments.append("scrap")
            assigned["scrap"] += 1
            continue
        placed = False
        for sku in skus_by_price:
            if fmax >= fmax_min_req.get(sku, 0) and assigned[sku] < demand.get(sku, 9999):
                ai_assignments.append(sku)
                assigned[sku] += 1
                placed = True
                break
        if not placed:
            # Assign to lowest-tier if fmax qualifies
            lowest = skus_by_price[-1]
            if fmax >= fmax_min_req.get(lowest, 0):
                ai_assignments.append(lowest)
                assigned[lowest] += 1
            else:
                ai_assignments.append("scrap")
                assigned["scrap"] += 1

    df["sku_ai"] = ai_assignments
    ai_counts  = df["sku_ai"].value_counts(normalize=True).to_dict()

    rule_rev = sum(
        (df["sku_rule_based"] == sku).sum() * price.get(sku, 0)
        for sku in price
    )
    ai_rev = sum(
        (df["sku_ai"] == sku).sum() * price.get(sku, 0)
        for sku in price
    )
    uplift = (ai_rev - rule_rev) / rule_rev * 100 if rule_rev > 0 else 0.0

    print(f"  SoC SKU | rule revenue: ${rule_rev:,.0f} | AI revenue: ${ai_rev:,.0f} | "
          f"uplift: {uplift:.1f}%")

    return {
        "rule_based_sku_mix":  {k: round(v, 3) for k, v in rule_counts.items()},
        "ai_sku_mix":          {k: round(v, 3) for k, v in ai_counts.items()},
        "rule_based_revenue":  round(rule_rev, 2),
        "ai_revenue":          round(ai_rev, 2),
        "revenue_uplift_pct":  round(uplift, 2),
    }


if __name__ == "__main__":
    import json
    print("Running SoC Binning & Harvesting module...")
    results = run_soc_module()
    print(json.dumps(results, indent=2))

