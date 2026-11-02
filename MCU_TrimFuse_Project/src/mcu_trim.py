"""
MCU Predictive Trim Engine
==========================
Predicts bandgap (BG), HSI oscillator, and LDO trim codes for MCU derivatives.
Uses GroupKFold on wafer_id to prevent data leakage.
Also performs transfer learning for MCU-C drift-aware targets.
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mcu_trim_fuse.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports")

FEATURES = [
    "pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq", "pcm_tox_A", "pcm_mobility_rel",
    "bg_untrim_V", "hsi_untrim_MHz", "ldo_untrim_V",
    "x_mm", "y_mm", "radius_mm",
]

TARGETS = {
    "bg":  "bg_trim_code_opt",
    "hsi": "hsi_trim_code_opt",
    "ldo": "ldo_trim_code_opt",
}

ITERATIONS_BASELINE = {"bg": "bg_conv_iterations", "hsi": "hsi_conv_iterations", "ldo": "ldo_conv_iterations"}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["radius_sq"]         = df["radius_mm"] ** 2
    df["vth_product"]       = df["pcm_vth_n_mV"] * df["pcm_vth_p_mV"].abs()
    df["bg_error"]          = df["bg_untrim_V"] - 1.200
    df["hsi_error"]         = df["hsi_untrim_MHz"] - 16.0
    df["ldo_error"]         = df["ldo_untrim_V"] - 1.800
    df["process_spread"]    = df["pcm_vth_n_mV"] - df["pcm_vth_p_mV"].abs()
    return df


FEAT_ENG = FEATURES + ["radius_sq", "vth_product", "bg_error", "hsi_error", "ldo_error", "process_spread"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    return df


def within_n_codes(y_true, y_pred, n=1):
    return float(np.mean(np.abs(np.round(y_pred) - y_true) <= n))


def ai_iterations(y_true, y_pred):
    """Estimate tester iterations under AI predict-then-verify flow."""
    exact = np.abs(np.round(y_pred) - y_true) == 0
    off1  = np.abs(np.round(y_pred) - y_true) == 1
    return float(np.where(exact, 1, np.where(off1, 2, 3)).mean())


def train_mcu_trim(df_train, df_val, target_col, label):
    X_tr = df_train[FEAT_ENG].values
    y_tr = df_train[target_col].values
    X_va = df_val[FEAT_ENG].values
    y_va = df_val[target_col].values

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)

    mae  = mean_absolute_error(y_va, y_pred)
    r2   = r2_score(y_va, y_pred)
    w1   = within_n_codes(y_va, y_pred, 1)
    ai_it = ai_iterations(y_va, y_pred)
    return model, {"mae": round(mae, 4), "r2": round(r2, 4), "within_1": round(w1, 4), "ai_iter": round(ai_it, 3)}


def run_mcu_module():
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)

    df = load_data()
    df_a = df[df["derivative"] == "MCU-A"].copy()

    gkf = GroupKFold(n_splits=5)
    groups = df_a["wafer_id"].values

    results = {}
    models  = {}

    for key, col in TARGETS.items():
        fold_maes, fold_w1 = [], []
        best_model = None
        best_mae   = np.inf

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(df_a, groups=groups)):
            df_tr = df_a.iloc[tr_idx]
            df_va = df_a.iloc[va_idx]
            m, metrics = train_mcu_trim(df_tr, df_va, col, key)
            fold_maes.append(metrics["mae"])
            fold_w1.append(metrics["within_1"])
            if metrics["mae"] < best_mae:
                best_mae   = metrics["mae"]
                best_model = m

        # Final model on all MCU-A
        X_all = df_a[FEAT_ENG].values
        y_all = df_a[col].values
        final_model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, random_state=42
        )
        final_model.fit(X_all, y_all)
        models[key] = final_model

        baseline_iter = df_a[ITERATIONS_BASELINE[key]].mean()
        y_pred_cv = np.concatenate([
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6,
                                          min_samples_leaf=20, random_state=42).fit(
                df_a.iloc[tr_idx][FEAT_ENG].values, df_a.iloc[tr_idx][col].values
            ).predict(df_a.iloc[va_idx][FEAT_ENG].values)
            for tr_idx, va_idx in gkf.split(df_a, groups=groups)
        ])
        y_true_cv = df_a[col].values
        ai_it = ai_iterations(y_true_cv, y_pred_cv)

        results[key] = {
            "target":          col,
            "derivative":      "MCU-A",
            "cv_mae":          round(np.mean(fold_maes), 4),
            "cv_r2":           round(r2_score(df_a[col].values, y_pred_cv), 4),
            "within_1_code":   round(np.mean(fold_w1), 4),
            "fold_maes":       [round(x, 4) for x in fold_maes],
            "baseline_iter":   round(baseline_iter, 2),
            "ai_iter":         round(ai_it, 3),
            "iter_reduction":  round((baseline_iter - ai_it) / baseline_iter * 100, 1),
        }

        print(f"  MCU {key.upper()} trim | within Â+/-1: {np.mean(fold_w1)*100:.1f}% | "
              f"iter {baseline_iter:.1f} â†’ {ai_it:.2f}")

    # Transfer learning: MCU-C drift
    drift_results = _run_drift_transfer(df, models["bg"])
    results["drift_transfer"] = drift_results

    # Save models
    with open(os.path.join(MODELS_PATH, "mcu_trim_models.pkl"), "wb") as f:
        pickle.dump(models, f)

    return results


def _run_drift_transfer(df, bg_base_model):
    """Transfer learning: predict sort-to-final drift for MCU-C (LQFP100)."""
    df_c = df[df["derivative"] == "MCU-C"].copy()
    if len(df_c) < 60:
        return {"note": "Not enough MCU-C data"}

    DRIFT_COL = "bg_final_V"
    if DRIFT_COL not in df_c.columns:
        return {"note": "bg_final_V not found"}

    df_c["drift_mV"] = (df_c["bg_trimmed_sort_V"] - df_c["bg_final_V"]) * 1000.0

    n_fine = 50
    df_fine = df_c.sample(n=n_fine, random_state=42)
    df_hold = df_c.drop(df_fine.index)

    X_hold = df_hold[FEAT_ENG].values
    y_hold = df_hold["drift_mV"].values

    # Zero-shot: use MCU-A base model raw predictions as proxy
    bg_pred_hold = bg_base_model.predict(X_hold)
    mae_zero = mean_absolute_error(y_hold, bg_pred_hold - bg_pred_hold.mean() + y_hold.mean())

    # Fine-tune: small GBR on the 50 finetune dies
    ft_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    X_fine = df_fine[FEAT_ENG].values
    y_fine = df_fine["drift_mV"].values
    ft_model.fit(X_fine, y_fine)
    y_pred_ft = ft_model.predict(X_hold)
    mae_ft = mean_absolute_error(y_hold, y_pred_ft)

    # Final pass improvement
    SPEC_MV = 10.0
    center_target = df_hold["bg_trimmed_sort_V"].mean()
    drift_offset  = y_pred_ft.mean() / 1000.0
    adjusted_final = df_hold["bg_final_V"].values + (df_hold["bg_trimmed_sort_V"].values - center_target + drift_offset)
    pass_naive = float(np.mean(np.abs(df_hold["bg_final_V"].values - 1.200) * 1000 < SPEC_MV))
    pass_ft    = float(np.mean(np.abs(adjusted_final - 1.200) * 1000 < SPEC_MV))

    with open(os.path.join(MODELS_PATH, "mcu_drift_model.pkl"), "wb") as f:
        pickle.dump(ft_model, f)

    return {
        "n_finetune":       n_fine,
        "n_holdout":        len(df_hold),
        "mae_zero_shot_mV": round(mae_zero, 3),
        "mae_finetuned_mV": round(mae_ft, 3),
        "bg_final_pass_naive_pct":  round(pass_naive * 100, 2),
        "bg_final_pass_ft_pct":     round(pass_ft * 100, 2),
    }


if __name__ == "__main__":
    print("Running MCU Trim module...")
    results = run_mcu_module()
    print(json.dumps(results, indent=2))

