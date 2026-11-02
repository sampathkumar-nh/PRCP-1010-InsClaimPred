"""
Sensor Calibration Engine
==========================
Predicts temperature coefficients (TCO, TCS) and calibration codes for
3-axis MEMS accelerometers from a single room-temperature (25Â°C) insertion.
Replaces the conventional 3-temperature test insertion, cutting test time.
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

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "sensor_calibration.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

# Features available at room-temperature only (25Â°C insertion)
FEATURES_25C = [
    "pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq", "pcm_tox_A", "pcm_mobility_rel",
    "x_mm", "y_mm", "radius_mm",
    "stress_gauge_mV", "asic_vref_mV", "mems_resonance_kHz", "mems_q_factor",
    "offset_x_25C_mg", "sens_x_25C_rel",
    "offset_y_25C_mg", "sens_y_25C_rel",
    "offset_z_25C_mg", "sens_z_25C_rel",
]

# Targets: temperature coefficients for all 3 axes
TC_TARGETS = {
    "tco_x":  "tco_x_true_mg_per_C",
    "tcs_x":  "tcs_x_true_ppm_per_C",
    "tco_y":  "tco_y_true_mg_per_C",
    "tcs_y":  "tcs_y_true_ppm_per_C",
    "tco_z":  "tco_z_true_mg_per_C",
    "tcs_z":  "tcs_z_true_ppm_per_C",
}

# Calibration code targets
CODE_TARGETS = {
    "cal_offset_code_x": "cal_offset_code_x",
    "cal_tco_code_x":    "cal_tco_code_x",
    "cal_tcs_code_x":    "cal_tcs_code_x",
    "cal_offset_code_y": "cal_offset_code_y",
    "cal_tco_code_y":    "cal_tco_code_y",
    "cal_tcs_code_y":    "cal_tcs_code_y",
    "cal_offset_code_z": "cal_offset_code_z",
    "cal_tco_code_z":    "cal_tco_code_z",
    "cal_tcs_code_z":    "cal_tcs_code_z",
}


def engineer_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["radius_sq"]     = df["radius_mm"] ** 2
    df["vth_product"]   = df["pcm_vth_n_mV"] * df["pcm_vth_p_mV"].abs()
    df["vref_drift"]    = df["asic_vref_mV"] - df["asic_vref_mV"].mean()
    df["mems_rq"]       = df["mems_resonance_kHz"] * df["mems_q_factor"]
    df["offset_xy_rms"] = np.sqrt(df["offset_x_25C_mg"] ** 2 + df["offset_y_25C_mg"] ** 2)
    return df


FEAT_ENG = FEATURES_25C + ["radius_sq", "vth_product", "vref_drift", "mems_rq", "offset_xy_rms"]


def within_n_codes(y_true, y_pred, n=1):
    return float(np.mean(np.abs(np.round(y_pred) - y_true) <= n))


def run_sensor_module():
    os.makedirs(MODELS_PATH, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = engineer_sensor_features(df)

    # Only use dies characterized at all 3 temperatures (20%) for training
    df_char  = df[df["characterized_3temp"] == True].copy()
    df_pred  = df[df["characterized_3temp"] == False].copy()

    gkf    = GroupKFold(n_splits=5)
    groups = df_char["wafer_id"].values

    results = {}
    models  = {}

    all_targets = {**TC_TARGETS, **CODE_TARGETS}

    for key, col in all_targets.items():
        fold_r2s, fold_maes = [], []
        cv_preds = np.empty(len(df_char))

        for tr_idx, va_idx in gkf.split(df_char, groups=groups):
            X_tr = df_char.iloc[tr_idx][FEAT_ENG].values
            y_tr = df_char.iloc[tr_idx][col].values
            X_va = df_char.iloc[va_idx][FEAT_ENG].values
            y_va = df_char.iloc[va_idx][col].values

            m = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=5,
                min_samples_leaf=15, random_state=42
            )
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)
            cv_preds[va_idx] = preds
            fold_maes.append(mean_absolute_error(y_va, preds))
            fold_r2s.append(r2_score(y_va, preds))

        y_true = df_char[col].values
        # Final model on all characterized data
        final_m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=5,
            min_samples_leaf=15, random_state=42
        )
        final_m.fit(df_char[FEAT_ENG].values, y_true)
        models[key] = final_m

        r2_cv  = r2_score(y_true, cv_preds)
        mae_cv = mean_absolute_error(y_true, cv_preds)

        extra = {}
        if key in CODE_TARGETS:
            extra["within_1_code"] = round(within_n_codes(y_true, cv_preds, 1), 4)

        results[key] = {
            "target":   col,
            "cv_r2":    round(r2_cv, 4),
            "cv_mae":   round(mae_cv, 4),
            "fold_r2s": [round(x, 4) for x in fold_r2s],
            **extra,
        }
        print(f"  Sensor {key} | RÂ²: {r2_cv:.3f} | MAE: {mae_cv:.4f}")

    # Hot-point prediction (85Â°C offset error reduction)
    hot_results = _hot_point_offset_error(df, df_char, models)
    results["hot_point_offset_85C"] = hot_results

    # Test insertion savings
    conv_time = df["conv_test_time_s"].mean() if "conv_test_time_s" in df.columns else 0
    ai_time   = conv_time / 3.0  # 1 insertion vs 3
    results["test_insertion_savings"] = {
        "conventional_insertions":  3,
        "ai_insertions":            1,
        "conv_test_time_s":         round(conv_time, 2),
        "ai_test_time_s":           round(ai_time, 2),
        "time_saved_pct":           round((1 - 1/3) * 100, 1),
        "characterized_train_pct":  round(len(df_char) / len(df) * 100, 1),
        "predicted_pct":            round(len(df_pred) / len(df) * 100, 1),
    }
    print(f"  Sensor time saved: {(1-1/3)*100:.1f}% (3 â†’ 1 insertion)")

    with open(os.path.join(MODELS_PATH, "sensor_calib_models.pkl"), "wb") as f:
        pickle.dump(models, f)

    return results


def _hot_point_offset_error(df_all, df_char, models):
    """Estimate 85Â°C offset error: conventional (no TC comp) vs AI (1-insertion)."""
    if "offset_x_85C_mg" not in df_char.columns or "tco_x_true_mg_per_C" not in df_char.columns:
        return {}

    # Conventional: no temperature compensation â†’ raw 85Â°C offset RMS
    conv_rms = float(np.sqrt(np.mean(df_char["offset_x_85C_mg"] ** 2)))

    # AI: predict TCO from 25Â°C data and compensate
    X     = df_char[FEAT_ENG].values
    tco_x = df_char["tco_x_true_mg_per_C"].values
    pred_tco = models["tco_x"].predict(X) if "tco_x" in models else tco_x
    delta_T  = 85 - 25  # degrees
    compensated = df_char["offset_x_85C_mg"].values - pred_tco * delta_T
    ai_rms   = float(np.sqrt(np.mean(compensated ** 2)))

    return {
        "conv_offset_85C_rms_mg": round(conv_rms, 3),
        "ai_offset_85C_rms_mg":   round(ai_rms, 3),
        "improvement_pct":        round((conv_rms - ai_rms) / conv_rms * 100, 1),
    }


if __name__ == "__main__":
    import json
    print("Running Sensor Calibration module...")
    results = run_sensor_module()
    print(json.dumps(results, indent=2))

