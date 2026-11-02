"""
PMIC Multi-Rail Trim Engine
============================
Predicts trim codes for all PMIC rails (BUCK1, BUCK2, LDO1, LDO2) and
threshold trims (OVP, UVLO, ILIM) from reference measurement + PCM only.
Also validates OTP configuration requests for design rule violations.
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

DATA_TRIM = os.path.join(os.path.dirname(__file__), "..", "data", "pmic_trim_otp.csv")
DATA_OTP  = os.path.join(os.path.dirname(__file__), "..", "data", "pmic_otp_config_requests.csv")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

PCM_FEATURES = [
    "pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq", "pcm_tox_A", "pcm_mobility_rel",
    "x_mm", "y_mm", "radius_mm",
    "vref_untrim_V",   # reference measurement â€” the key input
]

RAIL_TARGETS = {
    "buck1": "buck1_trim_code_opt",
    "buck2": "buck2_trim_code_opt",
    "ldo1":  "ldo1_trim_code_opt",
    "ldo2":  "ldo2_trim_code_opt",
}

THRESHOLD_TARGETS = {
    "ovp":  "ovp_trim_code_opt",
    "uvlo": "uvlo_trim_code_opt",
    "ilim": "ilim_trim_code_opt",
}

ITER_BASELINE_COL = "conv_trim_iterations"


def engineer_pmic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["radius_sq"]      = df["radius_mm"] ** 2
    df["vth_product"]    = df["pcm_vth_n_mV"] * df["pcm_vth_p_mV"].abs()
    df["vref_error"]     = df["vref_untrim_V"] - 1.200
    df["process_spread"] = df["pcm_vth_n_mV"] - df["pcm_vth_p_mV"].abs()
    return df


FEAT_ENG = PCM_FEATURES + ["radius_sq", "vth_product", "vref_error", "process_spread"]


def within_n_codes(y_true, y_pred, n=1):
    return float(np.mean(np.abs(np.round(y_pred) - y_true) <= n))


def ai_iterations(y_true, y_pred):
    exact = np.abs(np.round(y_pred) - y_true) == 0
    off1  = np.abs(np.round(y_pred) - y_true) == 1
    return float(np.where(exact, 1, np.where(off1, 2, 3)).mean())


def run_pmic_module():
    os.makedirs(MODELS_PATH, exist_ok=True)
    df = pd.read_csv(DATA_TRIM)
    df = engineer_pmic_features(df)

    # Use PMIC-P1 (base derivative) for training
    df_p1 = df[df["derivative"] == "PMIC-P1"].copy()
    gkf   = GroupKFold(n_splits=5)
    groups = df_p1["wafer_id"].values

    results  = {}
    models   = {}
    baseline_iter = df_p1[ITER_BASELINE_COL].mean()

    all_targets = {**RAIL_TARGETS, **THRESHOLD_TARGETS}

    for key, col in all_targets.items():
        fold_maes, fold_w1 = [], []
        cv_preds = np.empty(len(df_p1))

        for tr_idx, va_idx in gkf.split(df_p1, groups=groups):
            X_tr = df_p1.iloc[tr_idx][FEAT_ENG].values
            y_tr = df_p1.iloc[tr_idx][col].values
            X_va = df_p1.iloc[va_idx][FEAT_ENG].values
            y_va = df_p1.iloc[va_idx][col].values

            m = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=5,
                min_samples_leaf=20, random_state=42
            )
            m.fit(X_tr, y_tr)
            preds = m.predict(X_va)
            cv_preds[va_idx] = preds
            fold_maes.append(mean_absolute_error(y_va, preds))
            fold_w1.append(within_n_codes(y_va, preds))

        # Final model on all P1
        final_m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, random_state=42
        )
        final_m.fit(df_p1[FEAT_ENG].values, df_p1[col].values)
        models[key] = final_m

        y_true = df_p1[col].values
        ai_it  = ai_iterations(y_true, cv_preds)

        results[key] = {
            "target":        col,
            "cv_mae":        round(np.mean(fold_maes), 4),
            "cv_r2":         round(r2_score(y_true, cv_preds), 4),
            "within_1_code": round(np.mean(fold_w1), 4),
            "fold_maes":     [round(x, 4) for x in fold_maes],
            "baseline_iter": round(baseline_iter, 2),
            "ai_iter":       round(ai_it, 3),
            "iter_reduction": round((baseline_iter - ai_it) / baseline_iter * 100, 1),
        }
        print(f"  PMIC {key.upper()} | within Â+/-1: {np.mean(fold_w1)*100:.1f}% | "
              f"iter {baseline_iter:.1f} â†’ {ai_it:.2f}")

    # OTP config validation
    otp_results = _run_otp_validation()
    results["otp_validation"] = otp_results

    with open(os.path.join(MODELS_PATH, "pmic_trim_models.pkl"), "wb") as f:
        pickle.dump(models, f)

    return results


def _run_otp_validation():
    """Validate PMIC OTP configuration requests for design rule violations."""
    df = pd.read_csv(DATA_OTP)

    VALID_RAILS   = {"buck1_V": (0.5, 2.0), "buck2_V": (0.5, 2.0),
                     "ldo1_V":  (1.5, 3.6), "ldo2_V":  (1.5, 3.6)}
    RESERVED_ADDRS = {0x00, 0x7F, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E}
    VALID_RAILS_SET = {"buck1", "buck2", "ldo1", "ldo2"}

    def check_request(row):
        errs = []
        for col, (lo, hi) in VALID_RAILS.items():
            v = row.get(col, np.nan)
            if pd.notna(v) and not (lo <= v <= hi):
                errs.append("out_of_range")
        try:
            addr = int(str(row["i2c_addr"]).strip(), 16)
            if addr in RESERVED_ADDRS:
                errs.append("address_reserved")
            others = [int(x.strip(), 16) for x in str(row["other_addresses_on_bus"]).split(";") if x.strip()]
            if addr in others:
                errs.append("address_collision")
        except Exception:
            pass
        try:
            seq_parts = [s.strip() for s in str(row["sequence"]).replace(">", ",").split(",")]
            if len(seq_parts) != len(set(seq_parts)):
                errs.append("invalid_sequence")
            if not set(seq_parts).issubset(VALID_RAILS_SET):
                errs.append("sequence_conflict")
        except Exception:
            errs.append("invalid_sequence")
        return len(errs) > 0

    df["any_error_detected"] = df.apply(check_request, axis=1)

    if "label" in df.columns:
        y_true = df["label"] != "valid"
        y_pred = df["any_error_detected"]
        tp = int((y_true & y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        n_invalid = int(y_true.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        tp = tn = fp = fn = n_invalid = 0
        precision = recall = 0.0

    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    detected = int(df["any_error_detected"].sum())
    print(f"  PMIC OTP | invalid configs: {n_invalid} | detected: {detected} | "
          f"precision: {precision*100:.1f}% | recall: {recall*100:.1f}% | F1: {f1:.3f}")

    return {
        "total_requests":       len(df),
        "invalid_ground_truth": n_invalid,
        "errors_detected":      detected,
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "f1":                   round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


if __name__ == "__main__":
    import json
    print("Running PMIC Trim module...")
    results = run_pmic_module()
    print(json.dumps(results, indent=2))

