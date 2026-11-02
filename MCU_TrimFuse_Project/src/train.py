"""
Master Training Pipeline
========================
Runs all 6 AI modules in sequence, collects results, and saves the
consolidated metrics summary to reports/metrics_summary.json.

Usage:
    python src/train.py
"""

import json
import os
import sys
import time

# Ensure src directory is on path
sys.path.insert(0, os.path.dirname(__file__))

from mcu_trim       import run_mcu_module
from pmic_trim      import run_pmic_module
from sensor_calib   import run_sensor_module
from soc_binning    import run_soc_module
from fuse_validation import run_fuse_validation_module
from memory_repair  import run_memory_repair_module

REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports")


def main():
    os.makedirs(REPORTS_PATH, exist_ok=True)
    summary = {}
    t0 = time.time()

    print("=" * 70)
    print("  AI Trim & Fuse â€” Full Training Pipeline")
    print("=" * 70)

    print("\n[1/6] MCU Predictive Trim (BG, HSI, LDO) + Transfer Learning")
    t = time.time()
    summary["mcu_trim"] = run_mcu_module()
    summary["mcu_trim"]["runtime_s"] = round(time.time() - t, 1)

    print("\n[2/6] PMIC Multi-Rail Trim + OTP Config Validation")
    t = time.time()
    summary["pmic_trim"] = run_pmic_module()
    summary["pmic_trim"]["runtime_s"] = round(time.time() - t, 1)

    print("\n[3/6] Sensor Single-Insertion Calibration (TCO/TCS prediction)")
    t = time.time()
    summary["sensor_calib"] = run_sensor_module()
    summary["sensor_calib"]["runtime_s"] = round(time.time() - t, 1)

    print("\n[4/6] SoC Fmax/Vmin Prediction + SKU Harvesting Optimization")
    t = time.time()
    summary["soc_binning"] = run_soc_module()
    summary["soc_binning"]["runtime_s"] = round(time.time() - t, 1)

    print("\n[5/6] Fuse-Map Schema Validation")
    t = time.time()
    summary["fuse_validation"] = run_fuse_validation_module()
    summary["fuse_validation"]["runtime_s"] = round(time.time() - t, 1)

    print("\n[6/6] Memory Repair Optimization (Greedy vs Exact Solver)")
    t = time.time()
    summary["memory_repair"] = run_memory_repair_module()
    summary["memory_repair"]["runtime_s"] = round(time.time() - t, 1)

    summary["total_runtime_s"] = round(time.time() - t0, 1)

    out_path = os.path.join(REPORTS_PATH, "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print_summary(summary)
    print(f"\nTotal pipeline runtime: {summary['total_runtime_s']}s")
    print(f"Full results saved to: {out_path}")
    return summary


def print_summary(summary):
    mcu = summary.get("mcu_trim", {})
    print(f"\n  MCU Trim (MCU-A, GroupKFold 5-fold):")
    for key in ["bg", "hsi", "ldo"]:
        r = mcu.get(key, {})
        if r:
            print(f"    {key.upper()} | within Â+/-1: {r.get('within_1_code', 0)*100:.1f}% | "
                  f"MAE: {r.get('cv_mae', 0):.3f} | RÂ²: {r.get('cv_r2', 0):.3f} | "
                  f"iter {r.get('baseline_iter', 0):.1f} â†’ {r.get('ai_iter', 0):.2f} "
                  f"({r.get('iter_reduction', 0):.0f}% reduction)")

    drift = mcu.get("drift_transfer", {})
    if drift:
        print(f"  MCU-C Drift Transfer:")
        print(f"    zero-shot MAE: {drift.get('mae_zero_shot_mV', 0):.2f} mV â†’ "
              f"fine-tuned MAE: {drift.get('mae_finetuned_mV', 0):.2f} mV")
        print(f"    BG final pass: {drift.get('bg_final_pass_naive_pct', 0):.1f}% â†’ "
              f"{drift.get('bg_final_pass_ft_pct', 0):.1f}%")

    pmic = summary.get("pmic_trim", {})
    print(f"\n  PMIC Rail Trim (PMIC-P1, GroupKFold 5-fold):")
    for key in ["buck1", "buck2", "ldo1", "ldo2"]:
        r = pmic.get(key, {})
        if r:
            print(f"    {key.upper()} | within Â+/-1: {r.get('within_1_code', 0)*100:.1f}% | "
                  f"iter {r.get('baseline_iter', 0):.1f} â†’ {r.get('ai_iter', 0):.2f}")
    otp = pmic.get("otp_validation", {})
    if otp:
        print(f"    OTP Validation | precision: {otp.get('precision', 0)*100:.1f}% | "
              f"recall: {otp.get('recall', 0)*100:.1f}%")

    sensor = summary.get("sensor_calib", {})
    print(f"\n  Sensor Calibration (single-insertion, 3-axis):")
    for key in ["tco_x", "tcs_x", "tco_y", "tcs_y", "tco_z", "tcs_z"]:
        r = sensor.get(key, {})
        if r:
            print(f"    {key} | RÂ²: {r.get('cv_r2', 0):.3f} | MAE: {r.get('cv_mae', 0):.4f}")
    ts = sensor.get("test_insertion_savings", {})
    if ts:
        print(f"    Test time: {ts.get('conv_test_time_s', 0):.1f}s â†’ "
              f"{ts.get('ai_test_time_s', 0):.1f}s ({ts.get('time_saved_pct', 0):.0f}% saved)")

    soc = summary.get("soc_binning", {})
    print(f"\n  SoC Fmax/Vmin Prediction:")
    for key in ["fmax_min", "vmin_max"]:
        r = soc.get(key, {})
        if r:
            print(f"    {key} | RÂ²: {r.get('cv_r2', 0):.3f} | MAE: {r.get('cv_mae', 0):.4f}")
    sku = soc.get("sku_assignment", {})
    if sku:
        print(f"    SKU revenue uplift: {sku.get('revenue_uplift_pct', 0):.1f}%")

    fuse = summary.get("fuse_validation", {})
    if fuse:
        print(f"\n  Fuse-Map Validation:")
        print(f"    {fuse.get('injected_errors', 0)} injected errors | "
              f"{fuse.get('errors_detected', 0)} detected | "
              f"precision: {fuse.get('precision', 0)*100:.1f}% | "
              f"recall: {fuse.get('recall', 0)*100:.1f}% | "
              f"F1: {fuse.get('f1', 0):.3f}")

    mem = summary.get("memory_repair", {})
    if mem:
        print(f"\n  Memory Repair:")
        print(f"    greedy: {mem.get('greedy_repairable_pct', 0):.1f}% | "
              f"exact: {mem.get('optimal_repairable_pct', 0):.1f}% | "
              f"arrays saved: {mem.get('arrays_saved_vs_greedy', 0)}")


if __name__ == "__main__":
    main()

