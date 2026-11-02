"""
Fuse-Map Validation Engine
===========================
Validates MCU fuse maps against the derivative schema for:
  - wrong flash size for derivative
  - peripheral mask mismatch
  - missing debug lock bit (must be 1 in production)
  - reserved bits set (must be 0)
  - RDP missing on automotive (MCU-C must have RDP=1)
  - trim field corruption (out of range)
Reports precision, recall and F1 against ground-truth error labels.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_AUDIT  = os.path.join(os.path.dirname(__file__), "..", "data", "mcu_fuse_map_audit.csv")
DATA_SCHEMA = os.path.join(os.path.dirname(__file__), "..", "data", "mcu_fuse_schema.csv")

# Expected values per derivative from schema
FLASH_SIZE_EXPECTED  = {"MCU-A": 1, "MCU-B": 0, "MCU-C": 2}
PERIPH_EN_EXPECTED   = {"MCU-A": 0xFF, "MCU-B": 0x3F, "MCU-C": 0xFF}
TEMP_GRADE_EXPECTED  = {"MCU-A": 0, "MCU-B": 0, "MCU-C": 1}


def _parse_hex(h):
    try:
        return int(str(h).strip(), 16)
    except (ValueError, TypeError):
        return None


def _get_field(fuse_int: int, lsb: int, nbits: int) -> int:
    mask = (1 << nbits) - 1
    return (fuse_int >> lsb) & mask


def _check_fuse(fuse_int, derivative):
    """Apply schema rules. Return True if any rule violation detected."""
    if fuse_int is None:
        return True  # parse failure

    errors = []

    # Field positions from schema
    bg_trim     = _get_field(fuse_int,  0,  6)
    hsi_trim    = _get_field(fuse_int,  6,  7)
    ldo_trim    = _get_field(fuse_int, 13,  5)
    ts_cal1     = _get_field(fuse_int, 18, 12)
    flash_size  = _get_field(fuse_int, 30,  2)
    periph_en   = _get_field(fuse_int, 32,  8)
    temp_grade  = _get_field(fuse_int, 40,  1)
    debug_lock  = _get_field(fuse_int, 41,  1)
    rdp         = _get_field(fuse_int, 42,  1)
    reserved    = _get_field(fuse_int, 43, 21)

    # Rule 1: wrong flash size
    expected_flash = FLASH_SIZE_EXPECTED.get(derivative, -1)
    if expected_flash != -1 and flash_size != expected_flash:
        errors.append("wrong_flash_size")

    # Rule 2: peripheral mask mismatch
    expected_periph = PERIPH_EN_EXPECTED.get(derivative, -1)
    if expected_periph != -1 and periph_en != expected_periph:
        errors.append("periph_mask_mismatch")

    # Rule 3: missing debug lock (must be 1 in production)
    if debug_lock == 0:
        errors.append("missing_debug_lock")

    # Rule 4: reserved bits must be 0
    if reserved != 0:
        errors.append("reserved_bit_set")

    # Rule 5: RDP missing on automotive (MCU-C must have RDP=1)
    if derivative == "MCU-C" and rdp == 0:
        errors.append("rdp_missing_automotive")

    # Rule 6: trim field corruption — codes far outside valid range
    # BG: valid 0-63, nominal ~29-35; HSI: valid 0-127, nominal ~56-71; LDO: valid 0-31, nominal ~14-18
    if not (10 <= bg_trim <= 55):
        errors.append("trim_field_corrupt")
    if not (40 <= hsi_trim <= 90):
        errors.append("trim_field_corrupt")
    if not (8 <= ldo_trim <= 24):
        errors.append("trim_field_corrupt")

    return len(errors) > 0, errors


def run_fuse_validation_module():
    df_audit  = pd.read_csv(DATA_AUDIT)
    df_schema = pd.read_csv(DATA_SCHEMA)

    df_audit["fuse_int"] = df_audit["fuse_map_hex"].apply(_parse_hex)

    detected_list  = []
    detected_types = []

    for _, row in df_audit.iterrows():
        fuse_int   = row["fuse_int"]
        derivative = row.get("derivative", "MCU-A")
        result, errs = _check_fuse(fuse_int, derivative)
        detected_list.append(result)
        detected_types.append(errs)

    df_audit["detected"] = detected_list

    # Metrics against ground truth
    if "has_error" in df_audit.columns:
        y_true = df_audit["has_error"].astype(bool)
        y_pred = df_audit["detected"].astype(bool)

        tp = int((y_true & y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        n_injected = int(y_true.sum())
    else:
        tp = tn = fp = fn = n_injected = 0
        precision = recall = f1 = 0.0

    # Error type breakdown from ground truth
    if "error_type" in df_audit.columns:
        error_type_counts = (
            df_audit[df_audit["has_error"] == True]["error_type"]
            .value_counts().to_dict()
        )
    else:
        error_type_counts = {}

    # Per-type detection rates
    per_type = {}
    for etype, total in error_type_counts.items():
        mask    = (df_audit["error_type"] == etype) & (df_audit["has_error"] == True)
        caught  = int((df_audit.loc[mask, "detected"]).sum())
        per_type[etype] = {"total": total, "caught": caught,
                           "detection_rate": round(caught / total, 3) if total > 0 else 0.0}

    print(f"  Fuse-map | injected errors: {n_injected} | "
          f"detected: {int(df_audit['detected'].sum())} | "
          f"precision: {precision*100:.1f}% | recall: {recall*100:.1f}% | F1: {f1:.3f}")

    return {
        "total_fuse_maps":      len(df_audit),
        "injected_errors":      n_injected,
        "errors_detected":      int(df_audit["detected"].sum()),
        "schema_fields":        len(df_schema),
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "f1":                   round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "error_type_breakdown": {k: int(v) for k, v in error_type_counts.items()},
        "per_type_detection":   per_type,
    }


if __name__ == "__main__":
    import json
    print("Running Fuse-Map Validation module...")
    results = run_fuse_validation_module()
    print(json.dumps(results, indent=2))
