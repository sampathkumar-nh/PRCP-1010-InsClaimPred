"""
preprocess.py
=============
Loads train.csv, cleans missing values (encoded as -1),
drops low-quality columns, and returns a ready-to-split
feature matrix X and target vector y.

Design decisions (based on inspection):
  - ps_car_03_cat : 69 % missing  -> drop
  - ps_car_05_cat : 45 % missing  -> drop
  - ps_calc_*     : calculated noise features -> drop
  - Remaining -1  : replaced with NaN then imputed
                    (median for continuous, mode for categorical)
  - id            : dropped (not a feature)
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -- Column lists -------------------------------------------------------------

# Columns to drop outright
COLS_HIGH_MISSING = ["ps_car_03_cat", "ps_car_05_cat"]
COLS_CALC = [f"ps_calc_{i:02d}" for i in range(1, 4)] + \
            ["ps_calc_04", "ps_calc_05", "ps_calc_06", "ps_calc_07",
             "ps_calc_08", "ps_calc_09", "ps_calc_10", "ps_calc_11",
             "ps_calc_12", "ps_calc_13", "ps_calc_14",
             "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
             "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"]

COLS_DROP = ["id"] + COLS_HIGH_MISSING + COLS_CALC

# Remaining categorical columns (suffix _cat)
CAT_COLS = [
    "ps_ind_02_cat", "ps_ind_04_cat", "ps_ind_05_cat",
    "ps_car_01_cat", "ps_car_02_cat", "ps_car_04_cat",
    "ps_car_06_cat", "ps_car_07_cat", "ps_car_08_cat",
    "ps_car_09_cat", "ps_car_10_cat", "ps_car_11_cat",
]

# Continuous columns that had -1 missing
CONT_MISSING_COLS = ["ps_reg_03", "ps_car_14"]

# All other numeric columns (binary + integer + float)
# These will be determined dynamically after dropping above


def load_raw(data_path: str) -> pd.DataFrame:
    """Read CSV; keep all columns for now."""
    df = pd.read_csv(data_path)
    return df


def replace_minus_one(df: pd.DataFrame) -> pd.DataFrame:
    """Replace -1 sentinel with NaN."""
    df = df.replace(-1, np.nan)
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop id, high-missing, and calc noise columns."""
    cols_to_drop = [c for c in COLS_DROP if c in df.columns]
    return df.drop(columns=cols_to_drop)


def get_column_groups(df: pd.DataFrame, target: str = "target"):
    """
    Dynamically determine categorical vs continuous columns
    from the remaining DataFrame after drops.
    Returns (cat_cols, cont_cols).
    """
    feature_cols = [c for c in df.columns if c != target]
    cat_cols  = [c for c in CAT_COLS if c in feature_cols]
    cont_cols = [c for c in feature_cols if c not in cat_cols]
    return cat_cols, cont_cols


def build_preprocessor(cat_cols, cont_cols, scale: bool = False):
    """
    Build a ColumnTransformer that:
      - Imputes categoricals with most-frequent (mode)
      - Imputes continuous with median
      - Optionally scales continuous features (for Logistic Regression)
    Returns an unfitted ColumnTransformer.
    """
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    if scale:
        cont_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        cont_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat",  cat_pipeline,  cat_cols),
        ("cont", cont_pipeline, cont_cols),
    ], remainder="passthrough")

    return preprocessor


def load_and_prepare(data_path: str, target_col: str = "target"):
    """
    Full pipeline: load -> replace -1 -> drop columns -> split X/y.
    Returns (X, y, cat_cols, cont_cols).
    Does NOT fit any imputer — keeps raw NaNs for downstream Pipeline.
    """
    print(f"[preprocess] Loading data from: {data_path}")
    df = load_raw(data_path)
    print(f"[preprocess] Raw shape: {df.shape}")

    print("[preprocess] Replacing -1 sentinel values with NaN ...")
    df = replace_minus_one(df)

    print("[preprocess] Dropping low-quality and noise columns ...")
    df = drop_columns(df)
    print(f"[preprocess] Shape after drops: {df.shape}")

    y = df[target_col].values
    X = df.drop(columns=[target_col])

    cat_cols, cont_cols = get_column_groups(df, target=target_col)

    print(f"[preprocess] Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"[preprocess] Continuous columns  ({len(cont_cols)}): {cont_cols}")
    print(f"[preprocess] Target distribution: "
          f"0={int((y==0).sum())}  1={int((y==1).sum())}  "
          f"imbalance_ratio={round((y==0).sum()/(y==1).sum(),1)}:1")

    return X, y, cat_cols, cont_cols


def get_feature_names(preprocessor, cat_cols, cont_cols):
    """Return ordered list of feature names after ColumnTransformer."""
    return cat_cols + cont_cols


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
    X, y, cat_cols, cont_cols = load_and_prepare(DATA_PATH)
    print("\nFeature matrix shape:", X.shape)
    print("First 3 rows:\n", X.head(3))
