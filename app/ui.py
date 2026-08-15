"""
ui.py
=====
Streamlit web application for insurance claim prediction.
Accepts the 8 key feature inputs via widgets and returns:
  - The claim probability
  - The predicted label
  - The risk segment
  - Personalised marketing suggestion

Usage (from project root):
    streamlit run app/ui.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# ---------------------------------------------------------------------------
# Feature defaults (columns not exposed in the UI keep their median/mode value)
# ---------------------------------------------------------------------------
FEATURE_DEFAULTS = {
    "ps_ind_01":      2,
    "ps_ind_02_cat":  1,
    "ps_ind_03":      5,
    "ps_ind_04_cat":  0,
    "ps_ind_05_cat":  0,
    "ps_ind_06_bin":  0,
    "ps_ind_07_bin":  0,
    "ps_ind_08_bin":  0,
    "ps_ind_09_bin":  0,
    "ps_ind_10_bin":  0,
    "ps_ind_11_bin":  0,
    "ps_ind_12_bin":  0,
    "ps_ind_13_bin":  0,
    "ps_ind_14":      0,
    "ps_ind_15":      7,
    "ps_ind_16_bin":  1,
    "ps_ind_17_bin":  0,
    "ps_ind_18_bin":  0,
    "ps_reg_01":      0.7,
    "ps_reg_02":      0.2,
    "ps_reg_03":      0.7,
    "ps_car_01_cat":  7,
    "ps_car_02_cat":  1,
    "ps_car_04_cat":  0,
    "ps_car_06_cat":  1,
    "ps_car_07_cat":  1,
    "ps_car_08_cat":  1,
    "ps_car_09_cat":  0,
    "ps_car_10_cat":  1,
    "ps_car_11_cat": 12,
    "ps_car_11":      2,
    "ps_car_12":      0.316,
    "ps_car_13":      0.641,
    "ps_car_14":      0.374,
    "ps_car_15":      3.606,
}

# Risk thresholds and marketing copy (unchanged from CLI version)
RISK_ADVICE = {
    "Low": (
        "Low probability of filing a claim.",
        "> Offer standard coverage at competitive rates. "
        "Consider loyalty rewards to retain this low-risk customer.",
    ),
    "Medium": (
        "Moderate probability of filing a claim.",
        "-> Offer comprehensive coverage. Highlight value-add features "
        "like roadside assistance. Consider a small premium adjustment.",
    ),
    "High": (
        "High probability of filing a claim.",
        "! Recommend higher-tier coverage with clear excess explanation. "
        "Use risk-based pricing. Offer telematics/driving-score discounts "
        "to incentivise safer behaviour.",
    ),
    "Very High": (
        "Very high probability of filing a claim.",
        "X Flag for underwriting review. If offering a policy, apply "
        "appropriate premium loading. Consider value-for-money bundled "
        "packages to offset higher perceived cost.",
    ),
}

SEGMENT_COLORS = {
    "Low":       "🟢",
    "Medium":    "🟡",
    "High":      "🟠",
    "Very High": "🔴",
}


# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model not found at `{MODEL_PATH}`. "
            "Please run `python src/train.py` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Prediction logic (unchanged from CLI version)
# ---------------------------------------------------------------------------
def predict_single(artifact, feature_dict):
    model         = artifact["model"]
    feature_names = artifact["feature_names"]

    row = {col: [feature_dict.get(col, np.nan)] for col in feature_names}
    df  = pd.DataFrame(row)

    proba = model.predict_proba(df)[0, 1]
    label = int(proba >= 0.5)

    if proba < 0.03:
        segment = "Low"
    elif proba < 0.07:
        segment = "Medium"
    elif proba < 0.15:
        segment = "High"
    else:
        segment = "Very High"

    return proba, label, segment


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PRCP-1010 Insurance Claim Prediction",
        page_icon="🛡️",
        layout="centered",
    )

    st.title("🛡️ PRCP-1010 Insurance Claim Prediction")
    st.markdown("Enter customer details below and click **Predict** to assess claim risk.")

    artifact = load_model()
    st.caption(
        f"Model: **{artifact['model_name']}** &nbsp;|&nbsp; "
        f"Val ROC-AUC: **{artifact['metrics']['roc_auc']:.4f}**"
    )

    st.divider()

    # ---- Input widgets -------------------------------------------------------
    st.subheader("Customer Features")

    col1, col2 = st.columns(2)

    with col1:
        ps_ind_15 = st.slider(
            "Driver age band",
            min_value=0, max_value=13,
            value=int(FEATURE_DEFAULTS["ps_ind_15"]),
            help="Integer 0–13; higher value = older driver",
        )
        ps_reg_01 = st.slider(
            "Region risk score",
            min_value=0.0, max_value=0.9,
            value=float(FEATURE_DEFAULTS["ps_reg_01"]),
            step=0.1,
            help="Float 0.0–0.9; higher = riskier region",
        )
        ps_car_13 = st.number_input(
            "Vehicle value index",
            min_value=0.0, max_value=4.0,
            value=float(FEATURE_DEFAULTS["ps_car_13"]),
            step=0.01,
            format="%.3f",
            help="Float, typical range 0.2–3.7",
        )
        ps_car_12 = st.number_input(
            "Vehicle age index",
            min_value=0.0, max_value=1.5,
            value=float(FEATURE_DEFAULTS["ps_car_12"]),
            step=0.01,
            format="%.3f",
            help="Float, typical range 0.2–0.7",
        )

    with col2:
        ps_ind_03 = st.slider(
            "Policy tenure (years)",
            min_value=0, max_value=11,
            value=int(FEATURE_DEFAULTS["ps_ind_03"]),
            help="Integer 0–11",
        )
        ps_ind_17_bin = st.selectbox(
            "Optional benefit A",
            options=[0, 1],
            index=int(FEATURE_DEFAULTS["ps_ind_17_bin"]),
            format_func=lambda x: "Yes (1)" if x else "No (0)",
            help="Whether the customer has optional benefit A",
        )
        ps_ind_16_bin = st.selectbox(
            "Optional benefit B",
            options=[0, 1],
            index=int(FEATURE_DEFAULTS["ps_ind_16_bin"]),
            format_func=lambda x: "Yes (1)" if x else "No (0)",
            help="Whether the customer has optional benefit B",
        )
        ps_car_15 = st.number_input(
            "Vehicle power score",
            min_value=0.0, max_value=5.5,
            value=float(FEATURE_DEFAULTS["ps_car_15"]),
            step=0.01,
            format="%.3f",
            help="Float, typical range 0.0–5.0",
        )

    st.divider()

    # ---- Predict button ------------------------------------------------------
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        feature_dict = FEATURE_DEFAULTS.copy()
        feature_dict.update({
            "ps_ind_15":    ps_ind_15,
            "ps_reg_01":    ps_reg_01,
            "ps_car_13":    ps_car_13,
            "ps_car_12":    ps_car_12,
            "ps_ind_03":    ps_ind_03,
            "ps_ind_17_bin": ps_ind_17_bin,
            "ps_ind_16_bin": ps_ind_16_bin,
            "ps_car_15":    ps_car_15,
        })

        proba, label, segment = predict_single(artifact, feature_dict)
        assessment, marketing = RISK_ADVICE[segment]
        icon = SEGMENT_COLORS[segment]

        st.subheader("Prediction Result")

        # Metric cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Claim Probability", f"{proba * 100:.2f}%")
        m2.metric(
            "Predicted Label",
            "Will Claim (1)" if label else "Unlikely (0)",
        )
        m3.metric("Risk Segment", f"{icon} {segment}")

        # Assessment & marketing
        st.info(f"**Assessment:** {assessment}")
        st.warning(f"**Marketing Recommendation:** {marketing}")


if __name__ == "__main__":
    main()
