"""
AI Trim & Fuse — Streamlit Dashboard
======================================
Interactive dashboard covering all 6 AI modules:
  1. MCU Predictive Trim (BG, HSI, LDO)
  2. PMIC Multi-Rail Trim
  3. Sensor Single-Insertion Calibration
  4. SoC Fmax/Vmin + SKU Harvesting
  5. Fuse-Map Validation
  6. Memory Repair Optimization

Run: streamlit run app.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Ensure src is importable
SRC_DIR  = os.path.join(os.path.dirname(__file__), "src")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RPT_DIR  = os.path.join(os.path.dirname(__file__), "reports")
sys.path.insert(0, SRC_DIR)

st.set_page_config(
    page_title="AI Trim & Fuse Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 AI Trim & Fuse")
st.sidebar.markdown("**Semiconductor Testing — AI Pipeline**")
st.sidebar.markdown("---")

MODULE = st.sidebar.radio(
    "Select Module",
    [
        "📊 Overview & Results",
        "🔧 MCU Predictive Trim",
        "⚡ PMIC Multi-Rail Trim",
        "🌡️ Sensor Calibration",
        "💻 SoC Binning & Harvesting",
        "🔒 Fuse-Map Validation",
        "🔩 Memory Repair",
    ],
)

st.sidebar.markdown("---")

# Run pipeline button
if st.sidebar.button("▶ Run Full Pipeline", type="primary"):
    with st.spinner("Running all 6 AI modules... (~60s)"):
        try:
            from train import main as run_train
            summary = run_train()
            st.sidebar.success("Pipeline complete!")
            st.session_state["summary"] = summary
        except Exception as e:
            st.sidebar.error(f"Error: {e}")


def load_summary():
    path = os.path.join(RPT_DIR, "metrics_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


# ── OVERVIEW ─────────────────────────────────────────────────────────────────
if MODULE == "📊 Overview & Results":
    st.title("🔬 AI-Optimized Trim & Fuse for Derivative Chips")
    st.markdown("""
    This project implements the full AI pipeline described in the requirement document
    *AI-Optimized Trim & Fuse for Derivative Chips*, covering all four chip types:
    **MCU · PMIC · Sensor · SoC**.
    """)

    summary = load_summary()
    if summary is None:
        st.warning("No results yet. Click **Run Full Pipeline** in the sidebar.")
        st.stop()

    st.markdown("### Pipeline Results Summary")
    col1, col2, col3, col4 = st.columns(4)

    mcu = summary.get("mcu_trim", {})
    bg  = mcu.get("bg", {})
    hsi = mcu.get("hsi", {})
    ldo = mcu.get("ldo", {})
    col1.metric("MCU BG within ±1 code",  f"{bg.get('within_1_code',0)*100:.1f}%",
                f"iter {bg.get('baseline_iter',0):.1f} → {bg.get('ai_iter',0):.1f}")
    col2.metric("MCU HSI within ±1 code", f"{hsi.get('within_1_code',0)*100:.1f}%",
                f"iter {hsi.get('baseline_iter',0):.1f} → {hsi.get('ai_iter',0):.1f}")
    col3.metric("MCU LDO within ±1 code", f"{ldo.get('within_1_code',0)*100:.1f}%",
                f"iter {ldo.get('baseline_iter',0):.1f} → {ldo.get('ai_iter',0):.1f}")
    drift = mcu.get("drift_transfer", {})
    col4.metric("MCU-C Transfer Learning",
                f"MAE {drift.get('mae_finetuned_mV',0):.2f} mV",
                f"vs zero-shot {drift.get('mae_zero_shot_mV',0):.2f} mV")

    col1, col2, col3, col4 = st.columns(4)
    pmic   = summary.get("pmic_trim", {})
    sensor = summary.get("sensor_calib", {})
    soc    = summary.get("soc_binning", {})
    fuse   = summary.get("fuse_validation", {})
    mem    = summary.get("memory_repair", {})

    b1 = pmic.get("buck1", {})
    col1.metric("PMIC BUCK1 within ±1",  f"{b1.get('within_1_code',0)*100:.1f}%",
                f"iter {b1.get('baseline_iter',0):.0f} → {b1.get('ai_iter',0):.1f}")
    ts = sensor.get("test_insertion_savings", {})
    col2.metric("Sensor Test Time Saved", f"{ts.get('time_saved_pct',0):.0f}%",
                f"{ts.get('conv_test_time_s',0):.1f}s → {ts.get('ai_test_time_s',0):.1f}s")
    fm = soc.get("fmax_min", {})
    col3.metric("SoC Fmax Prediction R²", f"{fm.get('cv_r2',0):.3f}",
                f"MAE {fm.get('cv_mae',0):.3f} GHz")
    col4.metric("Fuse-Map Recall",
                f"{fuse.get('recall',0)*100:.1f}%",
                f"F1 = {fuse.get('f1',0):.3f}")

    st.markdown("---")
    st.markdown("### Detailed Metrics Table")
    rows = []
    for key, label in [("bg","MCU BG Trim"), ("hsi","MCU HSI Trim"), ("ldo","MCU LDO Trim")]:
        r = mcu.get(key, {})
        rows.append({"Module": label, "Key Metric": f"within ±1: {r.get('within_1_code',0)*100:.1f}%",
                     "R² / F1": f"{r.get('cv_r2',0):.3f}",
                     "Iter Reduction": f"{r.get('iter_reduction',0):.0f}%"})
    for key, label in [("buck1","PMIC BUCK1"), ("buck2","PMIC BUCK2"),
                        ("ldo1","PMIC LDO1"), ("ldo2","PMIC LDO2")]:
        r = pmic.get(key, {})
        rows.append({"Module": label, "Key Metric": f"within ±1: {r.get('within_1_code',0)*100:.1f}%",
                     "R² / F1": f"{r.get('cv_r2',0):.3f}",
                     "Iter Reduction": f"{r.get('iter_reduction',0):.0f}%"})
    for key, label in [("tco_x","Sensor TCO-X"), ("tcs_x","Sensor TCS-X")]:
        r = sensor.get(key, {})
        rows.append({"Module": label, "Key Metric": f"R²: {r.get('cv_r2',0):.3f}",
                     "R² / F1": f"{r.get('cv_r2',0):.3f}", "Iter Reduction": "3→1 insertion"})
    for key, label in [("fmax_min","SoC Fmax"), ("vmin_max","SoC Vmin")]:
        r = soc.get(key, {})
        rows.append({"Module": label, "Key Metric": f"R²: {r.get('cv_r2',0):.3f}",
                     "R² / F1": f"{r.get('cv_r2',0):.3f}", "Iter Reduction": "speed test reduced"})
    rows.append({"Module": "Fuse Validation", "Key Metric": f"recall: {fuse.get('recall',0)*100:.1f}%",
                 "R² / F1": f"{fuse.get('f1',0):.3f}", "Iter Reduction": "0 false negatives on safety"})
    rows.append({"Module": "Memory Repair", "Key Metric": f"exact: {mem.get('optimal_repairable_pct',0):.1f}%",
                 "R² / F1": "—",
                 "Iter Reduction": f"+{mem.get('arrays_saved_vs_greedy',0)} arrays vs greedy"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ── MCU TRIM ─────────────────────────────────────────────────────────────────
elif MODULE == "🔧 MCU Predictive Trim":
    st.title("🔧 MCU Predictive Trim (BG · HSI · LDO)")
    st.markdown("""
    Predicts the optimal bandgap, HSI oscillator, and LDO trim codes for MCU-A/B/C
    from PCM measurements + wafer position. Replaces binary-search loops with a
    single-shot ML prediction + one verification measurement.
    """)

    try:
        df = load_csv("mcu_trim_fuse.csv")
        deriv = st.selectbox("Filter by Derivative", ["All"] + sorted(df["derivative"].unique()))
        if deriv != "All":
            df = df[df["derivative"] == deriv]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Bandgap Trim Distribution")
            hist_data = df["bg_trim_code_opt"].value_counts().sort_index()
            st.bar_chart(hist_data)
        with col2:
            st.markdown("#### HSI Trim Distribution")
            hist_data = df["hsi_trim_code_opt"].value_counts().sort_index()
            st.bar_chart(hist_data)

        st.markdown("#### Wafer Spatial Map — BG Trim Code")
        sample = df.sample(min(2000, len(df)), random_state=42)
        import plotly.express as px
        fig = px.scatter(sample, x="x_mm", y="y_mm", color="bg_trim_code_opt",
                         color_continuous_scale="RdYlGn",
                         title="Bandgap Trim Code vs Wafer Position",
                         labels={"x_mm": "X (mm)", "y_mm": "Y (mm)", "bg_trim_code_opt": "BG Code"})
        st.plotly_chart(fig, use_container_width=True)

        summary = load_summary()
        if summary:
            mcu = summary.get("mcu_trim", {})
            st.markdown("#### Cross-Validation Results")
            rows = []
            for key, label in [("bg","Bandgap"), ("hsi","HSI"), ("ldo","LDO")]:
                r = mcu.get(key, {})
                rows.append({
                    "Trim": label,
                    "Within ±1 Code": f"{r.get('within_1_code',0)*100:.1f}%",
                    "CV MAE": r.get("cv_mae", "—"),
                    "CV R²": r.get("cv_r2", "—"),
                    "Baseline Iter": r.get("baseline_iter", "—"),
                    "AI Iter": r.get("ai_iter", "—"),
                    "Reduction": f"{r.get('iter_reduction',0):.0f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            drift = mcu.get("drift_transfer", {})
            if drift:
                st.markdown("#### MCU-C Transfer Learning (Drift-Aware Targets)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Fine-tune dies", drift.get("n_finetune", 0))
                col2.metric("Zero-shot MAE", f"{drift.get('mae_zero_shot_mV',0):.2f} mV")
                col3.metric("Fine-tuned MAE", f"{drift.get('mae_finetuned_mV',0):.2f} mV",
                            f"-{drift.get('mae_zero_shot_mV',0) - drift.get('mae_finetuned_mV',0):.2f} mV")
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ── PMIC TRIM ─────────────────────────────────────────────────────────────────
elif MODULE == "⚡ PMIC Multi-Rail Trim":
    st.title("⚡ PMIC Multi-Rail Trim + OTP Validation")
    st.markdown("""
    Predicts all rail trim codes (BUCK1, BUCK2, LDO1, LDO2) and threshold trims
    (OVP, UVLO, ILIM) from a single reference measurement + PCM data only —
    without measuring each rail individually.
    """)

    try:
        df = load_csv("pmic_trim_otp.csv")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Reference vs BUCK1 Trim Code")
            sample = df.sample(min(1500, len(df)), random_state=42)
            import plotly.express as px
            fig = px.scatter(sample, x="vref_untrim_V", y="buck1_trim_code_opt",
                             color="derivative", title="BUCK1 Trim Code vs Reference Voltage")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Rail Trim Code Distributions")
            rail = st.selectbox("Rail", ["buck1_trim_code_opt","buck2_trim_code_opt",
                                          "ldo1_trim_code_opt","ldo2_trim_code_opt"])
            hist = df[rail].value_counts().sort_index()
            st.bar_chart(hist)

        summary = load_summary()
        if summary:
            pmic = summary.get("pmic_trim", {})
            st.markdown("#### Rail Trim CV Results")
            rows = []
            for key, label in [("buck1","BUCK1"),("buck2","BUCK2"),("ldo1","LDO1"),
                                ("ldo2","LDO2"),("ovp","OVP"),("uvlo","UVLO"),("ilim","ILIM")]:
                r = pmic.get(key, {})
                rows.append({
                    "Rail": label,
                    "Within ±1": f"{r.get('within_1_code',0)*100:.1f}%",
                    "MAE": r.get("cv_mae","—"),
                    "R²": r.get("cv_r2","—"),
                    "Iter Reduction": f"{r.get('iter_reduction',0):.0f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            otp = pmic.get("otp_validation", {})
            if otp:
                st.markdown("#### OTP Config Validation")
                col1, col2, col3 = st.columns(3)
                col1.metric("Precision", f"{otp.get('precision',0)*100:.1f}%")
                col2.metric("Recall",    f"{otp.get('recall',0)*100:.1f}%")
                col3.metric("F1 Score",  f"{otp.get('f1',0):.3f}")
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ── SENSOR ────────────────────────────────────────────────────────────────────
elif MODULE == "🌡️ Sensor Calibration":
    st.title("🌡️ Sensor Single-Insertion Calibration")
    st.markdown("""
    Predicts temperature coefficients (TCO, TCS) and calibration codes for
    3-axis MEMS accelerometers from **one room-temperature (25°C) insertion**
    instead of the conventional 3-temperature insertion, saving **67%** of test time.
    """)

    try:
        df = load_csv("sensor_calibration.csv")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### TCO-X vs PCM Vth")
            sample = df.sample(min(1500, len(df)), random_state=42)
            import plotly.express as px
            fig = px.scatter(sample, x="pcm_vth_n_mV", y="tco_x_true_mg_per_C",
                             color="derivative",
                             title="TCO-X vs NMOS Vth")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Calibration: 3-temp vs 1-insertion")
            char_pct = (df["characterized_3temp"] == True).mean() * 100
            pred_pct = 100 - char_pct
            pie_df = pd.DataFrame({"Type": ["3-temp characterized", "1-insertion (AI predicted)"],
                                   "Percent": [char_pct, pred_pct]})
            fig = px.pie(pie_df, values="Percent", names="Type",
                         title="Training vs Predicted Split")
            st.plotly_chart(fig, use_container_width=True)

        summary = load_summary()
        if summary:
            sensor = summary.get("sensor_calib", {})
            st.markdown("#### TC Prediction R² Results")
            rows = []
            for key in ["tco_x","tcs_x","tco_y","tcs_y","tco_z","tcs_z"]:
                r = sensor.get(key, {})
                rows.append({"Target": key, "R²": r.get("cv_r2","—"), "MAE": r.get("cv_mae","—")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            ts = sensor.get("test_insertion_savings", {})
            hp = sensor.get("hot_point_offset_85C", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Test Time Saved", f"{ts.get('time_saved_pct',0):.0f}%")
            col2.metric("Conv 85°C Offset RMS", f"{hp.get('conv_offset_85C_rms_mg',0):.1f} mg")
            col3.metric("AI 85°C Offset RMS",   f"{hp.get('ai_offset_85C_rms_mg',0):.1f} mg",
                        f"-{hp.get('improvement_pct',0):.0f}% improvement")
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ── SOC ───────────────────────────────────────────────────────────────────────
elif MODULE == "💻 SoC Binning & Harvesting":
    st.title("💻 SoC Fmax/Vmin Prediction + SKU Harvesting")
    st.markdown("""
    Predicts per-die Fmax and Vmin from ring oscillators and process monitors,
    reducing functional speed test time. Optimizes die-to-SKU assignment to
    maximize revenue vs demand.
    """)

    try:
        df = load_csv("soc_binning_harvesting.csv")
        df_eng = df.copy()
        core_fmax_cols = [f"core{i}_fmax_GHz" for i in range(8)]
        core_def_cols  = [f"core{i}_defect" for i in range(8)]
        fmax_vals = df_eng[core_fmax_cols].copy()
        for i in range(8):
            fmax_vals[f"core{i}_fmax_GHz"] = np.where(df_eng[f"core{i}_defect"]==1,
                                                        np.nan, fmax_vals[f"core{i}_fmax_GHz"])
        df_eng["fmax_min_GHz"] = fmax_vals.min(axis=1)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fmax Distribution by SKU")
            import plotly.express as px
            fig = px.histogram(df_eng.dropna(subset=["fmax_min_GHz"]),
                               x="fmax_min_GHz", color="sku_rule_based",
                               title="Slowest-Core Fmax Distribution", nbins=40)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Fmax vs Ring Oscillator Frequency")
            sample = df_eng.dropna(subset=["fmax_min_GHz"]).sample(min(1000, len(df_eng)), random_state=42)
            fig = px.scatter(sample, x="ro_freq_MHz", y="fmax_min_GHz",
                             color="sku_rule_based", title="Fmax vs RO Frequency")
            st.plotly_chart(fig, use_container_width=True)

        summary = load_summary()
        if summary:
            soc = summary.get("soc_binning", {})
            col1, col2 = st.columns(2)
            with col1:
                fm = soc.get("fmax_min", {})
                st.metric("Fmax R²", f"{fm.get('cv_r2',0):.3f}", f"MAE {fm.get('cv_mae',0):.3f} GHz")
            with col2:
                vm = soc.get("vmin_max", {})
                st.metric("Vmin R²", f"{vm.get('cv_r2',0):.3f}", f"MAE {vm.get('cv_mae',0):.1f} mV")
            sku = soc.get("sku_assignment", {})
            if sku:
                st.markdown("#### SKU Assignment Comparison")
                rule_mix = sku.get("rule_based_sku_mix", {})
                ai_mix   = sku.get("ai_sku_mix", {})
                mix_df   = pd.DataFrame({"Rule-Based": rule_mix, "AI-Optimized": ai_mix}).fillna(0)
                st.dataframe(mix_df.style.format("{:.1%}"), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ── FUSE VALIDATION ───────────────────────────────────────────────────────────
elif MODULE == "🔒 Fuse-Map Validation":
    st.title("🔒 Fuse-Map Schema Validation")
    st.markdown("""
    Validates MCU fuse maps against the derivative schema. Checks for wrong flash size,
    peripheral mask mismatches, missing debug lock, reserved bit violations,
    missing RDP on automotive parts, and trim field corruption.
    """)

    try:
        df_audit  = load_csv("mcu_fuse_map_audit.csv")
        df_schema = load_csv("mcu_fuse_schema.csv")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Error Type Distribution")
            err_counts = df_audit[df_audit["has_error"]==True]["error_type"].value_counts()
            import plotly.express as px
            fig = px.bar(err_counts, title="Injected Error Types")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Fuse Schema")
            st.dataframe(df_schema[["field","lsb","nbits","description"]], use_container_width=True)

        summary = load_summary()
        if summary:
            fuse = summary.get("fuse_validation", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Fuse Maps", fuse.get("total_fuse_maps",0))
            col2.metric("Injected Errors", fuse.get("injected_errors",0))
            col3.metric("Precision",       f"{fuse.get('precision',0)*100:.1f}%")
            col4.metric("Recall",          f"{fuse.get('recall',0)*100:.1f}%")

            per_type = fuse.get("per_type_detection", {})
            if per_type:
                st.markdown("#### Per-Error-Type Detection Rate")
                pt_df = pd.DataFrame(per_type).T
                st.dataframe(pt_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ── MEMORY REPAIR ─────────────────────────────────────────────────────────────
elif MODULE == "🔩 Memory Repair":
    st.title("🔩 Memory Repair Optimization")
    st.markdown("""
    Compares greedy redundancy allocation vs an exact (set-cover) solver for SRAM
    repair. The exact solver saves additional arrays that greedy allocation misses,
    improving repairable yield.
    """)

    try:
        df = load_csv("memory_repair_bitmaps.csv")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Fail Cells Distribution")
            import plotly.express as px
            fig = px.histogram(df, x="n_fail_cells", color="greedy_repairable",
                               title="Fail Cell Count by Greedy Repair Status", nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### Greedy vs Optimal Repair")
            rep_df = pd.DataFrame({
                "Method":     ["Greedy", "Optimal (Exact Solver)"],
                "Repairable": [df["greedy_repairable"].mean()*100,
                               df["optimal_repairable"].mean()*100]
            })
            fig = px.bar(rep_df, x="Method", y="Repairable",
                         title="Repairable Yield (%)", text_auto=".1f")
            st.plotly_chart(fig, use_container_width=True)

        summary = load_summary()
        if summary:
            mem = summary.get("memory_repair", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Greedy Repairable",  f"{mem.get('greedy_repairable_pct',0):.1f}%")
            col2.metric("Exact Repairable",   f"{mem.get('optimal_repairable_pct',0):.1f}%")
            col3.metric("Arrays Saved",       mem.get("arrays_saved_vs_greedy",0),
                        f"+{mem.get('yield_gain_pct',0):.2f}% yield")
    except Exception as e:
        st.error(f"Error loading data: {e}")
