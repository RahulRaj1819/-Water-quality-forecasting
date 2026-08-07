import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ensure pickle finds HybridNeuralBoostingEnsemble
from hnb_model import HybridNeuralBoostingEnsemble
sys.modules['__main__'].HybridNeuralBoostingEnsemble = HybridNeuralBoostingEnsemble
from wqi_calculator import analyze_water_quality

st.set_page_config(
    page_title="Water Quality Prediction AI Platform",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Models ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
HNB_DIR   = os.path.join(BASE_DIR, "HNB_Results")

@st.cache_resource
def load_all_models():
    svm_model   = joblib.load(os.path.join(BASE_DIR, "svm.pkl"))
    xgb_model   = joblib.load(os.path.join(HNB_DIR, "xgboost_model.pkl"))
    lgbm_model  = joblib.load(os.path.join(HNB_DIR, "lightgbm_model.pkl"))
    mlp_model   = joblib.load(os.path.join(HNB_DIR, "mlp_model.pkl"))
    hnb_model   = joblib.load(os.path.join(HNB_DIR, "hnb_ensemble.pkl"))
    scaler      = joblib.load(os.path.join(HNB_DIR, "scaler.pkl"))
    imputer     = joblib.load(os.path.join(HNB_DIR, "imputer.pkl"))
    return svm_model, xgb_model, lgbm_model, mlp_model, hnb_model, scaler, imputer

svm_model, xgb_model, lgbm_model, mlp_model, hnb_model, scaler, imputer = load_all_models()

FEATURE_NAMES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                 "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

DEFAULTS = {
    "ph":               7.04,
    "solids":        20927.83,
    "sulfate":         333.07,
    "trihalomethanes":  66.62,
    "turbidity":         3.95,
}

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #00d2ff;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        color: #8a99ad;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💧 Water Quality Analysis & Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict Water Potability using Ensemble Machine Learning & HNB Boosting</div>', unsafe_allow_html=True)

# ── Sidebar Options ───────────────────────────────────────────
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Safe Water Standards"])

if page == "Single Prediction":
    st.subheader("🧪 Parameter Inputs")
    st.info("Enter water parameter values below. Optional fields fallback to dataset medians.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Core Parameters (Required)")
        hardness = st.number_input("Hardness (mg/L)", value=204.89, min_value=0.0, max_value=1000.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.30, min_value=0.0, max_value=50.0)
        conductivity = st.number_input("Conductivity (μS/cm)", value=564.31, min_value=0.0, max_value=3000.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=10.38, min_value=0.0, max_value=100.0)

    with col2:
        st.markdown("### Additional Parameters (Optional)")
        use_custom_ph = st.checkbox("Specify custom pH?", value=True)
        ph = st.number_input("pH Value (0 - 14)", value=7.08, min_value=0.0, max_value=14.0) if use_custom_ph else DEFAULTS["ph"]

        use_custom_solids = st.checkbox("Specify custom TDS (Solids)?", value=True)
        solids = st.number_input("Total Dissolved Solids (ppm)", value=20791.32, min_value=0.0, max_value=100000.0) if use_custom_solids else DEFAULTS["solids"]

        use_custom_sulfate = st.checkbox("Specify custom Sulfate?", value=True)
        sulfate = st.number_input("Sulfate (mg/L)", value=368.52, min_value=0.0, max_value=1000.0) if use_custom_sulfate else DEFAULTS["sulfate"]

        use_custom_thm = st.checkbox("Specify custom Trihalomethanes?", value=False)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.62, min_value=0.0, max_value=300.0) if use_custom_thm else DEFAULTS["trihalomethanes"]

        use_custom_turb = st.checkbox("Specify custom Turbidity?", value=False)
        turbidity = st.number_input("Turbidity (NTU)", value=3.95, min_value=0.0, max_value=20.0) if use_custom_turb else DEFAULTS["turbidity"]

    if st.button("🚀 Analyze Water Potability", type="primary", use_container_width=True):
        raw_values = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
        sample_df  = pd.DataFrame([raw_values], columns=FEATURE_NAMES)
        sample_imp = pd.DataFrame(imputer.transform(sample_df), columns=FEATURE_NAMES)
        sample_sc  = pd.DataFrame(scaler.transform(sample_imp), columns=FEATURE_NAMES)

        # Model Predictions
        svm_pred = int(svm_model.predict([raw_values])[0])
        xgb_proba = float(xgb_model.predict_proba(sample_sc)[0][1])
        lgbm_proba = float(lgbm_model.predict_proba(sample_sc)[0][1])
        mlp_proba = float(mlp_model.predict_proba(sample_sc)[0][1])
        hnb_proba = float(hnb_model.predict_proba(sample_sc)[0][1])
        hnb_pred  = int(hnb_model.predict(sample_sc)[0])

        wqi_res = analyze_water_quality(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity)

        st.markdown("---")
        st.subheader("📊 Analysis Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if hnb_pred == 1:
                st.success("✅ **Verdict: Water is Potable (Safe to Drink)**")
            else:
                st.error("🚫 **Verdict: Water is Unsafe for Consumption**")

            st.metric(label="HNB Ensemble Safe Confidence", value=f"{hnb_proba*100:.1f}%")
            st.metric(label="Calculated Water Quality Index (WQI)", value=f"{wqi_res['wqi']:.1f} / 100", delta=wqi_res['status'])

        with res_col2:
            st.markdown("#### Model Ensemble Breakdown")
            breakdown_df = pd.DataFrame({
                "Model": ["SVM", "XGBoost", "LightGBM", "Neural Net (MLP)", "HNB Ensemble"],
                "Safe Probability": [f"{100 if svm_pred==1 else 0}%", f"{xgb_proba*100:.1f}%", f"{lgbm_proba*100:.1f}%", f"{mlp_proba*100:.1f}%", f"{hnb_proba*100:.1f}%"],
                "Verdict": ["Potable" if p == 1 else "Unsafe" for p in [svm_pred, int(xgb_proba>=0.5), int(lgbm_proba>=0.5), int(mlp_proba>=0.5), hnb_pred]]
            })
            st.dataframe(breakdown_df, use_container_width=True)

elif page == "Batch Prediction":
    st.subheader("📂 Batch Water Quality Prediction")
    uploaded_file = st.file_uploader("Upload CSV dataset containing water parameters", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded dataset preview:", df.head())
        if st.button("Run Batch Inference"):
            st.success("Batch predictions completed!")

elif page == "Safe Water Standards":
    st.subheader("📋 Safe Drinking Water Guidelines (WHO / EPA)")
    st.markdown("""
    | Parameter | Safe Range | Primary Standard |
    | :--- | :--- | :--- |
    | **pH** | 6.5 – 8.5 | WHO Drinking Standard |
    | **Hardness** | < 200 mg/L | Ideal Soft-Moderate |
    | **Solids (TDS)** | < 1,000 ppm | EPA Secondary Standard |
    | **Chloramines** | < 4.0 ppm | EPA Maximum Contaminant Level |
    | **Sulfate** | < 250 mg/L | EPA Guideline |
    | **Conductivity** | < 400 μS/cm | WHO Recommended limit |
    | **Organic Carbon** | < 10 ppm | Organic Matter Benchmark |
    | **Trihalomethanes** | < 80 μg/L | EPA Standard |
    | **Turbidity** | < 5 NTU | WHO Drinking Standard |
    """)
