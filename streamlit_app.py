"""
Water Quality Forecasting & Analytics Platform — Streamlit App
================================================================
Comprehensive interactive application integrating:
 - Single Parameter Water Potability Prediction
 - Multi-Model Ensemble (SVM, XGBoost, LightGBM, MLP, HNB)
 - Water Quality Index (WQI) Calculation & Remediation Suggestions
 - Exploratory Data Analysis & Model Performance Analytics
 - Batch CSV Prediction & Export
 - WHO/EPA Safe Water Standards Guide
 - Gemini AI Water Research Assistant
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Ensure pickle finds HybridNeuralBoostingEnsemble
from hnb_model import HybridNeuralBoostingEnsemble
sys.modules['__main__'].HybridNeuralBoostingEnsemble = HybridNeuralBoostingEnsemble
from wqi_calculator import analyze_water_quality
from chart_generator import (
    generate_confusion_matrices,
    generate_model_comparison,
    generate_roc_curves,
    generate_feature_importance,
)
import data_analyzer

# Safe ranges based on WHO / EPA drinking water guidelines
SAFE_RANGES = {
    "ph":              (6.5, 8.5),
    "hardness":        (0, 300),       # mg/L
    "solids":          (0, 1000),      # ppm
    "chloramines":     (0, 4),         # ppm
    "sulfate":         (0, 250),       # mg/L
    "conductivity":    (0, 500),       # μS/cm
    "organicCarbon":   (0, 2),         # ppm
    "trihalomethanes": (0, 80),        # μg/L
    "turbidity":       (0, 4),         # NTU
}

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Water Quality AI Platform",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Models & Scalers ──────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
HNB_DIR   = os.path.join(BASE_DIR, "HNB_Results")
CSV_PATH  = os.path.join(BASE_DIR, "Project_Resources", "water_potability.csv")

@st.cache_resource
def load_models_and_scalers():
    svm_model   = joblib.load(os.path.join(BASE_DIR, "svm.pkl"))
    xgb_model   = joblib.load(os.path.join(HNB_DIR, "xgboost_model.pkl"))
    lgbm_model  = joblib.load(os.path.join(HNB_DIR, "lightgbm_model.pkl"))
    mlp_model   = joblib.load(os.path.join(HNB_DIR, "mlp_model.pkl"))
    hnb_model   = joblib.load(os.path.join(HNB_DIR, "hnb_ensemble.pkl"))
    scaler      = joblib.load(os.path.join(HNB_DIR, "scaler.pkl"))
    imputer     = joblib.load(os.path.join(HNB_DIR, "imputer.pkl"))
    return svm_model, xgb_model, lgbm_model, mlp_model, hnb_model, scaler, imputer

svm_model, xgb_model, lgbm_model, mlp_model, hnb_model, scaler, imputer = load_models_and_scalers()

FEATURE_NAMES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                 "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

DEFAULTS = {
    "ph":               7.04,
    "solids":        20927.83,
    "sulfate":         333.07,
    "trihalomethanes":  66.62,
    "turbidity":         3.95,
}

# ── Configure Gemini AI ────────────────────────────────────────
def get_active_gemini_key():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    if not key and st.session_state.get("custom_gemini_key"):
        key = st.session_state.get("custom_gemini_key")
    if not key:
        key = "AIzaSyBmppH6aFbeu18WUPXeGaMVVVFtYnIU584"
    return key


# ── Pre-compute Dataset Evaluation Metrics ────────────────────
@st.cache_data
def get_dataset_evaluation():
    if not os.path.exists(CSV_PATH):
        return None, None, None, None
    df = pd.read_csv(CSV_PATH)
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    imp = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    sc.fit(X_train)
    X_test_scaled = pd.DataFrame(sc.transform(X_test), columns=X.columns, index=X_test.index)

    models = {
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model,
        "Neural Net (MLP)": mlp_model,
    }

    results = []
    preds = {}
    probas = {}

    for name, m in models.items():
        yp = m.predict(X_test_scaled)
        ypr = m.predict_proba(X_test_scaled)[:, 1]
        results.append({
            "name": name,
            "accuracy": accuracy_score(y_test, yp),
            "precision": precision_score(y_test, yp, zero_division=0),
            "recall": recall_score(y_test, yp, zero_division=0),
            "f1": f1_score(y_test, yp, zero_division=0),
            "auc": roc_auc_score(y_test, ypr)
        })
        preds[name] = yp
        probas[name] = ypr

    # HNB Ensemble
    hnb_pred = hnb_model.predict(X_test_scaled)
    hnb_proba = hnb_model.predict_proba(X_test_scaled)[:, 1]
    results.append({
        "name": "HNB Ensemble",
        "accuracy": accuracy_score(y_test, hnb_pred),
        "precision": precision_score(y_test, hnb_pred, zero_division=0),
        "recall": recall_score(y_test, hnb_pred, zero_division=0),
        "f1": f1_score(y_test, hnb_pred, zero_division=0),
        "auc": roc_auc_score(y_test, hnb_proba)
    })
    preds["HNB Ensemble"] = hnb_pred
    probas["HNB Ensemble"] = hnb_proba

    return y_test, results, preds, probas

# ── Remediation Suggestions Helper ────────────────────────────
def generate_treatment_suggestions(ph, hardness, solids, chloramines, sulfate,
                                  conductivity, organicCarbon, trihalomethanes, turbidity):
    suggestions = []
    if ph < SAFE_RANGES["ph"][0]:
        suggestions.append(("🧪 pH Level", f"Too Acidic ({ph:.2f})", "Add lime (calcium hydroxide) or soda ash to raise pH."))
    elif ph > SAFE_RANGES["ph"][1]:
        suggestions.append(("🧪 pH Level", f"Too Alkaline ({ph:.2f})", "Inject CO₂ or add food-grade citric acid to lower pH."))

    if hardness > SAFE_RANGES["hardness"][1]:
        suggestions.append(("🪨 Hardness", f"Too Hard ({hardness:.0f} mg/L)", "Install an ion-exchange water softener or Reverse Osmosis (RO) system."))

    if solids > SAFE_RANGES["solids"][1]:
        suggestions.append(("🔬 Total Dissolved Solids", f"Elevated ({solids:.0f} ppm)", "Use Reverse Osmosis (RO) filtration or distillation."))

    if chloramines > SAFE_RANGES["chloramines"][1]:
        suggestions.append(("⚗️ Chloramines", f"Elevated ({chloramines:.2f} ppm)", "Install a catalytic activated carbon filter designed for chloramines."))

    if sulfate > SAFE_RANGES["sulfate"][1]:
        suggestions.append(("💎 Sulfate", f"Elevated ({sulfate:.0f} mg/L)", "Use RO membrane filtration or anion exchange."))

    if conductivity > SAFE_RANGES["conductivity"][1]:
        suggestions.append(("⚡ Conductivity", f"High ({conductivity:.0f} μS/cm)", "Apply RO or deionization to remove dissolved mineral ions."))

    if organicCarbon > SAFE_RANGES["organicCarbon"][1]:
        suggestions.append(("🌿 Organic Carbon", f"Elevated ({organicCarbon:.2f} ppm)", "Use granular activated carbon (GAC) filtration and UV treatment."))

    if trihalomethanes > SAFE_RANGES["trihalomethanes"][1]:
        suggestions.append(("☢️ Trihalomethanes", f"Elevated ({trihalomethanes:.1f} μg/L)", "Use activated carbon filtration; switch to UV/ozone disinfection."))

    if turbidity > SAFE_RANGES["turbidity"][1]:
        suggestions.append(("🌫️ Turbidity", f"Too Cloudy ({turbidity:.2f} NTU)", "Apply multi-stage sediment filtration or coagulation-flocculation."))

    if not suggestions:
        suggestions.append(("💡 Overall Quality", "All parameters within standard limits", "Water parameters meet primary safety guidelines. Routine multi-stage carbon filtering recommended."))

    return suggestions

# ── Sidebar Navigation ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/isometric/96/water.png", width=70)
st.sidebar.title("Water Quality AI Platform")
st.sidebar.caption("Predictive Analytics & WQI Forecasting")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🧪 Single Prediction Dashboard",
        "📈 EDA & Model Evaluation",
        "📂 Batch CSV Prediction",
        "📋 Safe Water Standards",
        "🤖 AI Research Assistant"
    ]
)

# ══════════════════════════════════════════════════════════════
#  1. SINGLE PREDICTION DASHBOARD
# ══════════════════════════════════════════════════════════════
if nav_choice == "🧪 Single Prediction Dashboard":
    st.title("💧 Water Quality Analysis & Potability Prediction")
    st.write("Enter physical and chemical water parameters below to calculate the **Water Quality Index (WQI)** and run **Ensemble Machine Learning** predictions.")

    st.markdown("---")

    col_req, col_opt = st.columns(2)

    with col_req:
        st.subheader("🔹 Core Parameters (Required)")
        hardness = st.number_input("Hardness (mg/L)", value=204.89, min_value=0.0, max_value=1000.0, step=1.0)
        chloramines = st.number_input("Chloramines (ppm)", value=7.30, min_value=0.0, max_value=50.0, step=0.1)
        conductivity = st.number_input("Conductivity (μS/cm)", value=564.31, min_value=0.0, max_value=3000.0, step=1.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", value=10.38, min_value=0.0, max_value=100.0, step=0.1)

    with col_opt:
        st.subheader("🔸 Additional Parameters (Optional)")
        ph_use = st.checkbox("Custom pH value?", value=True)
        ph = st.number_input("pH (0 – 14)", value=7.08, min_value=0.0, max_value=14.0, step=0.1) if ph_use else DEFAULTS["ph"]

        solids_use = st.checkbox("Custom Total Dissolved Solids (TDS)?", value=True)
        solids = st.number_input("Solids (ppm)", value=20791.32, min_value=0.0, max_value=100000.0, step=10.0) if solids_use else DEFAULTS["solids"]

        sulfate_use = st.checkbox("Custom Sulfate?", value=True)
        sulfate = st.number_input("Sulfate (mg/L)", value=368.52, min_value=0.0, max_value=1000.0, step=1.0) if sulfate_use else DEFAULTS["sulfate"]

        thm_use = st.checkbox("Custom Trihalomethanes?", value=False)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=66.62, min_value=0.0, max_value=300.0, step=0.5) if thm_use else DEFAULTS["trihalomethanes"]

        turb_use = st.checkbox("Custom Turbidity?", value=False)
        turbidity = st.number_input("Turbidity (NTU)", value=3.95, min_value=0.0, max_value=20.0, step=0.1) if turb_use else DEFAULTS["turbidity"]

    st.markdown("---")

    if st.button("🚀 Run Complete Water Quality Analysis", type="primary", use_container_width=True):
        raw_values = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
        sample_df  = pd.DataFrame([raw_values], columns=FEATURE_NAMES)
        sample_imp = pd.DataFrame(imputer.transform(sample_df), columns=FEATURE_NAMES)
        sample_sc  = pd.DataFrame(scaler.transform(sample_imp), columns=FEATURE_NAMES)

        # Model Inference
        svm_pred = int(svm_model.predict([raw_values])[0])
        xgb_proba = float(xgb_model.predict_proba(sample_sc)[0][1])
        lgbm_proba = float(lgbm_model.predict_proba(sample_sc)[0][1])
        mlp_proba = float(mlp_model.predict_proba(sample_sc)[0][1])
        hnb_proba = float(hnb_model.predict_proba(sample_sc)[0][1])
        hnb_pred  = int(hnb_model.predict(sample_sc)[0])

        wqi, impacts = analyze_water_quality(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity)

        wqi_status = "Excellent" if wqi >= 80 else "Good" if wqi >= 60 else "Fair" if wqi >= 40 else "Poor"

        # Overview Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("HNB Verdict", "POTABLE ✅" if hnb_pred == 1 else "UNSAFE 🚫")
        with m2:
            st.metric("Safety Confidence", f"{hnb_proba*100:.1f}%")
        with m3:
            st.metric("WQI Index", f"{wqi:.1f} / 100")
        with m4:
            st.metric("WQI Classification", wqi_status)

        st.markdown("---")

        res_left, res_right = st.columns([1, 1])

        with res_left:
            st.subheader("🤖 Model Ensemble Predictions")
            m_df = pd.DataFrame({
                "Model": ["SVM", "XGBoost", "LightGBM", "Neural Net (MLP)", "HNB Ensemble (Final)"],
                "Safe Probability": [
                    "100.0%" if svm_pred == 1 else "0.0%",
                    f"{xgb_proba*100:.1f}%",
                    f"{lgbm_proba*100:.1f}%",
                    f"{mlp_proba*100:.1f}%",
                    f"{hnb_proba*100:.1f}%"
                ],
                "Verdict": [
                    "Potable" if p == 1 else "Unsafe"
                    for p in [svm_pred, int(xgb_proba>=0.5), int(lgbm_proba>=0.5), int(mlp_proba>=0.5), hnb_pred]
                ]
            })
            st.dataframe(m_df, use_container_width=True)

        with res_right:
            st.subheader("🛠️ Water Treatment & Remediation Advice")
            suggestions = generate_treatment_suggestions(
                ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity
            )
            for title, status_str, advice_str in suggestions:
                st.warning(f"**{title}** — *{status_str}*\n\n{advice_str}")

# ══════════════════════════════════════════════════════════════
#  2. EDA & MODEL EVALUATION
# ══════════════════════════════════════════════════════════════
elif nav_choice == "📈 EDA & Model Evaluation":
    st.title("📈 Exploratory Data Analysis & Model Evaluation")
    st.write("Performance evaluation metrics and dataset feature distributions computed on the Water Potability benchmark dataset.")

    y_test, results, preds, probas = get_dataset_evaluation()

    if results:
        st.subheader("📊 Model Performance Comparison")
        res_df = pd.DataFrame(results).set_index("name")
        st.dataframe(res_df.style.highlight_max(axis=0, color="#1e3a8a"), use_container_width=True)

        col_fig1, col_fig2 = st.columns(2)

        with col_fig1:
            st.subheader("🎯 Confusion Matrices")
            fig_cm_b64 = generate_confusion_matrices(y_test, preds)
            st.image(f"data:image/png;base64,{fig_cm_b64}")

        with col_fig2:
            st.subheader("📈 ROC Curves")
            fig_roc_b64 = generate_roc_curves(y_test, probas)
            st.image(f"data:image/png;base64,{fig_roc_b64}")

        st.subheader("⭐ Feature Importance (XGBoost vs LightGBM)")
        fig_fi_b64 = generate_feature_importance(xgb_model, lgbm_model, FEATURE_NAMES)
        st.image(f"data:image/png;base64,{fig_fi_b64}")

# ══════════════════════════════════════════════════════════════
#  3. BATCH CSV PREDICTION
# ══════════════════════════════════════════════════════════════
elif nav_choice == "📂 Batch CSV Prediction":
    st.title("📂 Batch CSV Water Quality Prediction")
    st.write("Upload a CSV file containing water parameters to run batch predictions across all ensemble models and download the results.")

    uploaded_file = st.file_uploader("Upload Water Quality CSV", type=["csv"])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.write("📋 Dataset Preview:", batch_df.head())

        # Check column presence
        missing_cols = [col for col in ["Hardness", "Chloramines", "Conductivity", "Organic_carbon"] if col not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            if st.button("🚀 Process Batch Predictions", type="primary"):
                # Fill missing optional columns with defaults
                for col in ["ph", "Solids", "Sulfate", "Trihalomethanes", "Turbidity"]:
                    if col not in batch_df.columns:
                        batch_df[col] = DEFAULTS[col.lower()]

                X_batch = batch_df[FEATURE_NAMES].fillna(
                    {"ph": DEFAULTS["ph"], "Solids": DEFAULTS["solids"], "Sulfate": DEFAULTS["sulfate"],
                     "Trihalomethanes": DEFAULTS["trihalomethanes"], "Turbidity": DEFAULTS["turbidity"]}
                )

                X_batch_imp = pd.DataFrame(imputer.transform(X_batch), columns=FEATURE_NAMES)
                X_batch_sc = pd.DataFrame(scaler.transform(X_batch_imp), columns=FEATURE_NAMES)

                batch_preds = hnb_model.predict(X_batch_sc)
                batch_probas = hnb_model.predict_proba(X_batch_sc)[:, 1]

                out_df = batch_df.copy()
                out_df["Predicted_Potability"] = batch_preds
                out_df["Potability_Verdict"] = ["Potable" if p == 1 else "Unsafe" for p in batch_preds]
                out_df["HNB_Confidence_Pct"] = (batch_probas * 100).round(2)

                st.success("✅ Batch predictions successfully completed!")
                st.dataframe(out_df.head(20), use_container_width=True)

                csv_buffer = io.StringIO()
                out_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_buffer.getvalue(),
                    file_name="water_quality_predictions.csv",
                    mime="text/csv"
                )

# ══════════════════════════════════════════════════════════════
#  4. SAFE WATER STANDARDS
# ══════════════════════════════════════════════════════════════
elif nav_choice == "📋 Safe Water Standards":
    st.title("📋 Drinking Water Quality Standards (WHO & EPA Guidelines)")
    st.write("Reference acceptable parameter ranges and health implications for drinking water potability.")

    st.markdown("""
    | Parameter | Safe Range | Standard Body | Primary Health / Quality Impact |
    | :--- | :--- | :--- | :--- |
    | **pH** | 6.5 – 8.5 | WHO / EPA | Corrosiveness, pipe leaching, scale formation |
    | **Hardness** | 0 – 200 mg/L | WHO | Scale buildup in boilers, soap lathering efficiency |
    | **Total Dissolved Solids (TDS)** | < 1,000 ppm | EPA Secondary | High mineral content, bitter taste |
    | **Chloramines** | < 4.0 ppm | EPA MCL | Eye/nose irritation, disinfectant residual |
    | **Sulfate** | < 250 mg/L | EPA | Laxative effects, taste impairment |
    | **Conductivity** | < 400 μS/cm | WHO | Indicator of high dissolved mineral ion concentration |
    | **Organic Carbon** | < 10 ppm | EPA | Precursor to harmful disinfection byproducts |
    | **Trihalomethanes (THMs)** | < 80 μg/L | EPA | Carcinogenic disinfection byproducts |
    | **Turbidity** | < 5.0 NTU | WHO | Cloudiness, viral/bacterial shelter in suspended solids |
    """)

# ══════════════════════════════════════════════════════════════
#  5. AI RESEARCH ASSISTANT (CHATBOT)
# ══════════════════════════════════════════════════════════════
elif nav_choice == "🤖 AI Research Assistant":
    st.title("🤖 Gemini AI Water Research Assistant")
    st.write("Ask questions regarding water quality forecasting, WQI methodology, and predictive analytics models.")

    current_key = get_active_gemini_key()

    with st.expander("🔑 Gemini API Key Configuration", expanded=not bool(current_key)):
        user_key_input = st.text_input(
            "Enter your Gemini API Key (Optional override):",
            value=current_key,
            type="password",
            help="Get your key from Google AI Studio (https://aistudio.google.com/app/apikey)"
        )
        if user_key_input:
            st.session_state["custom_gemini_key"] = user_key_input.strip()
            current_key = user_key_input.strip()

    if current_key:
        try:
            genai.configure(api_key=current_key)
            st.success("✅ Gemini AI Core Connected")
        except Exception as e:
            st.error(f"Failed to configure Gemini API Key: {e}")
    else:
        st.warning("⚠️ No Gemini API Key provided. Please enter a key above or set `GEMINI_API_KEY` in environment variables.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("Ask a question about water quality forecasting..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        if current_key:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                sys_prompt = (
                    "You are an expert AI research assistant specializing in smart environmental technologies and predictive analytics for water management based on water quality parameters.\n"
                    f"User question: {user_prompt}"
                )
                resp = model.generate_content(sys_prompt)
                ai_text = resp.text
            except Exception as e:
                ai_text = f"Experiencing technical issues contacting AI core: {e}"
        else:
            ai_text = f"Simulated Response: Based on standard predictive models, parameter thresholds (such as pH 6.5–8.5 and Chloramines < 4.0 ppm) govern water quality index scores."

        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        with st.chat_message("assistant"):
            st.markdown(ai_text)

