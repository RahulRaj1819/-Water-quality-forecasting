"""
Water Quality Prediction — Flask Application
=============================================
Loads SVM + HNB ensemble (XGBoost, LightGBM, MLP) models
and generates comprehensive analysis with charts on prediction.
"""

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import numpy as np
import joblib
import os
import io
import google.generativeai as genai

# Configure Gemini — key loaded from environment variable (never hardcode secrets)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from chart_generator import (
    generate_confusion_matrices,
    generate_model_comparison,
    generate_roc_curves,
    generate_feature_importance,
    generate_prediction_gauge,
)

# Must import so pickle can find HybridNeuralBoostingEnsemble when loading hnb_ensemble.pkl
# The pkl was saved when hnb_model.py was run as __main__, so the class is stored
# as "__main__.HybridNeuralBoostingEnsemble". We patch sys.modules accordingly.
import sys
from hnb_model import HybridNeuralBoostingEnsemble
sys.modules['__main__'].HybridNeuralBoostingEnsemble = HybridNeuralBoostingEnsemble

from wqi_calculator import analyze_water_quality

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
HNB_DIR   = os.path.join(BASE_DIR, "HNB_Results")
CSV_PATH  = os.path.join(BASE_DIR, "Project_Resources", "water_potability.csv")

# ── Load Models ────────────────────────────────────────────────
svm_model   = joblib.load(os.path.join(BASE_DIR, "svm.pkl"))
xgb_model   = joblib.load(os.path.join(HNB_DIR, "xgboost_model.pkl"))
lgbm_model  = joblib.load(os.path.join(HNB_DIR, "lightgbm_model.pkl"))
mlp_model   = joblib.load(os.path.join(HNB_DIR, "mlp_model.pkl"))
hnb_model   = joblib.load(os.path.join(HNB_DIR, "hnb_ensemble.pkl"))
scaler      = joblib.load(os.path.join(HNB_DIR, "scaler.pkl"))
imputer     = joblib.load(os.path.join(HNB_DIR, "imputer.pkl"))

print("[INFO] All models loaded successfully.")

# ── Dataset median defaults for optional fields ────────────────
DEFAULTS = {
    "ph":               7.04,
    "solids":        20927.83,
    "sulfate":         333.07,
    "trihalomethanes":  66.62,
    "turbidity":         3.95,
}

FEATURE_NAMES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                 "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

# ── Pre-compute test set evaluation metrics on startup ─────────
def _precompute_test_metrics():
    """Evaluate all models on the test set once at startup."""
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
        "XGBoost":          xgb_model,
        "LightGBM":         lgbm_model,
        "Neural Net (MLP)": mlp_model,
    }

    results_list    = []
    predictions_dict = {}
    probas_dict     = {}

    for name, model in models.items():
        y_pred  = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1v  = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba)
        results_list.append({
            "name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1v, "auc": auc
        })
        predictions_dict[name] = y_pred
        probas_dict[name]      = y_proba

    # HNB Ensemble
    hnb_pred  = hnb_model.predict(X_test_scaled)
    hnb_proba = hnb_model.predict_proba(X_test_scaled)[:, 1]
    acc  = accuracy_score(y_test, hnb_pred)
    prec = precision_score(y_test, hnb_pred, zero_division=0)
    rec  = recall_score(y_test, hnb_pred, zero_division=0)
    f1v  = f1_score(y_test, hnb_pred, zero_division=0)
    auc  = roc_auc_score(y_test, hnb_proba)
    results_list.append({
        "name": "HNB Ensemble", "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1v, "auc": auc
    })
    predictions_dict["HNB Ensemble"] = hnb_pred
    probas_dict["HNB Ensemble"]      = hnb_proba

    return y_test, results_list, predictions_dict, probas_dict, X_test_scaled

print("[INFO] Pre-computing test metrics...")
y_test_global, results_global, preds_global, probas_global, X_test_scaled_global = _precompute_test_metrics()
print("[INFO] Test metrics computed.")


# ── Pre-generate static charts (computed once) ─────────────────
print("[INFO] Generating charts...")
chart_confusion   = generate_confusion_matrices(y_test_global, preds_global)
chart_comparison  = generate_model_comparison(results_global)
chart_roc         = generate_roc_curves(y_test_global, probas_global)
chart_feature_imp = generate_feature_importance(xgb_model, lgbm_model, FEATURE_NAMES)
print("[INFO] Charts ready.")


# ════════════════════════════════════════════════════════════════
#  TREATMENT SUGGESTIONS
# ════════════════════════════════════════════════════════════════
# Safe ranges based on WHO / EPA drinking water guidelines
SAFE_RANGES = {
    "ph":              (6.5, 8.5),
    "hardness":        (0, 300),       # mg/L
    "solids":          (0, 1000),      # ppm (ideal < 500, max 1000)
    "chloramines":     (0, 4),         # ppm
    "sulfate":         (0, 250),       # mg/L
    "conductivity":    (0, 500),       # μS/cm
    "organicCarbon":   (0, 2),         # ppm (ideal < 2 for treated water)
    "trihalomethanes": (0, 80),        # μg/L
    "turbidity":       (0, 4),         # NTU (ideal < 1)
}

def _generate_suggestions(ph, hardness, solids, chloramines, sulfate,
                          conductivity, organicCarbon, trihalomethanes, turbidity):
    """
    Analyze each parameter against safe ranges and generate
    specific water treatment suggestions.
    Returns a list of dicts: {icon, param, status, advice}
    """
    suggestions = []

    # pH
    lo, hi = SAFE_RANGES["ph"]
    if ph < lo:
        suggestions.append({
            "icon": "🧪", "param": "pH",
            "status": f"Too acidic ({ph:.2f} — safe range {lo}–{hi})",
            "advice": "Add lime (calcium hydroxide) or soda ash (sodium carbonate) to raise pH. Aeration can also help neutralize acidic water."
        })
    elif ph > hi:
        suggestions.append({
            "icon": "🧪", "param": "pH",
            "status": f"Too alkaline ({ph:.2f} — safe range {lo}–{hi})",
            "advice": "Inject CO₂ or add acid (citric acid, dilute hydrochloric acid) to lower pH. Install an acid neutralizing filter."
        })

    # Hardness
    lo, hi = SAFE_RANGES["hardness"]
    if hardness > hi:
        suggestions.append({
            "icon": "🪨", "param": "Hardness",
            "status": f"Too hard ({hardness:.0f} mg/L — max {hi} mg/L)",
            "advice": "Install a water softener (ion-exchange system) to remove excess calcium and magnesium. Reverse osmosis (RO) also effectively reduces hardness."
        })

    # Total Dissolved Solids
    lo, hi = SAFE_RANGES["solids"]
    if solids > hi:
        suggestions.append({
            "icon": "🔬", "param": "Total Dissolved Solids",
            "status": f"Elevated ({solids:.0f} ppm — max {hi} ppm)",
            "advice": "Use Reverse Osmosis (RO) filtration to reduce TDS. Distillation or deionization systems are also effective for high-TDS water."
        })

    # Chloramines
    lo, hi = SAFE_RANGES["chloramines"]
    if chloramines > hi:
        suggestions.append({
            "icon": "⚗️", "param": "Chloramines",
            "status": f"Elevated ({chloramines:.2f} ppm — max {hi} ppm)",
            "advice": "Use a catalytic activated carbon filter specifically designed for chloramine removal. Standard carbon filters are less effective — ensure the filter is rated for chloramines."
        })

    # Sulfate
    lo, hi = SAFE_RANGES["sulfate"]
    if sulfate > hi:
        suggestions.append({
            "icon": "💎", "param": "Sulfate",
            "status": f"Elevated ({sulfate:.0f} mg/L — max {hi} mg/L)",
            "advice": "Install a Reverse Osmosis (RO) system or an anion exchange unit to reduce sulfate levels. Distillation is also effective."
        })

    # Conductivity
    lo, hi = SAFE_RANGES["conductivity"]
    if conductivity > hi:
        suggestions.append({
            "icon": "⚡", "param": "Conductivity",
            "status": f"Too high ({conductivity:.0f} μS/cm — max {hi} μS/cm)",
            "advice": "High conductivity indicates excess dissolved minerals. Use RO filtration or deionization to reduce ionic content. Check for industrial contamination sources."
        })

    # Organic Carbon
    lo, hi = SAFE_RANGES["organicCarbon"]
    if organicCarbon > hi:
        suggestions.append({
            "icon": "🌿", "param": "Organic Carbon",
            "status": f"Elevated ({organicCarbon:.2f} ppm — ideal < {hi} ppm)",
            "advice": "Use activated carbon filtration to adsorb organic compounds. Enhanced coagulation/flocculation followed by filtration is recommended for municipal treatment. UV treatment can also help break down organics."
        })

    # Trihalomethanes
    lo, hi = SAFE_RANGES["trihalomethanes"]
    if trihalomethanes > hi:
        suggestions.append({
            "icon": "☢️", "param": "Trihalomethanes",
            "status": f"Elevated ({trihalomethanes:.1f} μg/L — max {hi} μg/L)",
            "advice": "Install a granular activated carbon (GAC) filter to remove THMs. Reduce at source by switching from chlorine to UV or ozone disinfection. Aeration can also help volatilize THMs."
        })

    # Turbidity
    lo, hi = SAFE_RANGES["turbidity"]
    if turbidity > hi:
        suggestions.append({
            "icon": "🌫️", "param": "Turbidity",
            "status": f"Too cloudy ({turbidity:.2f} NTU — ideal < {hi} NTU)",
            "advice": "Use sediment filtration or a multi-stage filtration system. For higher turbidity, apply coagulation (alum or ferric chloride) followed by sedimentation and filtration."
        })

    # If nothing is out of range but still unsafe, add general advice
    if not suggestions:
        suggestions.append({
            "icon": "💡", "param": "General",
            "status": "Individual parameters are within ranges, but the combined pattern suggests risk",
            "advice": "Consider multi-stage treatment: sediment filter → activated carbon → UV disinfection → RO membrane. Have the water professionally tested for bacteria, heavy metals, and emerging contaminants."
        })

    return suggestions


# ════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_waterQ', methods=['GET', 'POST'])
def predict_waterQ():
    # ── Parse form inputs ──────────────────────────────────────
    # Required parameters
    hardness      = float(request.form["hardness"])
    chloramines   = float(request.form["chloramines"])
    conductivity  = float(request.form["conductivity"])
    organicCarbon = float(request.form["organicCarbon"])

    # Optional parameters — fall back to dataset median if blank
    ph_raw              = request.form.get("ph", "").strip()
    solids_raw          = request.form.get("solids", "").strip()
    sulfate_raw         = request.form.get("sulfate", "").strip()
    trihalomethanes_raw = request.form.get("trihalomethanes", "").strip()
    turbidity_raw       = request.form.get("turbidity", "").strip()

    ph              = float(ph_raw)              if ph_raw              else DEFAULTS["ph"]
    solids          = float(solids_raw)          if solids_raw          else DEFAULTS["solids"]
    sulfate         = float(sulfate_raw)         if sulfate_raw         else DEFAULTS["sulfate"]
    trihalomethanes = float(trihalomethanes_raw) if trihalomethanes_raw else DEFAULTS["trihalomethanes"]
    turbidity       = float(turbidity_raw)       if turbidity_raw       else DEFAULTS["turbidity"]

    # Raw feature values (unscaled) — matches dataset column order
    raw_values = [ph, hardness, solids, chloramines, sulfate,
                  conductivity, organicCarbon, trihalomethanes, turbidity]

    # ── Prepare sample for HNB models (impute → scale) ─────────
    sample_df  = pd.DataFrame([raw_values], columns=FEATURE_NAMES)
    sample_imp = pd.DataFrame(imputer.transform(sample_df), columns=FEATURE_NAMES)
    sample_sc  = pd.DataFrame(scaler.transform(sample_imp), columns=FEATURE_NAMES)

    # ── Run predictions through each model ─────────────────────
    model_predictions = []

    # SVM (uses raw unscaled values, as the original svm.pkl was trained on raw data)
    svm_pred = svm_model.predict([raw_values])[0]
    model_predictions.append({
        "name": "SVM",
        "prediction": int(svm_pred),
        "confidence": 100.0,  # SVM doesn't give probabilities easily via predict
        "proba_safe": 1.0 if svm_pred == 1 else 0.0,
    })

    # XGBoost
    xgb_pred  = xgb_model.predict(sample_sc)[0]
    xgb_proba = xgb_model.predict_proba(sample_sc)[0]
    model_predictions.append({
        "name": "XGBoost",
        "prediction": int(xgb_pred),
        "confidence": float(max(xgb_proba) * 100),
        "proba_safe": float(xgb_proba[1]),
    })

    # LightGBM
    lgbm_pred  = lgbm_model.predict(sample_sc)[0]
    lgbm_proba = lgbm_model.predict_proba(sample_sc)[0]
    model_predictions.append({
        "name": "LightGBM",
        "prediction": int(lgbm_pred),
        "confidence": float(max(lgbm_proba) * 100),
        "proba_safe": float(lgbm_proba[1]),
    })

    # Neural Net (MLP)
    mlp_pred  = mlp_model.predict(sample_sc)[0]
    mlp_proba = mlp_model.predict_proba(sample_sc)[0]
    model_predictions.append({
        "name": "Neural Net (MLP)",
        "prediction": int(mlp_pred),
        "confidence": float(max(mlp_proba) * 100),
        "proba_safe": float(mlp_proba[1]),
    })

    # HNB Ensemble (primary model)
    hnb_pred  = hnb_model.predict(sample_sc)[0]
    hnb_proba = hnb_model.predict_proba(sample_sc)[0]
    model_predictions.append({
        "name": "HNB Ensemble",
        "prediction": int(hnb_pred),
        "confidence": float(max(hnb_proba) * 100),
        "proba_safe": float(hnb_proba[1]),
    })

    # ── Primary prediction (HNB) ──────────────────────────────
    primary_pred = "Safe" if hnb_pred == 1 else "Unsafe"

    # ── Input parameters summary ──────────────────────────────
    params = {
        "pH":              f"{ph:.2f}"            + ("  ⌀" if not ph_raw else ""),
        "Hardness":        f"{hardness:.2f} mg/L",
        "Solids":          f"{solids:.2f} ppm"    + ("  ⌀" if not solids_raw else ""),
        "Chloramines":     f"{chloramines:.2f} ppm",
        "Sulfate":         f"{sulfate:.2f} mg/L"  + ("  ⌀" if not sulfate_raw else ""),
        "Conductivity":    f"{conductivity:.2f} μS/cm",
        "Organic Carbon":  f"{organicCarbon:.2f} ppm",
        "Trihalomethanes": f"{trihalomethanes:.2f} μg/L" + ("  ⌀" if not trihalomethanes_raw else ""),
        "Turbidity":       f"{turbidity:.2f} NTU"  + ("  ⌀" if not turbidity_raw else ""),
    }

    # ── Generate treatment suggestions for unsafe water ────────
    suggestions = _generate_suggestions(
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organicCarbon, trihalomethanes, turbidity
    )

    # ── WQI & Parameter Impact ─────────────────────────────────
    wqi, impacts = analyze_water_quality(
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organicCarbon, trihalomethanes, turbidity
    )

    # ── Generate per-sample prediction gauge chart ─────────────
    chart_gauge = generate_prediction_gauge(model_predictions)

    # ── Metrics table data ─────────────────────────────────────
    metrics_table = results_global  # pre-computed at startup

    return render_template(
        'result.html',
        prediction=primary_pred,
        params=params,
        model_predictions=model_predictions,
        metrics_table=metrics_table,
        suggestions=suggestions,
        chart_gauge=chart_gauge,
        chart_confusion=chart_confusion,
        chart_comparison=chart_comparison,
        chart_roc=chart_roc,
        chart_feature_imp=chart_feature_imp,
        wqi=wqi,
        impacts=impacts,
    )


@app.route('/guidelines')
def guidelines():
    """Water quality standards reference page."""
    standards = [
        {
            "param": "pH",
            "icon": "🧪",
            "unit": "",
            "safe_min": 6.5,
            "safe_max": 8.5,
            "ideal": "7.0 – 7.5",
            "color": "#38bdf8",
            "description": "Measures how acidic or alkaline water is on a scale of 0–14. Pure water has a pH of 7.0.",
            "health_effects": "Water below 6.5 can leach metals (lead, copper) from pipes causing toxicity. Above 8.5 gives a bitter taste and reduces disinfection efficiency.",
            "treatment": "Too low → add lime or soda ash. Too high → inject CO₂ or use acid neutralizers.",
        },
        {
            "param": "Hardness",
            "icon": "🪨",
            "unit": "mg/L",
            "safe_min": 0,
            "safe_max": 300,
            "ideal": "60 – 120",
            "color": "#34d399",
            "description": "Caused by dissolved calcium and magnesium. Hard water forms scale in pipes and appliances.",
            "health_effects": "Very hard water (>300) can cause scale buildup, reduce soap effectiveness, and may contribute to kidney stones with prolonged consumption.",
            "treatment": "Water softeners (ion-exchange), reverse osmosis, or boiling for temporary hardness.",
        },
        {
            "param": "Total Dissolved Solids",
            "icon": "🔬",
            "unit": "ppm",
            "safe_min": 0,
            "safe_max": 500,
            "ideal": "< 300",
            "color": "#fbbf24",
            "description": "Total concentration of all dissolved substances — minerals, salts, metals, and organic matter.",
            "health_effects": "High TDS (>1000) causes unpleasant taste, indicates possible contamination. Very high TDS can cause gastrointestinal irritation.",
            "treatment": "Reverse osmosis (RO) is most effective. Distillation and deionization also work.",
        },
        {
            "param": "Chloramines",
            "icon": "⚗️",
            "unit": "ppm",
            "safe_min": 0,
            "safe_max": 4.0,
            "ideal": "1.0 – 3.0",
            "color": "#f87171",
            "description": "Used as a disinfectant in water treatment. Formed by combining chlorine with ammonia.",
            "health_effects": "Above 4 ppm can irritate eyes and skin, cause digestive issues. Harmful to kidney dialysis patients and fish in aquariums.",
            "treatment": "Catalytic activated carbon filters (standard carbon is less effective). UV treatment as alternative disinfection.",
        },
        {
            "param": "Sulfate",
            "icon": "💎",
            "unit": "mg/L",
            "safe_min": 0,
            "safe_max": 250,
            "ideal": "< 200",
            "color": "#a78bfa",
            "description": "Naturally occurring mineral from soil/rock dissolution. Found in most water supplies.",
            "health_effects": "High sulfate (>500) acts as a laxative, causes diarrhea especially in children and newcomers. Gives bitter taste above 250 mg/L.",
            "treatment": "Reverse osmosis, distillation, or anion exchange systems.",
        },
        {
            "param": "Conductivity",
            "icon": "⚡",
            "unit": "μS/cm",
            "safe_min": 0,
            "safe_max": 500,
            "ideal": "200 – 400",
            "color": "#fb923c",
            "description": "Measures water's ability to conduct electricity — directly related to dissolved ion concentration.",
            "health_effects": "High conductivity (>800) indicates excess minerals or contamination. Very low (<50) means demineralized water lacking essential minerals.",
            "treatment": "RO filtration or deionization to reduce. For too-low conductivity, mineral cartridges can add back essential minerals.",
        },
        {
            "param": "Organic Carbon",
            "icon": "🌿",
            "unit": "ppm",
            "safe_min": 0,
            "safe_max": 2.0,
            "ideal": "< 2.0",
            "color": "#4ade80",
            "description": "Measures total organic matter in water — from decaying vegetation, agricultural runoff, and sewage.",
            "health_effects": "High organic carbon reacts with chlorine disinfectants to form harmful disinfection byproducts (like trihalomethanes). Indicates possible microbial contamination.",
            "treatment": "Activated carbon filtration, enhanced coagulation, UV oxidation, or ozone treatment.",
        },
        {
            "param": "Trihalomethanes",
            "icon": "☢️",
            "unit": "μg/L",
            "safe_min": 0,
            "safe_max": 80,
            "ideal": "< 40",
            "color": "#f472b6",
            "description": "Disinfection byproducts formed when chlorine reacts with natural organic matter in water.",
            "health_effects": "Long-term exposure above 80 μg/L linked to increased cancer risk (bladder, colon), liver and kidney damage, and reproductive issues.",
            "treatment": "Granular activated carbon (GAC) filters. Reduce at source by using UV or ozone disinfection instead of chlorine.",
        },
        {
            "param": "Turbidity",
            "icon": "🌫️",
            "unit": "NTU",
            "safe_min": 0,
            "safe_max": 4.0,
            "ideal": "< 1.0",
            "color": "#94a3b8",
            "description": "Measures water cloudiness caused by suspended particles — clay, silt, algae, and microorganisms.",
            "health_effects": "High turbidity shields pathogens from disinfection, increases risk of waterborne diseases. Above 5 NTU is visibly cloudy.",
            "treatment": "Coagulation + flocculation (alum/ferric chloride), followed by sedimentation and sand/membrane filtration.",
        },
    ]
    return render_template('guidelines.html', standards=standards)


@app.route('/upload')
def upload():
    """Dataset upload page."""
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle file upload and run full analysis pipeline."""
    if 'dataset' not in request.files:
        return "No file uploaded", 400

    file = request.files['dataset']
    if file.filename == '':
        return "No file selected", 400

    # Save to temp location
    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    try:
        from data_analyzer import run_full_analysis
        results = run_full_analysis(filepath)

        return render_template(
            'analysis.html',
            overview=results["overview"],
            chart_missing=results["chart_missing"],
            chart_correlation=results["chart_correlation"],
            chart_distributions=results["chart_distributions"],
            training=results["training"],
        )
    except Exception as e:
        import traceback
        return f"<pre>Analysis error:\n{traceback.format_exc()}</pre>", 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/batch')
def batch():
    """Batch prediction page."""
    return render_template('batch.html')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction via CSV upload."""
    if 'dataset' not in request.files:
        return "No file uploaded", 400

    file = request.files['dataset']
    if file.filename == '':
        return "No file selected", 400

    try:
        df = pd.read_csv(file)
        
        # Verify columns exist
        required = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            return f"Missing required columns: {', '.join(missing_cols)}", 400
            
        # Keep original to return
        df_out = df.copy()
        
        # Prepare for prediction
        X = df[required]
        # Impute and scale
        X_imp = pd.DataFrame(imputer.transform(X), columns=required)
        X_sc = pd.DataFrame(scaler.transform(X_imp), columns=required)
        
        # Predict using HNB Ensemble
        preds = hnb_model.predict(X_sc)
        probas = hnb_model.predict_proba(X_sc)[:, 1]
        
        df_out["Prediction"] = ["Safe" if p == 1 else "Unsafe" for p in preds]
        df_out["Confidence (%)"] = np.round(np.where(preds == 1, probas * 100, (1 - probas) * 100), 2)
        
        # Calculate WQI for all
        from wqi_calculator import analyze_water_quality
        wqis = []
        for _, row in df.iterrows():
            wqi, _ = analyze_water_quality(
                row['ph'] if not pd.isna(row['ph']) else DEFAULTS['ph'],
                row['Hardness'],
                row['Solids'] if not pd.isna(row['Solids']) else DEFAULTS['solids'],
                row['Chloramines'],
                row['Sulfate'] if not pd.isna(row['Sulfate']) else DEFAULTS['sulfate'],
                row['Conductivity'],
                row['Organic_carbon'],
                row['Trihalomethanes'] if not pd.isna(row['Trihalomethanes']) else DEFAULTS['trihalomethanes'],
                row['Turbidity'] if not pd.isna(row['Turbidity']) else DEFAULTS['turbidity']
            )
            wqis.append(wqi)
            
        df_out["WQI"] = wqis
        
        # Return CSV
        output = io.StringIO()
        df_out.to_csv(output, index=False)
        from flask import Response
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=batch_predictions.csv"}
        )
        
    except Exception as e:
        import traceback
        return f"<pre>Prediction error:\n{traceback.format_exc()}</pre>", 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chatbot endpoint using Gemini API."""
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"response": "Please provide a message."})
        
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            "You are an expert AI research assistant specializing in smart environmental technologies and predictive analytics for water management. Your knowledge is strictly based on the paper \"Towards Smarter Water Management: Predictive Analytics for Water Quality.\"\n\n"
            "Key Guidelines for your responses:\n"
            "1. Core Focus: Assist users with queries regarding forecasting the Water Quality Index (WQI) and Water Quality Classification (WQC).\n"
            "2. Data & Metrics: The dataset contains 2,005 records with seven features (including hardness, pH, sulfates, turbidity, chloramines, and potability). Data preprocessing involves mean imputation and normalization.\n"
            "3. Model Performance:\n"
            "   - For Classification (WQC): Gradient Boosting (GB) is the top performer with an optimized accuracy of 89.50%.\n"
            "   - For Regression (WQI): The Multi-Layer Perceptron (MLP) regressor achieves the best performance with an R² value of 89.8%.\n"
            "4. Hyperparameter Settings: Use the specific Grid Search optimized boundaries evaluated in the paper (e.g., N_Estimators up to 250, Criterion, Max_Depth) for Random Forest, XGBoost, Gradient Boosting, and AdaBoost.\n"
            "5. Absolute Restriction: Completely ignore the accidental references to \"emotion classification,\" \"mental health,\" or \"psychological health\" found in the original document's conclusion. Always ground the final utility in public health, environmental monitoring, and sustainable water management.\n"
            "You are a literal document-retrieval AI trained precisely on the text file \"Towards_Smarter_Water_Management_final2025.docx\". You must answer user queries by matching the exact context provided in the paper.\n\n"
            "Strict Guidelines:\n"
            "1. Maintain the precise metrics mentioned in the abstract and results (e.g., Gradient Boosting classification accuracy at 89.50% and MLP regressor R² at 89.8%).\n"
            "2. If asked about the Conclusion or Future Work, you must accurately reflect that the document transitions into discussing the practical implications of \"emotion classification,\" \"mental health diagnostics,\" and \"workplace psychology,\" despite the rest of the paper focusing on water quality parameters. Do not try to correct this anomaly; report it exactly as stated in the text.\n\n"
            f"User message: {user_message}"
        )
        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"response": "I'm currently experiencing technical difficulties connecting to my AI core. Please try again later."})



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
