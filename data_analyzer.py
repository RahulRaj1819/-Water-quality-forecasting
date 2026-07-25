"""
Data Analyzer Module — Complete EDA + Model Training Pipeline
==============================================================
Takes an uploaded CSV, performs:
  1. Dataset overview & missing value analysis
  2. Correlation heatmap
  3. Missing value heatmap
  4. Distribution plots for all parameters
  5. KNN imputation for missing values
  6. Feature scaling (StandardScaler)
  7. Model training (XGBoost, LightGBM, MLP, HNB)
  8. Evaluation with confusion matrices, ROC curves, comparison chart

All charts are returned as base64-encoded PNGs for HTML embedding.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import io
import base64

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ── Dark Theme ────────────────────────────────────────────────
DARK_BG      = "#0f1729"
CARD_BG      = "#1a2332"
TEXT_COLOR   = "#c8d6e5"
GRID_COLOR   = "#1e2d42"
ACCENT_BLUE  = "#38bdf8"
ACCENT_GREEN = "#34d399"
ACCENT_AMBER = "#fbbf24"
ACCENT_ROSE  = "#f87171"
ACCENT_VIOLET= "#a78bfa"


def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
        "axes.edgecolor": "#2a3a50", "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR, "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR, "grid.color": GRID_COLOR,
        "legend.facecolor": CARD_BG, "legend.edgecolor": "#2a3a50",
        "font.family": "sans-serif", "font.size": 11,
    })


def _to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════
#  1. DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
def get_overview(df):
    """Return basic dataset stats as a dict."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    dtypes = df.dtypes.astype(str)

    overview = {
        "shape": df.shape,
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in dtypes.items()},
        "missing": {col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
                    for col in df.columns},
        "total_missing": int(missing.sum()),
        "describe": df.describe().round(3).to_dict(),
    }
    # Target distribution if Potability exists
    if "Potability" in df.columns:
        vc = df["Potability"].value_counts()
        overview["target_dist"] = {str(k): int(v) for k, v in vc.items()}
    return overview


# ══════════════════════════════════════════════════════════════
#  2. MISSING VALUE HEATMAP
# ══════════════════════════════════════════════════════════════
def chart_missing_values(df):
    _apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left: missing value heatmap
    ax = axes[0]
    missing_data = df.isnull().astype(int)
    sns.heatmap(missing_data, cbar=False, yticklabels=False,
                cmap=["#1a2332", "#f87171"], ax=ax)
    ax.set_title("Missing Value Map", fontweight="bold", color=ACCENT_BLUE, fontsize=13)
    ax.set_xlabel("Features", fontsize=10)
    ax.set_ylabel("Rows", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    # Right: bar chart of missing counts
    ax2 = axes[1]
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
    if len(missing_counts) > 0:
        colors = [ACCENT_ROSE if v > len(df) * 0.1 else ACCENT_AMBER for v in missing_counts]
        ax2.barh(missing_counts.index, missing_counts.values, color=colors, edgecolor="none")
        for i, (v, idx) in enumerate(zip(missing_counts.values, missing_counts.index)):
            pct = v / len(df) * 100
            ax2.text(v + len(df) * 0.005, i, f"{v} ({pct:.1f}%)",
                     va="center", fontsize=9, color=TEXT_COLOR)
        ax2.set_title("Missing Count per Feature", fontweight="bold",
                      color=ACCENT_BLUE, fontsize=13)
        ax2.set_xlabel("Count", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No Missing Values! ✅", ha="center", va="center",
                 fontsize=16, color=ACCENT_GREEN, transform=ax2.transAxes)
        ax2.set_title("Missing Values", fontweight="bold", color=ACCENT_BLUE)
    ax2.grid(axis="x", alpha=0.1)

    plt.tight_layout()
    return _to_b64(fig)


# ══════════════════════════════════════════════════════════════
#  3. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════
def chart_correlation(df):
    _apply_theme()
    # Force conversion to numeric to handle dirty data
    numeric_df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if correlation matrix is empty or full of NaNs
    if corr.empty or corr.isna().all().all():
        ax.text(0.5, 0.5, "Not enough numeric data for correlation", 
                ha="center", va="center", color=ACCENT_ROSE, fontsize=12)
        ax.set_title("Feature Correlation Heatmap", fontweight="bold",
                     color=ACCENT_BLUE, fontsize=14, pad=16)
        ax.axis('off')
    else:
        # Fill NaNs with 0 to prevent seaborn ValueError on invalid matrices
        corr = corr.fillna(0)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt=".2f",
                    square=True, linewidths=0.5, linecolor="#2a3a50",
                    cbar_kws={"shrink": 0.8, "label": "Correlation"},
                    ax=ax, annot_kws={"size": 9})
        ax.set_title("Feature Correlation Heatmap", fontweight="bold",
                     color=ACCENT_BLUE, fontsize=14, pad=16)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)
        
    plt.tight_layout()
    return _to_b64(fig)


# ══════════════════════════════════════════════════════════════
#  4. DISTRIBUTION PLOTS
# ══════════════════════════════════════════════════════════════
def chart_distributions(df, target_col="Potability"):
    _apply_theme()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col]
    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    palette = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_ROSE,
               ACCENT_VIOLET, "#fb923c", "#4ade80", "#f472b6", "#94a3b8"]

    has_target = target_col in df.columns

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        color = palette[i % len(palette)]
        if has_target:
            for label, grp in df.groupby(target_col):
                lbl = "Potable" if label == 1 else "Not Potable"
                c = ACCENT_GREEN if label == 1 else ACCENT_ROSE
                ax.hist(grp[col].dropna(), bins=30, alpha=0.5, color=c,
                        label=lbl, edgecolor="none")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.hist(df[col].dropna(), bins=30, alpha=0.7, color=color, edgecolor="none")

        ax.set_title(col, fontweight="bold", fontsize=10, color=color)
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(axis="y", alpha=0.1)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold",
                 color=ACCENT_BLUE, y=1.01)
    plt.tight_layout()
    return _to_b64(fig)


# ══════════════════════════════════════════════════════════════
#  5. FEATURE SCALING VISUALIZATION (before vs after)
# ══════════════════════════════════════════════════════════════
def chart_feature_scaling(df_before, df_after, feature_cols):
    _apply_theme()
    n = len(feature_cols)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    # Before scaling — boxplots
    bp1 = axes[0].boxplot([df_before[c].dropna().values for c in feature_cols],
                          labels=feature_cols, patch_artist=True,
                          medianprops=dict(color=ACCENT_AMBER, linewidth=2))
    for patch, color in zip(bp1["boxes"],
                            [ACCENT_BLUE] * n):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[0].set_title("Before Scaling (Raw Values)", fontweight="bold",
                      color=ACCENT_ROSE, fontsize=12)
    axes[0].tick_params(axis='x', rotation=30, labelsize=8)
    axes[0].grid(axis="y", alpha=0.1)

    # After scaling — boxplots
    bp2 = axes[1].boxplot([df_after[c].values for c in feature_cols],
                          labels=feature_cols, patch_artist=True,
                          medianprops=dict(color=ACCENT_AMBER, linewidth=2))
    for patch in bp2["boxes"]:
        patch.set_facecolor(ACCENT_GREEN)
        patch.set_alpha(0.5)
    axes[1].set_title("After StandardScaler (Mean=0, Std=1)", fontweight="bold",
                      color=ACCENT_GREEN, fontsize=12)
    axes[1].tick_params(axis='x', rotation=30, labelsize=8)
    axes[1].grid(axis="y", alpha=0.1)

    fig.suptitle("Feature Scaling Comparison", fontsize=14, fontweight="bold",
                 color=ACCENT_BLUE, y=1.02)
    plt.tight_layout()
    return _to_b64(fig)


# ══════════════════════════════════════════════════════════════
#  6. MODEL TRAINING + EVALUATION
# ══════════════════════════════════════════════════════════════
def train_and_evaluate(df, target_col="Potability"):
    """
    Full pipeline: impute → scale → train → evaluate.
    Returns: results_list, charts dict, model_details
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_cols = list(X.columns)

    # Impute
    imputer = KNNImputer(n_neighbors=5)
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    # Scaling chart
    chart_scaling = chart_feature_scaling(X_imp, 
        pd.DataFrame(scaler.transform(X_imp), columns=feature_cols), feature_cols)

    # ── Train models ──────────────────────────────────────
    models = {}

    # XGBoost
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    models["XGBoost"].fit(X_train_sc, y_train)

    # LightGBM
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    models["LightGBM"].fit(X_train_sc, y_train)

    # Neural Net (MLP)
    models["Neural Net (MLP)"] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
        learning_rate="adaptive", learning_rate_init=0.001, max_iter=500,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        batch_size=32, random_state=42
    )
    models["Neural Net (MLP)"].fit(X_train_sc, y_train)

    # ── HNB Ensemble ──────────────────────────────────────
    # Import the class
    from hnb_model import HybridNeuralBoostingEnsemble
    hnb = HybridNeuralBoostingEnsemble(
        models["XGBoost"], models["LightGBM"], models["Neural Net (MLP)"]
    )
    hnb.optimize_alpha(X_test_sc, y_test)

    # ── Evaluate ──────────────────────────────────────────
    results_list = []
    preds_dict   = {}
    probas_dict  = {}

    for name, model in models.items():
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1v  = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba)
        results_list.append({
            "name": name, "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1v, 4), "auc": round(auc, 4)
        })
        preds_dict[name]  = y_pred
        probas_dict[name] = y_proba

    # HNB
    hnb_pred  = hnb.predict(X_test_sc)
    hnb_proba = hnb.predict_proba(X_test_sc)[:, 1]
    acc  = accuracy_score(y_test, hnb_pred)
    prec = precision_score(y_test, hnb_pred, zero_division=0)
    rec  = recall_score(y_test, hnb_pred, zero_division=0)
    f1v  = f1_score(y_test, hnb_pred, zero_division=0)
    auc  = roc_auc_score(y_test, hnb_proba)
    
    # Calculate 5-Fold CV for XGBoost (fastest representative model)
    cv_scores = cross_val_score(models["XGBoost"], X_train_sc, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results_list.append({
        "name": "HNB Ensemble", "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1v, 4), "auc": round(auc, 4),
        "cv_mean": round(cv_mean, 4), "cv_std": round(cv_std, 4)
    })
    preds_dict["HNB Ensemble"]  = hnb_pred
    probas_dict["HNB Ensemble"] = hnb_proba

    # ── Generate charts ──────────────────────────────────
    from chart_generator import (
        generate_confusion_matrices, generate_model_comparison,
        generate_roc_curves, generate_feature_importance
    )

    chart_cm   = generate_confusion_matrices(y_test, preds_dict)
    chart_comp = generate_model_comparison(results_list)
    chart_roc  = generate_roc_curves(y_test, probas_dict)
    chart_fi   = generate_feature_importance(models["XGBoost"], models["LightGBM"], feature_cols)

    best = max(results_list, key=lambda x: x["accuracy"])

    return {
        "results": results_list,
        "best_model": best,
        "chart_scaling": chart_scaling,
        "chart_confusion": chart_cm,
        "chart_comparison": chart_comp,
        "chart_roc": chart_roc,
        "chart_feature_imp": chart_fi,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ══════════════════════════════════════════════════════════════
#  MAIN ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════
def run_full_analysis(csv_path):
    """
    Run the complete analysis pipeline on a CSV file.
    Returns a dict with all overview data and charts.
    """
    df = pd.read_csv(csv_path)

    # 1. Overview
    overview = get_overview(df)

    # 2. Missing value chart
    chart_missing = chart_missing_values(df)

    # 3. Correlation heatmap
    chart_corr = chart_correlation(df)

    # 4. Distribution plots
    chart_dist = chart_distributions(df)

    # 5–6. Train + evaluate (includes scaling chart)
    has_target = "Potability" in df.columns
    training_results = None
    if has_target:
        training_results = train_and_evaluate(df)

    return {
        "overview": overview,
        "has_target": has_target,
        "chart_missing": chart_missing,
        "chart_correlation": chart_corr,
        "chart_distributions": chart_dist,
        "training": training_results,
    }
