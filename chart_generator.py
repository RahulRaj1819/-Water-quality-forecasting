"""
Chart Generator for Water Quality Prediction
=============================================
Generates base64-encoded PNG charts for embedding in HTML results.
All charts use a dark theme to match the application's UI.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import io
import base64


# ── Dark Theme Configuration ──────────────────────────────────
DARK_BG      = "#0f1729"
CARD_BG      = "#1a2332"
TEXT_COLOR    = "#c8d6e5"
GRID_COLOR   = "#1e2d42"
ACCENT_BLUE  = "#38bdf8"
ACCENT_GREEN = "#34d399"
ACCENT_AMBER = "#fbbf24"
ACCENT_ROSE  = "#f87171"
ACCENT_VIOLET= "#a78bfa"
MODEL_COLORS = {
    "XGBoost":          "#38bdf8",
    "LightGBM":         "#34d399",
    "Neural Net (MLP)": "#fbbf24",
    "HNB Ensemble":     "#a78bfa",
    "SVM":              "#f87171",
}


def _apply_dark_theme():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    "#2a3a50",
        "axes.labelcolor":   TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "grid.color":        GRID_COLOR,
        "legend.facecolor":  CARD_BG,
        "legend.edgecolor":  "#2a3a50",
        "font.family":       "sans-serif",
        "font.size":         11,
    })


def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── 1. Confusion Matrices ────────────────────────────────────
def generate_confusion_matrices(y_true, predictions_dict):
    """
    Generate a side-by-side confusion matrix chart for all models.
    predictions_dict: {model_name: y_pred_array}
    Returns base64 PNG string.
    """
    _apply_dark_theme()
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        color = MODEL_COLORS.get(name, ACCENT_BLUE)

        # Draw heatmap manually for dark theme
        im = ax.imshow(cm, cmap="Blues", aspect="auto", alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Potable", "Potable"], fontsize=9)
        ax.set_yticklabels(["Not Potable", "Potable"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel("Actual", fontsize=10, color=TEXT_COLOR)
        ax.set_title(name, fontsize=12, fontweight="bold", color=color, pad=10)

        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                txt_color = "#ffffff" if val > cm.max() * 0.5 else TEXT_COLOR
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=16, fontweight="bold", color=txt_color)

    fig.suptitle("Confusion Matrices — All Models", fontsize=14,
                 fontweight="bold", color=ACCENT_BLUE, y=1.02)
    plt.tight_layout()
    return _fig_to_base64(fig)


# ── 2. Model Comparison Bar Chart ────────────────────────────
def generate_model_comparison(results_list):
    """
    Generate a grouped bar chart comparing metrics across models.
    results_list: list of dicts with keys: name, accuracy, precision, recall, f1, auc
    Returns base64 PNG string.
    """
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    names = [r["name"] for r in results_list]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, ACCENT_ROSE, ACCENT_VIOLET]

    x = np.arange(len(names))
    width = 0.14
    offsets = np.arange(len(metrics)) - len(metrics) / 2 + 0.5

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [r[metric] for r in results_list]
        bars = ax.bar(x + offsets[i] * width, values, width, label=label,
                      color=color, alpha=0.9, edgecolor="none", zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.set_title("Model Performance Comparison", fontsize=14,
                 fontweight="bold", color=ACCENT_BLUE, pad=14)
    plt.tight_layout()
    return _fig_to_base64(fig)


# ── 3. ROC Curves ────────────────────────────────────────────
def generate_roc_curves(y_true, probas_dict):
    """
    Generate ROC curves for all models.
    probas_dict: {model_name: y_proba_array}
    Returns base64 PNG string.
    """
    _apply_dark_theme()
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for name, proba in probas_dict.items():
        color = MODEL_COLORS.get(name, ACCENT_BLUE)
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_val = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {auc_val:.4f})", alpha=0.9)

    ax.plot([0, 1], [0, 1], "--", color="#4a5568", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14,
                 fontweight="bold", color=ACCENT_BLUE, pad=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.12)
    plt.tight_layout()
    return _fig_to_base64(fig)


# ── 4. Feature Importance ────────────────────────────────────
def generate_feature_importance(xgb_model, lgbm_model, feature_names):
    """
    Generate horizontal bar chart of feature importance from boosting models.
    Returns base64 PNG string.
    """
    _apply_dark_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # XGBoost
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values()
    axes[0].barh(xgb_imp.index, xgb_imp.values, color=ACCENT_BLUE, alpha=0.85, edgecolor="none")
    axes[0].set_title("XGBoost", fontsize=12, fontweight="bold", color=ACCENT_BLUE)
    axes[0].set_xlabel("Importance", fontsize=10)

    # LightGBM
    lgbm_imp = pd.Series(lgbm_model.feature_importances_, index=feature_names).sort_values()
    axes[1].barh(lgbm_imp.index, lgbm_imp.values, color=ACCENT_GREEN, alpha=0.85, edgecolor="none")
    axes[1].set_title("LightGBM", fontsize=12, fontweight="bold", color=ACCENT_GREEN)
    axes[1].set_xlabel("Importance", fontsize=10)

    fig.suptitle("Feature Importance Analysis", fontsize=14,
                 fontweight="bold", color=ACCENT_BLUE, y=1.01)
    plt.tight_layout()
    return _fig_to_base64(fig)


# ── 5. Individual Prediction Gauge / Radar ────────────────────
def generate_prediction_gauge(model_predictions):
    """
    Generate a horizontal bar showing each model's prediction (Safe/Unsafe)
    and confidence level.
    model_predictions: list of dicts: {name, prediction, confidence, proba_safe}
    Returns base64 PNG string.
    """
    _apply_dark_theme()
    n = len(model_predictions)
    fig, ax = plt.subplots(figsize=(8, max(2.5, n * 0.8 + 0.5)))

    names = [m["name"] for m in model_predictions]
    confidences = [m["confidence"] for m in model_predictions]
    proba_safe = [m["proba_safe"] for m in model_predictions]
    colors = []
    for m in model_predictions:
        if m["prediction"] == 1:
            colors.append(ACCENT_GREEN)
        else:
            colors.append(ACCENT_ROSE)

    y_pos = np.arange(n)
    bars = ax.barh(y_pos, proba_safe, color=colors, alpha=0.85,
                   edgecolor="none", height=0.55, zorder=3)

    for i, (bar, m) in enumerate(zip(bars, model_predictions)):
        label = "Safe" if m["prediction"] == 1 else "Unsafe"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{label}  ({m["confidence"]:.1f}%)',
                va="center", fontsize=10, fontweight="bold",
                color=ACCENT_GREEN if m["prediction"] == 1 else ACCENT_ROSE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Probability of Potable (Safe)", fontsize=11)
    ax.axvline(x=0.5, color="#4a5568", linestyle="--", lw=1, alpha=0.5, zorder=2)
    ax.text(0.5, n - 0.15, "Threshold", ha="center", fontsize=8, color="#4a5568")
    ax.set_title("Per-Model Prediction for Your Sample", fontsize=14,
                 fontweight="bold", color=ACCENT_BLUE, pad=14)
    ax.grid(axis="x", alpha=0.1, zorder=0)
    ax.invert_yaxis()
    plt.tight_layout()
    return _fig_to_base64(fig)
