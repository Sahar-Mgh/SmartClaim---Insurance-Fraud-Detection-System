"""
Evaluate trained models and generate visualization plots.
Creates ROC curves, PR curves, and confusion matrices.
"""

import argparse
import sys
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import load_claims_data
from src.models.utils import split_data, calculate_metrics, print_metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path
):
    """
    Plot and save confusion matrix.
    
    Shows how many predictions were correct vs incorrect for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name for plot title
        save_path: Where to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        cbar_kws={"label": "Count"}
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[OK] Saved confusion matrix to: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    save_path: Path
):
    """
    Plot and save ROC curve.
    
    ROC (Receiver Operating Characteristic) shows the trade-off between
    true positive rate and false positive rate at different thresholds.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name for plot title
        save_path: Where to save the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve: {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[OK] Saved ROC curve to: {save_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    save_path: Path
):
    """
    Plot and save Precision-Recall curve.
    
    PR curve is often more informative than ROC for imbalanced datasets.
    It focuses on the performance on the positive (fraud) class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name for plot title
        save_path: Where to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision,
        color="darkgreen",
        lw=2,
        label=f"PR curve (AUC = {pr_auc:.3f})"
    )
    
    # Baseline: fraud rate in dataset
    baseline = y_true.mean()
    plt.axhline(y=baseline, color="navy", linestyle="--", lw=2, label=f"Baseline ({baseline:.3f})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve: {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[OK] Saved PR curve to: {save_path}")


def evaluate_model(
    model_path: Path,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path
):
    """
    Load a model and evaluate it with plots.
    
    Args:
        model_path: Path to saved model
        model_name: Name for labeling
        X_test: Test features
        y_test: Test labels
        plots_dir: Directory to save plots
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    model = joblib.load(model_path)
    print(f"[OK] Loaded model from: {model_path}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics, model_name)
    
    # Generate plots
    plot_confusion_matrix(
        y_test, y_pred, model_name,
        plots_dir / f"{model_name}_confusion_matrix.png"
    )
    plot_roc_curve(
        y_test, y_pred_proba, model_name,
        plots_dir / f"{model_name}_roc_curve.png"
    )
    plot_precision_recall_curve(
        y_test, y_pred_proba, model_name,
        plots_dir / f"{model_name}_pr_curve.png"
    )


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained fraud detection models"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/synthetic_claims.csv",
        help="Path to data CSV"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory with saved models"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (must match training)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SmartClaim Model Evaluation")
    print("="*60)
    
    # Load data and split (must use same random state as training!)
    df = load_claims_data(args.data_path)
    _, X_test, _, y_test = split_data(df, random_state=args.random_state)
    
    artifacts_dir = Path(args.artifacts_dir)
    plots_dir = artifacts_dir / "reports" / "plots"
    
    # Evaluate baseline model
    baseline_path = artifacts_dir / "models" / "baseline_model.pkl"
    if baseline_path.exists():
        evaluate_model(baseline_path, "Baseline (Logistic Regression)", X_test, y_test, plots_dir)
    else:
        print(f"[WARNING] Baseline model not found at: {baseline_path}")
    
    # Evaluate XGBoost model
    xgboost_path = artifacts_dir / "models" / "xgboost_model.pkl"
    if xgboost_path.exists():
        evaluate_model(xgboost_path, "XGBoost", X_test, y_test, plots_dir)
    else:
        print(f"[WARNING] XGBoost model not found at: {xgboost_path}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Plots saved to: {plots_dir}")
    print("\nNext step: Generate SHAP explanations")
    print("  python -m src.models.explain")


if __name__ == "__main__":
    main()

