"""
Generate SHAP explanations for model predictions.
Creates global feature importance and local explanation visualizations.
"""

import argparse
import sys
from pathlib import Path
import joblib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import load_claims_data
from src.models.utils import split_data


def explain_model_global(
    model,
    X_test: pd.DataFrame,
    model_name: str,
    save_path: Path,
    max_samples: int = 500
):
    """
    Generate global SHAP feature importance.
    
    Shows which features are most important overall for the model's predictions.
    
    Args:
        model: Trained model pipeline
        X_test: Test data
        model_name: Name for plot title
        save_path: Where to save plot
        max_samples: Max samples to use (SHAP can be slow on large datasets)
    """
    print(f"\nGenerating global SHAP explanations for {model_name}...")
    
    # Limit samples for computational efficiency
    if len(X_test) > max_samples:
        X_sample = X_test.sample(n=max_samples, random_state=42)
        print(f"  Using {max_samples} samples (SHAP is computationally intensive)")
    else:
        X_sample = X_test
    
    # Transform data through preprocessing
    X_transformed = model.named_steps["preprocessor"].transform(X_sample)
    
    # Get the actual classifier
    classifier = model.named_steps["classifier"]
    
    # Create SHAP explainer
    # For tree-based models, use TreeExplainer (faster)
    # For linear models, use LinearExplainer
    try:
        if hasattr(classifier, "tree_method"):  # XGBoost
            explainer = shap.TreeExplainer(classifier)
        else:  # Logistic Regression
            explainer = shap.LinearExplainer(classifier, X_transformed)
        
        # Calculate SHAP values
        print("  Calculating SHAP values...")
        shap_values = explainer.shap_values(X_transformed)
        
        # For binary classification, some explainers return values for both classes
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (fraud)
        
        # Get feature names
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        
        # Create summary plot (bar chart showing mean absolute SHAP values)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=15  # Show top 15 features
        )
        plt.title(f"SHAP Feature Importance: {model_name}", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"[OK] Saved global SHAP plot to: {save_path}")
        
        # Print top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        return explainer, shap_values, X_transformed, feature_names
        
    except Exception as e:
        print(f"[WARNING] Error generating SHAP explanations: {e}")
        print("  This is normal for some model types. Continuing...")
        return None, None, None, None


def explain_single_prediction(
    explainer,
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list,
    sample_idx: int,
    model_name: str,
    save_path: Path
):
    """
    Generate local SHAP explanation for a single prediction.
    
    Shows which features contributed most to this specific prediction.
    
    Args:
        explainer: SHAP explainer object
        shap_values: Computed SHAP values
        X_transformed: Transformed feature matrix
        feature_names: List of feature names
        sample_idx: Which sample to explain
        model_name: Name for plot title
        save_path: Where to save plot
    """
    if explainer is None:
        print("[WARNING] Skipping single prediction explanation (no explainer available)")
        return
    
    print(f"\nGenerating local SHAP explanation for sample {sample_idx}...")
    
    try:
        # Create waterfall plot (shows how features add up to final prediction)
        plt.figure(figsize=(10, 8))
        
        # Use waterfall plot if available (SHAP >= 0.40)
        try:
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=explainer.expected_value if hasattr(explainer, "expected_value") else 0,
                    data=X_transformed[sample_idx],
                    feature_names=feature_names
                ),
                show=False,
                max_display=10
            )
        except:
            # Fallback to force plot
            shap.force_plot(
                explainer.expected_value if hasattr(explainer, "expected_value") else 0,
                shap_values[sample_idx],
                X_transformed[sample_idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
        
        plt.title(f"SHAP Explanation for Sample {sample_idx}: {model_name}", 
                 fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"[OK] Saved local SHAP plot to: {save_path}")
        
        # Print top contributing features for this sample
        feature_contributions = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_values[sample_idx]
        }).sort_values("shap_value", key=abs, ascending=False)
        
        print(f"\nTop features for sample {sample_idx}:")
        for idx, row in feature_contributions.head(5).iterrows():
            direction = "increases" if row["shap_value"] > 0 else "decreases"
            print(f"  {row['feature']:30s}: {row['shap_value']:+.4f} ({direction} fraud risk)")
        
    except Exception as e:
        print(f"[WARNING] Error generating local explanation: {e}")


def main():
    """Main SHAP explanation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for fraud detection models"
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
        "--sample_idx",
        type=int,
        default=10,
        help="Sample index for local explanation"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (must match training)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SmartClaim SHAP Explanations")
    print("="*60)
    print("\nSHAP (SHapley Additive exPlanations) helps us understand:")
    print("  1. Which features matter most (global)")
    print("  2. How features affect specific predictions (local)")
    
    # Load data and split
    df = load_claims_data(args.data_path)
    _, X_test, _, y_test = split_data(df, random_state=args.random_state)
    
    artifacts_dir = Path(args.artifacts_dir)
    plots_dir = artifacts_dir / "reports" / "plots"
    
    # Explain XGBoost model (usually more interesting than baseline)
    xgboost_path = artifacts_dir / "models" / "xgboost_model.pkl"
    
    if xgboost_path.exists():
        model = joblib.load(xgboost_path)
        print(f"\n[OK] Loaded model from: {xgboost_path}")
        
        # Global explanation
        explainer, shap_values, X_transformed, feature_names = explain_model_global(
            model, X_test, "XGBoost",
            plots_dir / "xgboost_shap_importance.png"
        )
        
        # Local explanation
        if explainer is not None and args.sample_idx < len(X_test):
            explain_single_prediction(
                explainer, shap_values, X_transformed, feature_names,
                args.sample_idx, "XGBoost",
                plots_dir / f"xgboost_shap_sample_{args.sample_idx}.png"
            )
        
    else:
        print(f"[WARNING] XGBoost model not found at: {xgboost_path}")
        print("  Run training first: python -m src.models.train")
    
    print("\n" + "="*60)
    print("SHAP Explanations Complete!")
    print("="*60)
    print(f"Plots saved to: {plots_dir}")
    print("\nNext step: Try the interactive app")
    print("  streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()

