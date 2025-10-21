"""
Train baseline and improved models for fraud detection.
Saves trained models and preprocessing artifacts.
"""

import argparse
import sys
from pathlib import Path
import joblib

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import load_claims_data
from src.models.utils import (
    split_data,
    create_preprocessor,
    calculate_metrics,
    print_metrics,
    save_metrics,
    print_classification_report
)


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Pipeline:
    """
    Train a baseline Logistic Regression model.
    
    Uses class_weight='balanced' to handle class imbalance.
    This automatically adjusts weights inversely proportional to class frequencies.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        Trained sklearn Pipeline
    """
    print("\n" + "="*60)
    print("Training Baseline Model: Logistic Regression")
    print("="*60)
    
    # Create preprocessing + model pipeline
    preprocessor = create_preprocessor()
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",  # Handle imbalanced classes
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        ))
    ])
    
    # Fit the model
    print("Fitting model...")
    model.fit(X_train, y_train)
    print("[OK] Training complete")
    
    return model


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Pipeline:
    """
    Train an improved XGBoost model with light tuning.
    
    XGBoost often performs better than logistic regression because:
    - It can capture non-linear relationships
    - It handles feature interactions automatically
    - scale_pos_weight handles class imbalance well
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        Trained sklearn Pipeline with XGBoost
    """
    print("\n" + "="*60)
    print("Training Improved Model: XGBoost")
    print("="*60)
    
    # Calculate scale_pos_weight for imbalanced data
    # This tells XGBoost to pay more attention to the minority class
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}")
    print(f"Setting scale_pos_weight={scale_pos_weight:.2f}")
    
    # Create preprocessing + model pipeline
    preprocessor = create_preprocessor()
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            n_estimators=100,          # Number of trees (kept small for speed)
            max_depth=5,                # Maximum tree depth
            learning_rate=0.1,          # Step size shrinkage
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            random_state=random_state,
            eval_metric="logloss",      # Metric to optimize
            use_label_encoder=False
        ))
    ])
    
    # Fit the model
    print("Fitting model...")
    model.fit(X_train, y_train)
    print("[OK] Training complete")
    
    return model


def evaluate_and_save_model(
    model: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    artifacts_dir: Path
):
    """
    Evaluate model on train and test sets, then save everything.
    
    Args:
        model: Trained model pipeline
        model_name: Name for saving (e.g., "baseline" or "xgboost")
        X_train, y_train: Training data
        X_test, y_test: Test data
        artifacts_dir: Directory to save artifacts
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    # Print results
    print_metrics(train_metrics, f"{model_name} - Train Set")
    print_metrics(test_metrics, f"{model_name} - Test Set")
    
    # Check for overfitting
    train_test_gap = train_metrics["f1"] - test_metrics["f1"]
    if train_test_gap > 0.1:
        print(f"[WARNING] Large train-test gap ({train_test_gap:.3f}) suggests overfitting")
    
    # Save model
    model_path = artifacts_dir / "models" / f"{model_name}_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[OK] Saved model to: {model_path}")
    
    # Save metrics
    metrics_path = artifacts_dir / "reports" / f"{model_name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(
        {"train": train_metrics, "test": test_metrics},
        str(metrics_path),
        model_name
    )
    
    # Save classification report
    report_path = artifacts_dir / "reports" / f"{model_name}_classification_report.txt"
    print_classification_report(y_test, y_test_pred, str(report_path))


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train fraud detection models"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/synthetic_claims.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory to save model artifacts"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SmartClaim Model Training Pipeline")
    print("="*60)
    
    # Load data
    df = load_claims_data(args.data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, random_state=args.random_state)
    
    artifacts_dir = Path(args.artifacts_dir)
    
    # Train baseline model
    baseline_model = train_baseline_model(X_train, y_train, args.random_state)
    evaluate_and_save_model(
        baseline_model, "baseline",
        X_train, y_train, X_test, y_test,
        artifacts_dir
    )
    
    # Train XGBoost model
    xgboost_model = train_xgboost_model(X_train, y_train, args.random_state)
    evaluate_and_save_model(
        xgboost_model, "xgboost",
        X_train, y_train, X_test, y_test,
        artifacts_dir
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Models saved to: {artifacts_dir / 'models'}")
    print(f"Reports saved to: {artifacts_dir / 'reports'}")
    print("\nNext steps:")
    print("  1. Run evaluation: python -m src.models.evaluate")
    print("  2. Generate SHAP explanations: python -m src.models.explain")
    print("  3. Try the app: streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()

