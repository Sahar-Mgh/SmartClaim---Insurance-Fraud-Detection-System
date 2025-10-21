"""
Shared utilities for model training and evaluation.
Includes preprocessing, splitting, and metric calculation functions.
"""

from typing import Tuple, Dict, Any
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve
)


# Feature definitions
NUMERIC_FEATURES = [
    "age",
    "vehicle_age",
    "claim_amount",
    "num_prior_claims",
    "policy_tenure_months",
    "reported_delay_days"
]

CATEGORICAL_FEATURES = [
    "accident_type",
    "region"
]

BINARY_FEATURES = [
    "has_police_report"
]

# All feature columns (excluding target)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TARGET_COLUMN = "is_fraud"


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Uses stratified split to maintain class balance in both sets.
    
    Args:
        df: Full dataset
        test_size: Proportion for test set (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Keep class balance in both sets
    )
    
    print(f"Train set: {len(X_train)} rows ({y_train.mean():.2%} fraud)")
    print(f"Test set:  {len(X_test)} rows ({y_test.mean():.2%} fraud)")
    
    return X_train, X_test, y_train, y_test


def create_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessing pipeline for features.
    
    - Numeric features: StandardScaler (zero mean, unit variance)
    - Categorical features: OneHotEncoder
    - Binary features: pass through (already 0/1)
    
    Returns:
        ColumnTransformer ready to fit and transform data
    """
    preprocessor = ColumnTransformer(
        transformers=[
            # Scale numeric features
            ("num", StandardScaler(), NUMERIC_FEATURES),
            # One-hot encode categorical features
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
            # Pass through binary features (already 0/1)
            ("bin", "passthrough", BINARY_FEATURES)
        ],
        remainder="drop"  # Drop any other columns
    )
    
    return preprocessor


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    For imbalanced datasets, PR-AUC (precision-recall) is often
    more informative than ROC-AUC.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_pred_proba: Predicted probabilities (0 to 1)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        "accuracy": (y_true == y_pred).mean(),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print metrics in a nice format.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name to display for this model
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Performance")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:15s}: {value:.4f}")
    print(f"{'='*50}\n")


def save_metrics(
    metrics: Dict[str, float],
    output_path: str,
    model_name: str
):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
        model_name: Name of the model
    """
    output_dict = {
        "model": model_name,
        "metrics": metrics
    }
    
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"[OK] Saved metrics to: {output_path}")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = None
):
    """
    Print and optionally save a detailed classification report.
    
    Shows precision, recall, f1 for each class and overall.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Optional path to save report as text file
    """
    report = classification_report(
        y_true, y_pred,
        target_names=["Legitimate", "Fraud"],
        digits=4
    )
    
    print("\nDetailed Classification Report:")
    print(report)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"[OK] Saved classification report to: {output_path}")


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Get feature names after preprocessing transformation.
    
    Useful for interpreting model coefficients or feature importances.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        List of feature names in order
    """
    feature_names = []
    
    # Numeric features (unchanged names)
    feature_names.extend(NUMERIC_FEATURES)
    
    # Categorical features (get one-hot encoded names)
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_feature_names)
    
    # Binary features (unchanged names)
    feature_names.extend(BINARY_FEATURES)
    
    return feature_names

