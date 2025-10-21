"""
Load insurance claims data from CSV files.
By default loads the synthetic dataset, but can be adapted for external data.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_claims_data(
    data_path: Optional[str] = None,
    use_synthetic: bool = True
) -> pd.DataFrame:
    """
    Load insurance claims data from a CSV file.
    
    By default, loads the synthetic dataset from data/processed/.
    You can provide a custom path to load external data instead.
    
    Args:
        data_path: Path to CSV file. If None, uses default synthetic data path.
        use_synthetic: If True and data_path is None, loads synthetic data.
        
    Returns:
        DataFrame with claims data
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
    """
    if data_path is None and use_synthetic:
        # Default to synthetic data
        data_path = "data/processed/synthetic_claims.csv"
    
    if data_path is None:
        raise ValueError("Must provide data_path or set use_synthetic=True")
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Run: python -m src.data.generate_synthetic to create synthetic data"
        )
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Basic validation
    required_columns = [
        "age", "vehicle_age", "claim_amount", "accident_type",
        "num_prior_claims", "region", "policy_tenure_months",
        "reported_delay_days", "has_police_report", "is_fraud"
    ]
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
    
    return df


def validate_input_features(df: pd.DataFrame) -> bool:
    """
    Validate that input features are in expected ranges.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if validation passes, raises ValueError otherwise
    """
    checks = [
        (df["age"] >= 18, "age must be >= 18"),
        (df["age"] <= 100, "age must be <= 100"),
        (df["vehicle_age"] >= 0, "vehicle_age must be >= 0"),
        (df["claim_amount"] > 0, "claim_amount must be > 0"),
        (df["num_prior_claims"] >= 0, "num_prior_claims must be >= 0"),
        (df["policy_tenure_months"] > 0, "policy_tenure_months must be > 0"),
        (df["reported_delay_days"] >= 0, "reported_delay_days must be >= 0"),
        (df["has_police_report"].isin([0, 1]), "has_police_report must be 0 or 1"),
    ]
    
    for condition, message in checks:
        if not condition.all():
            raise ValueError(f"Validation failed: {message}")
    
    return True


if __name__ == "__main__":
    # Quick test
    df = load_claims_data()
    validate_input_features(df)
    print("\n[OK] Data validation passed")
    print(df.info())

