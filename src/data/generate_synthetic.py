"""
Generate synthetic insurance claims data for fraud detection.
Creates realistic-looking tabular data with mild correlations and class imbalance.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def generate_synthetic_claims(
    n_rows: int = 5000,
    fraud_rate: float = 0.12,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic insurance claims dataset.
    
    Creates a realistic dataset with:
    - Numeric features: age, vehicle_age, claim_amount, etc.
    - Categorical features: accident_type, region
    - Binary features: has_police_report
    - Target: is_fraud (0 or 1)
    
    Args:
        n_rows: Number of rows to generate
        fraud_rate: Proportion of fraudulent claims (0 to 1)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic claims data
    """
    np.random.seed(random_state)
    
    # Calculate how many fraudulent vs legitimate claims
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud
    
    print(f"Generating {n_rows} claims ({n_fraud} fraud, {n_legit} legitimate)...")
    
    # --- Generate legitimate claims first ---
    legit_age = np.random.normal(45, 12, n_legit).clip(18, 85)
    legit_vehicle_age = np.random.exponential(5, n_legit).clip(0, 25)
    legit_claim_amount = np.random.lognormal(8.5, 0.8, n_legit).clip(500, 100000)
    legit_num_prior_claims = np.random.poisson(0.5, n_legit).clip(0, 10)
    legit_policy_tenure = np.random.exponential(36, n_legit).clip(1, 240)
    legit_reported_delay = np.random.exponential(2, n_legit).clip(0, 30)
    
    # Legitimate claims are more likely to have police reports
    legit_has_police = np.random.binomial(1, 0.7, n_legit)
    
    # Categorical features (legitimate distribution)
    legit_accident_type = np.random.choice(
        ["collision", "theft", "vandalism", "weather"],
        n_legit,
        p=[0.5, 0.2, 0.15, 0.15]
    )
    legit_region = np.random.choice(
        ["north", "south", "east", "west"],
        n_legit,
        p=[0.25, 0.25, 0.25, 0.25]
    )
    
    # --- Generate fraudulent claims (with different distributions) ---
    # Fraudsters tend to be younger on average
    fraud_age = np.random.normal(38, 10, n_fraud).clip(18, 85)
    
    # Fraudulent claims often involve older vehicles
    fraud_vehicle_age = np.random.exponential(8, n_fraud).clip(0, 25)
    
    # Fraudulent claims tend to be higher amounts
    fraud_claim_amount = np.random.lognormal(9.2, 0.9, n_fraud).clip(500, 100000)
    
    # Fraudsters often have more prior claims
    fraud_num_prior_claims = np.random.poisson(1.8, n_fraud).clip(0, 10)
    
    # Fraudulent claims may have shorter policy tenure
    fraud_policy_tenure = np.random.exponential(18, n_fraud).clip(1, 240)
    
    # Fraudulent claims are reported with longer delays
    fraud_reported_delay = np.random.exponential(5, n_fraud).clip(0, 30)
    
    # Fraudulent claims are less likely to have police reports
    fraud_has_police = np.random.binomial(1, 0.3, n_fraud)
    
    # Fraudulent accident types (more theft and vandalism)
    fraud_accident_type = np.random.choice(
        ["collision", "theft", "vandalism", "weather"],
        n_fraud,
        p=[0.3, 0.35, 0.25, 0.1]
    )
    fraud_region = np.random.choice(
        ["north", "south", "east", "west"],
        n_fraud,
        p=[0.25, 0.25, 0.25, 0.25]
    )
    
    # --- Combine legitimate and fraudulent data ---
    df = pd.DataFrame({
        "age": np.concatenate([legit_age, fraud_age]),
        "vehicle_age": np.concatenate([legit_vehicle_age, fraud_vehicle_age]),
        "claim_amount": np.concatenate([legit_claim_amount, fraud_claim_amount]),
        "accident_type": np.concatenate([legit_accident_type, fraud_accident_type]),
        "num_prior_claims": np.concatenate([legit_num_prior_claims, fraud_num_prior_claims]),
        "region": np.concatenate([legit_region, fraud_region]),
        "policy_tenure_months": np.concatenate([legit_policy_tenure, fraud_policy_tenure]),
        "reported_delay_days": np.concatenate([legit_reported_delay, fraud_reported_delay]),
        "has_police_report": np.concatenate([legit_has_police, fraud_has_police]),
        "is_fraud": np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    })
    
    # Shuffle the dataset so fraud and legit are mixed
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Round numeric columns for readability
    df["age"] = df["age"].round(0).astype(int)
    df["vehicle_age"] = df["vehicle_age"].round(1)
    df["claim_amount"] = df["claim_amount"].round(2)
    df["num_prior_claims"] = df["num_prior_claims"].astype(int)
    df["policy_tenure_months"] = df["policy_tenure_months"].round(0).astype(int)
    df["reported_delay_days"] = df["reported_delay_days"].round(1)
    df["has_police_report"] = df["has_police_report"].astype(int)
    df["is_fraud"] = df["is_fraud"].astype(int)
    
    print(f"[OK] Generated {len(df)} rows")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"  Features: {list(df.columns)}")
    
    return df


def main():
    """CLI entry point for generating synthetic data."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic insurance claims data"
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=5000,
        help="Number of rows to generate (default: 5000)"
    )
    parser.add_argument(
        "--fraud_rate",
        type=float,
        default=0.12,
        help="Proportion of fraudulent claims (default: 0.12)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/synthetic_claims.csv",
        help="Output CSV path (default: data/processed/synthetic_claims.csv)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Generate data
    df = generate_synthetic_claims(
        n_rows=args.n_rows,
        fraud_rate=args.fraud_rate,
        random_state=args.random_state
    )
    
    # Save to CSV
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()

