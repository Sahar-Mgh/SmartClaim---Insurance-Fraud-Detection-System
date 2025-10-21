"""
SmartClaim: Interactive Fraud Detection Demo
A simple Streamlit app to predict fraud and explain predictions.
"""

import sys
from pathlib import Path
import joblib
import warnings

import numpy as np
import pandas as pd
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import load_claims_data


# Configure page
st.set_page_config(
    page_title="SmartClaim Fraud Detector",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_model(model_path: str):
    """Load the trained model (cached for performance)."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_sample_data(data_path: str):
    """Load sample data for testing (cached)."""
    try:
        return load_claims_data(data_path)
    except Exception as e:
        st.warning(f"Could not load sample data: {e}")
        return None


def get_shap_explanation(model, input_df: pd.DataFrame, top_n: int = 5):
    """
    Get feature contributions for explanation.
    
    For simplicity, we use the model's feature importances (XGBoost)
    or coefficients (Logistic Regression) as a proxy for SHAP values.
    In a production app, you'd compute actual SHAP values.
    
    Args:
        model: Trained model
        input_df: Input features
        top_n: Number of top features to show
        
    Returns:
        DataFrame with feature names and contributions
    """
    try:
        # Transform input
        X_transformed = model.named_steps["preprocessor"].transform(input_df)
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        
        classifier = model.named_steps["classifier"]
        
        # Get feature importances/coefficients
        if hasattr(classifier, "feature_importances_"):
            # XGBoost: use feature importances weighted by input values
            importances = classifier.feature_importances_
            contributions = X_transformed[0] * importances
        elif hasattr(classifier, "coef_"):
            # Logistic Regression: use coefficients weighted by input values
            contributions = X_transformed[0] * classifier.coef_[0]
        else:
            return None
        
        # Create explanation dataframe
        explanation_df = pd.DataFrame({
            "feature": feature_names,
            "contribution": contributions
        }).sort_values("contribution", key=abs, ascending=False).head(top_n)
        
        return explanation_df
        
    except Exception as e:
        st.warning(f"Could not generate explanation: {e}")
        return None


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("🔍 SmartClaim: Insurance Fraud Detector")
    st.markdown("""
    This tool predicts whether an insurance claim is likely to be **fraudulent** based on claim characteristics.
    
    **How it works:** Enter claim details below, and the model will predict the fraud probability.
    """)
    
    # Load model
    model_path = "artifacts/models/xgboost_model.pkl"
    model = load_model(model_path)
    
    if model is None:
        st.error("Model not found! Please train the model first:")
        st.code("python -m src.data.generate_synthetic")
        st.code("python -m src.models.train")
        st.stop()
    
    # Load sample data for random selection
    sample_data = load_sample_data("data/processed/synthetic_claims.csv")
    
    # Sidebar: Input features
    st.sidebar.header("📋 Claim Details")
    
    # Random sample button
    if sample_data is not None and st.sidebar.button("🎲 Try Random Sample"):
        random_row = sample_data.sample(1, random_state=np.random.randint(0, 10000)).iloc[0]
        st.session_state["random_sample"] = random_row
    
    # Get default values (from random sample if exists)
    defaults = st.session_state.get("random_sample", {})
    
    # Numeric inputs
    st.sidebar.subheader("Claimant & Vehicle")
    age = st.sidebar.slider(
        "Age",
        min_value=18,
        max_value=85,
        value=int(defaults.get("age", 40)),
        help="Claimant's age in years"
    )
    
    vehicle_age = st.sidebar.slider(
        "Vehicle Age",
        min_value=0.0,
        max_value=25.0,
        value=float(defaults.get("vehicle_age", 5.0)),
        step=0.5,
        help="Age of the vehicle in years"
    )
    
    st.sidebar.subheader("Claim Details")
    claim_amount = st.sidebar.number_input(
        "Claim Amount ($)",
        min_value=500,
        max_value=100000,
        value=int(defaults.get("claim_amount", 5000)),
        step=500,
        help="Total claim amount in dollars"
    )
    
    accident_type = st.sidebar.selectbox(
        "Accident Type",
        options=["collision", "theft", "vandalism", "weather"],
        index=["collision", "theft", "vandalism", "weather"].index(
            defaults.get("accident_type", "collision")
        ),
        help="Type of accident/incident"
    )
    
    region = st.sidebar.selectbox(
        "Region",
        options=["north", "south", "east", "west"],
        index=["north", "south", "east", "west"].index(
            defaults.get("region", "north")
        ),
        help="Geographic region"
    )
    
    st.sidebar.subheader("Claim History")
    num_prior_claims = st.sidebar.slider(
        "Number of Prior Claims",
        min_value=0,
        max_value=10,
        value=int(defaults.get("num_prior_claims", 0)),
        help="Number of previous claims filed"
    )
    
    policy_tenure_months = st.sidebar.slider(
        "Policy Tenure (months)",
        min_value=1,
        max_value=240,
        value=int(defaults.get("policy_tenure_months", 24)),
        help="How long the policy has been active"
    )
    
    st.sidebar.subheader("Reporting Details")
    reported_delay_days = st.sidebar.slider(
        "Reported Delay (days)",
        min_value=0.0,
        max_value=30.0,
        value=float(defaults.get("reported_delay_days", 2.0)),
        step=0.5,
        help="Days between incident and report"
    )
    
    has_police_report = st.sidebar.radio(
        "Police Report Filed?",
        options=[1, 0],
        index=0 if defaults.get("has_police_report", 1) == 1 else 1,
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Was a police report filed?"
    )
    
    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "vehicle_age": [vehicle_age],
        "claim_amount": [claim_amount],
        "accident_type": [accident_type],
        "num_prior_claims": [num_prior_claims],
        "region": [region],
        "policy_tenure_months": [policy_tenure_months],
        "reported_delay_days": [reported_delay_days],
        "has_police_report": [has_police_report]
    })
    
    # Main area: Predictions and explanations
    st.header("🎯 Prediction Results")
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        fraud_probability = prediction_proba[1]
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Predicted Class",
                value="🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Fraud Probability",
                value=f"{fraud_probability:.1%}",
                delta=None
            )
        
        with col3:
            risk_level = "High" if fraud_probability > 0.7 else "Medium" if fraud_probability > 0.3 else "Low"
            st.metric(
                label="Risk Level",
                value=risk_level,
                delta=None
            )
        
        # Risk interpretation
        st.markdown("---")
        st.subheader("📊 Risk Assessment")
        
        if fraud_probability > 0.7:
            st.error("""
            **High Risk**: This claim shows strong indicators of potential fraud. 
            Recommend thorough investigation and additional documentation.
            """)
        elif fraud_probability > 0.3:
            st.warning("""
            **Medium Risk**: This claim has some suspicious characteristics. 
            Consider reviewing claim details and requesting additional information.
            """)
        else:
            st.success("""
            **Low Risk**: This claim appears legitimate based on the provided information. 
            Standard processing recommended.
            """)
        
        # Feature explanation
        st.markdown("---")
        st.subheader("💡 Key Contributing Factors")
        st.markdown("These factors had the most influence on the prediction:")
        
        explanation = get_shap_explanation(model, input_data, top_n=5)
        
        if explanation is not None:
            for idx, row in explanation.iterrows():
                feature_name = row["feature"]
                contribution = row["contribution"]
                
                # Simplify feature names for display
                display_name = feature_name.replace("_", " ").title()
                
                # Determine direction
                if contribution > 0:
                    direction = "⬆️ Increases"
                    color = "🔴"
                else:
                    direction = "⬇️ Decreases"
                    color = "🟢"
                
                st.markdown(f"**{color} {display_name}** — {direction} fraud risk")
        else:
            st.info("Feature importance explanation not available for this model type.")
        
        # Additional info
        st.markdown("---")
        st.caption("""
        ℹ️ **About this model**: Trained on synthetic insurance claims data using XGBoost. 
        This is an educational demo and should not be used for actual fraud detection without 
        proper validation and human oversight.
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("Make sure the model is properly trained.")


if __name__ == "__main__":
    main()

