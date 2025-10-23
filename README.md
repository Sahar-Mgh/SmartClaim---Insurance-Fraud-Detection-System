# 🔍 SmartClaim: Insurance Fraud Detection

> An educational, end-to-end ML project for detecting fraudulent insurance claims.  
> **Goal**: Build an explainable simple classifier.

---
<img width="1280" height="905" alt="image" src="https://github.com/user-attachments/assets/4fab52c8-33b6-446f-92cf-2fc483fbc62f" />

## 📋 What Is This?

SmartClaim is a **beginner-friendly fraud detection system** that predicts whether an insurance claim is fraudulent based on claim characteristics. It's designed as a learning project to understand:

- Binary classification on imbalanced tabular data
- Model comparison (baseline vs improved)
- Handling class imbalance
- Model explainability with SHAP
- Building a minimal ML demo app

**This is educational code** — not production-ready, but production-aware.

---

## 🚀 Quickstart (5 Steps)

### Prerequisites
- Python 3.10 or higher
- 10 minutes of your time

### 1. Clone & Setup
```bash
cd smartclaim
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python -m src.data.generate_synthetic --n_rows 5000 --fraud_rate 0.12
```
Creates `data/processed/synthetic_claims.csv` with ~5000 claims (~12% fraudulent).

### 3. Train Models
```bash
python -m src.models.train --data_path data/processed/synthetic_claims.csv
```
Trains two models:
- **Baseline**: Logistic Regression
- **Improved**: XGBoost

Models saved to `artifacts/models/`.

### 4. Evaluate & Visualize
```bash
python -m src.models.evaluate --data_path data/processed/synthetic_claims.csv
```
Generates plots:
- ROC curves
- Precision-Recall curves
- Confusion matrices

Saved to `artifacts/reports/plots/`.

### 5. Run the App
```bash
streamlit run src/app/streamlit_app.py
```
Opens an interactive web app where you can:
- Enter claim details
- See fraud predictions
- View explanations

---

## 📂 Project Structure

```
smartclaim/
├── README.md                   # You are here
├── HOW_I_DID_IT.md            # Detailed build log (for learning)
├── requirements.txt            # Python dependencies
├── .gitignore                 
│
├── src/                        # Source code
│   ├── data/
│   │   ├── generate_synthetic.py   # Creates synthetic claims data
│   │   └── load_data.py            # Loads and validates data
│   ├── eda/
│   │   └── eda_report.ipynb        # Exploratory data analysis notebook
│   ├── models/
│   │   ├── utils.py                # Preprocessing, metrics, helpers
│   │   ├── train.py                # Train baseline + XGBoost
│   │   ├── evaluate.py             # Generate evaluation plots
│   │   └── explain.py              # SHAP explanations
│   └── app/
│       └── streamlit_app.py        # Interactive demo app
│
├── data/                       # Data storage
│   ├── raw/                    # (Placeholder for external data)
│   └── processed/              # Generated synthetic data
│
└── artifacts/                  # Model outputs
    ├── models/                 # Saved .pkl models
    ├── encoders/               # Fitted preprocessors
    └── reports/
        ├── metrics.json        # Performance metrics
        ├── classification_report.txt
        └── plots/              # ROC, PR, confusion matrix, SHAP plots
```

---

## 🧠 The Models

### Baseline: Logistic Regression
- **Why**: Simple, interpretable, fast
- **How**: Uses `class_weight='balanced'` to handle imbalanced classes
- **When to use**: When you need a fast, explainable baseline

### Improved: XGBoost
- **Why**: Captures non-linear patterns and feature interactions
- **How**: Uses `scale_pos_weight` for class imbalance + light hyperparameter tuning
- **When to use**: When you need better performance and can afford slightly more complexity

Both models use a preprocessing pipeline:
- **Numeric features** → StandardScaler
- **Categorical features** → OneHotEncoder
- **Binary features** → Pass through

---

## 📊 Features in the Dataset

| Feature | Type | Description | Fraud Signal |
|---------|------|-------------|--------------|
| `age` | Numeric | Claimant's age (18-85) | Fraudsters tend to be younger |
| `vehicle_age` | Numeric | Vehicle age in years | Older vehicles = higher fraud risk |
| `claim_amount` | Numeric | Claim amount ($) | Higher amounts = more suspicious |
| `accident_type` | Categorical | collision, theft, vandalism, weather | Theft/vandalism have higher fraud rates |
| `num_prior_claims` | Numeric | Number of previous claims | More claims = higher risk |
| `region` | Categorical | north, south, east, west | Minor variations by region |
| `policy_tenure_months` | Numeric | How long policy has been active | Shorter tenure = higher risk |
| `reported_delay_days` | Numeric | Days between incident and report | Longer delay = more suspicious |
| `has_police_report` | Binary | Police report filed? (0/1) | **Strongest signal**: presence = legitimate |
| `is_fraud` | Binary | **Target**: 1 = fraud, 0 = legitimate | What we're predicting |

---

## ⚖️ Handling Class Imbalance

**Problem**: Only ~12% of claims are fraudulent → models can get high accuracy by always predicting "legitimate".

**Solutions Used**:
1. **Class weights**: Automatically penalize misclassifying the minority class
   - Logistic Regression: `class_weight='balanced'`
   - XGBoost: `scale_pos_weight` (ratio of negative to positive samples)

2. **Better metrics**: Don't rely on accuracy alone
   - **F1 score**: Harmonic mean of precision and recall
   - **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)
   - **ROC-AUC**: Area under ROC curve
   - **Confusion matrix**: See where errors occur

**Why PR-AUC matters**: With imbalanced data, PR-AUC focuses on performance on the minority class (fraud), which is what we care about most.

---

## 🔬 Explainability with SHAP

SHAP (SHapley Additive exPlanations) helps us understand:

### Global Explanation
**Question**: Which features are most important overall?

**How to generate**:
```bash
python -m src.models.explain --data_path data/processed/synthetic_claims.csv
```

**Output**: Bar plot showing mean absolute SHAP values → higher = more important.

### Local Explanation
**Question**: Why did the model predict fraud for *this specific claim*?

**How to generate**:
```bash
python -m src.models.explain --sample_idx 10
```

**Output**: Waterfall/force plot showing which features pushed the prediction toward or away from fraud.

### In the App
The Streamlit app shows top 5 contributing features for each prediction with simple +/- indicators.

---

## 🎯 Performance Expectations

On synthetic data with default settings:

| Model | F1 Score | PR-AUC | ROC-AUC |
|-------|----------|--------|---------|
| Baseline (Logistic) | ~0.55-0.65 | ~0.60-0.70 | ~0.80-0.85 |
| XGBoost | ~0.65-0.75 | ~0.70-0.80 | ~0.85-0.90 |

**Note**: Exact numbers vary with random seed and synthetic data generation.

---

## 🔄 Using Your Own Data

Want to use real insurance claims data instead of synthetic?

### Option 1: Drop-in Replacement
1. Place your CSV in `data/raw/your_claims.csv`
2. Ensure it has these columns:
   ```
   age, vehicle_age, claim_amount, accident_type, num_prior_claims,
   region, policy_tenure_months, reported_delay_days, has_police_report, is_fraud
   ```
3. Run training with your data:
   ```bash
   python -m src.models.train --data_path data/raw/your_claims.csv
   ```

### Option 2: Modify the Loader
Edit `src/data/load_data.py` to add custom preprocessing for your data format.

---

## 📈 Next Steps & Extensions

Want to improve this project? Try:

### Model Improvements
- [ ] **Threshold tuning**: Adjust classification threshold (default 0.5) based on cost of false positives vs false negatives
- [ ] **More features**: Add engineered features (e.g., age × vehicle_age, claim_amount / policy_tenure)
- [ ] **Ensemble**: Combine multiple models
- [ ] **Calibration**: Use Platt scaling or isotonic regression to calibrate probabilities

### Production Readiness
- [ ] **API**: Wrap model in FastAPI for REST endpoint
- [ ] **Monitoring**: Track prediction distributions and model drift
- [ ] **Validation**: Add more comprehensive input validation
- [ ] **Testing**: Add unit tests and integration tests
- [ ] **CI/CD**: Automate testing and deployment

### Advanced Topics
- [ ] **Fairness**: Check for bias across demographic groups
- [ ] **Uncertainty**: Add prediction confidence intervals
- [ ] **Active learning**: Flag uncertain predictions for human review
- [ ] **Time-based splits**: Train on older data, test on newer (for time-series leakage)

---

## 📚 Learning Resources

### Understanding the Concepts
- **Class imbalance**: [Imbalanced Learn Documentation](https://imbalanced-learn.org/)
- **SHAP**: [SHAP Paper](https://arxiv.org/abs/1705.07874) | [SHAP Tutorials](https://shap.readthedocs.io/)
- **XGBoost**: [Official Documentation](https://xgboost.readthedocs.io/)

### Why These Choices?
See `HOW_I_DID_IT.md` for a detailed narrative of design decisions.

---

## 🐛 Troubleshooting

### "Model not found" error in app
**Solution**: Run training first: `python -m src.models.train`

### "Data file not found"
**Solution**: Generate data first: `python -m src.data.generate_synthetic`

### SHAP is slow / runs out of memory
**Solution**: Reduce `--max_samples` in `src/models/explain.py` (default: 500)

### Import errors
**Solution**: Make sure you're running commands from the `smartclaim/` root directory

---

## 📄 License & Usage

This is an **educational project**. Feel free to:
- ✅ Use for learning
- ✅ Adapt for your own projects
- ✅ Share with others

Please:
- ❌ Don't use for actual fraud detection without proper validation
- ❌ Don't claim this is production-ready without significant hardening

---

## 🙏 Credits

Built as an educational ML project demonstrating:
- Scikit-learn pipelines
- XGBoost classification
- SHAP explanations
- Streamlit apps

**Synthetic data only** — no real insurance claims or customer information used.

---

## 📬 Questions?

Check `HOW_I_DID_IT.md` for a detailed walkthrough of how this was built, including design decisions and learning notes.

**Happy learning!** 🚀

