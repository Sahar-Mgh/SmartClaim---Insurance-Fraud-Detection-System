# ⚡ SmartClaim Quickstart Guide

**Get started with SmartClaim in under 10 minutes!**

---

## 🎯 What You'll Build

A complete fraud detection system that:
- Generates realistic synthetic insurance claims data
- Trains baseline (Logistic Regression) and improved (XGBoost) models
- Evaluates with comprehensive metrics and visualizations
- Explains predictions with SHAP
- Provides an interactive web demo

---

## 📋 Prerequisites

- **Python 3.10+** installed
- **10 minutes** of your time
- **Terminal/Command Prompt** access

---

## 🚀 Step-by-Step Setup

### Step 1: Navigate to Project Directory

```bash
cd smartclaim
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected time**: 2-3 minutes

**Packages installed**: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn, streamlit, jupyter

### Step 3: Verify Setup (Optional but Recommended)

```bash
python test_setup.py
```

**Expected output**: ✓ checkmarks for all tests

**If errors**: Check that you're in the `smartclaim/` directory and Python 3.10+ is installed.

---

## 🎮 Running the Pipeline

### Step 4: Generate Synthetic Data

```bash
python -m src.data.generate_synthetic --n_rows 5000 --fraud_rate 0.12
```

**What this does**:
- Creates 5,000 synthetic insurance claims
- ~12% are fraudulent (realistic imbalance)
- Saves to `data/processed/synthetic_claims.csv`

**Expected time**: 5 seconds

**Output**: You'll see a summary of the generated data

### Step 5: Train Models

```bash
python -m src.models.train --data_path data/processed/synthetic_claims.csv
```

**What this does**:
- Trains Logistic Regression (baseline)
- Trains XGBoost (improved model)
- Saves models to `artifacts/models/`
- Prints performance metrics

**Expected time**: 30 seconds - 1 minute

**Output**: 
```
Baseline (Logistic Regression) Performance
=============================================
F1: 0.5900
Precision: 0.5500
Recall: 0.6400
ROC-AUC: 0.8100
...

XGBoost Performance
=============================================
F1: 0.6800
Precision: 0.6400
Recall: 0.7300
ROC-AUC: 0.8600
...
```

### Step 6: Evaluate with Visualizations

```bash
python -m src.models.evaluate --data_path data/processed/synthetic_claims.csv
```

**What this does**:
- Generates ROC curves
- Generates Precision-Recall curves
- Generates confusion matrices
- Saves plots to `artifacts/reports/plots/`

**Expected time**: 10-15 seconds

**Output**: PNG files in `artifacts/reports/plots/`

### Step 7: Generate SHAP Explanations

```bash
python -m src.models.explain --data_path data/processed/synthetic_claims.csv --sample_idx 10
```

**What this does**:
- Calculates SHAP values (feature importance)
- Creates global importance bar plot
- Creates local explanation for sample #10
- Saves plots to `artifacts/reports/plots/`

**Expected time**: 30 seconds - 1 minute

**Output**: SHAP importance plots

### Step 8: Launch the Interactive App

```bash
streamlit run src/app/streamlit_app.py
```

**What this does**:
- Opens a web browser automatically
- Shows interactive fraud prediction interface
- Allows you to try random samples

**Expected time**: 5 seconds to start

**Output**: Browser opens to `http://localhost:8501`

**To stop**: Press `Ctrl+C` in terminal

---

## 🎨 Using the App

Once the Streamlit app is running:

1. **Adjust claim details** using sliders and dropdowns in the left sidebar
2. **Click "Try Random Sample"** to load a real claim from the dataset
3. **View prediction** in the main area:
   - Fraud probability
   - Risk level (Low/Medium/High)
   - Top contributing features
4. **Experiment** with different values to see how predictions change

---

## 📊 Exploring Results

### View Generated Plots

```bash
# On Windows
explorer artifacts\reports\plots

# On Mac
open artifacts/reports/plots

# On Linux
xdg-open artifacts/reports/plots
```

**You'll find**:
- `baseline_roc_curve.png` - Baseline ROC curve
- `xgboost_roc_curve.png` - XGBoost ROC curve
- `baseline_pr_curve.png` - Baseline Precision-Recall curve
- `xgboost_pr_curve.png` - XGBoost Precision-Recall curve
- `baseline_confusion_matrix.png` - Baseline confusion matrix
- `xgboost_confusion_matrix.png` - XGBoost confusion matrix
- `xgboost_shap_importance.png` - Global feature importance
- `xgboost_shap_sample_10.png` - Local explanation example

### Explore the Data in Jupyter

```bash
jupyter notebook src/eda/eda_report.ipynb
```

**What's inside**:
- Class distribution analysis
- Feature distributions (fraud vs legitimate)
- Correlation heatmaps
- Categorical feature analysis
- Police report impact analysis

---

## 🐛 Troubleshooting

### "Model not found" error in app

**Problem**: You haven't trained the model yet

**Solution**: Run Step 5 (train models)

### "Data file not found"

**Problem**: You haven't generated synthetic data yet

**Solution**: Run Step 4 (generate data)

### Import errors

**Problem**: Dependencies not installed or wrong directory

**Solution**: 
1. Make sure you're in `smartclaim/` directory
2. Run `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.10+)

### Streamlit doesn't open browser automatically

**Problem**: Browser didn't auto-open

**Solution**: Manually open `http://localhost:8501` in your browser

### SHAP is very slow

**Problem**: SHAP calculations can be computationally intensive

**Solution**: Edit `src/models/explain.py` and reduce `max_samples` (default: 500) to a lower number like 100

---

## 📚 What to Read Next

### Understanding the Code
- **README.md** - Full project documentation
- **HOW_I_DID_IT.md** - Detailed build log with design decisions

### Learning More
- Check out the EDA notebook for data exploration insights
- Read the code comments (every file is heavily commented)
- Try modifying hyperparameters in `src/models/train.py`

### Customization
- Change fraud rate: `--fraud_rate 0.15` in Step 4
- Change dataset size: `--n_rows 10000` in Step 4
- Modify model parameters in `src/models/train.py`

---

## 🎓 Learning Exercises

### Beginner
1. Change the fraud rate to 5% and see how it affects model performance
2. Try different sample indices in Step 7 to see different explanations
3. Play with the Streamlit app to understand feature effects

### Intermediate
1. Add a new feature to the synthetic data generator
2. Modify XGBoost hyperparameters and compare performance
3. Change the classification threshold from 0.5 to 0.3 or 0.7

### Advanced
1. Implement SMOTE for handling imbalance
2. Add cross-validation to the training script
3. Create a cost-sensitive learning approach
4. Add actual SHAP value computation to the Streamlit app

---

## ✅ Quick Checklist

Use this to verify you've completed all steps:

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Verified setup (`python test_setup.py`)
- [ ] Generated data (`python -m src.data.generate_synthetic`)
- [ ] Trained models (`python -m src.models.train`)
- [ ] Evaluated models (`python -m src.models.evaluate`)
- [ ] Generated SHAP explanations (`python -m src.models.explain`)
- [ ] Ran the app (`streamlit run src/app/streamlit_app.py`)
- [ ] Explored the EDA notebook (`jupyter notebook src/eda/eda_report.ipynb`)

**All checked?** Congratulations! You've completed the SmartClaim project! 🎉

---

## 🎯 Next Steps

Now that you've run through the project:

1. **Read HOW_I_DID_IT.md** to understand the design decisions
2. **Experiment** with different parameters
3. **Modify the code** to add your own features
4. **Apply this template** to your own classification problem
5. **Share your learnings** with others

---

**Questions?** Check README.md or HOW_I_DID_IT.md for detailed explanations.

**Happy learning!** 🚀

