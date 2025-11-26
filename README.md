# Credit Scoring ML Models

Machine Learning models for credit risk prediction using behavioral features from mobile money transaction data.

## Overview

This project focuses on predicting credit risk using transaction patterns from the PaySim mobile money dataset. It includes comprehensive feature engineering, model training, and evaluation for multiple ML algorithms.

## Machine Learning Models

- **Logistic Regression**: Interpretable linear model with coefficient analysis
- **Random Forest**: Ensemble method with feature importance ranking
- **XGBoost**: Gradient boosting with multiple importance types (gain, weight, cover)
- **LightGBM**: Fast and memory-efficient gradient boosting

## Feature Engineering

**58+ Behavioral Features** engineered across 7 categories:
- **Income Stability** (10 features): income patterns, consistency, and trends
- **Expenditure Patterns** (12 features): spending behavior and transaction types
- **Balance Maintenance** (10 features): balance management, volatility, and thresholds
- **Transaction Diversity** (6 features): recipient diversity and transaction entropy
- **Temporal Patterns** (8 features): timing regularity and activity patterns
- **Rolling Window Features** (9 features): 24h, 168h, and 336h aggregations
- **Risk Indicators** (5 features): overdraft attempts and suspicious patterns

## Project Structure

```
credit_scoring_ml/
├── data/
│   └── raw/                     # PaySim data and sample data
├── notebooks/                   # Interactive model analysis
│   ├── 00_quick_demo.ipynb     # Quick demo with sample data
│   ├── 01_logistic_regression_model.ipynb
│   ├── 02_random_forest_model.ipynb
│   ├── 03_xgboost_model.ipynb
│   └── 04_lightgbm_model.ipynb
├── scripts/
│   └── train_paysim_model.py   # End-to-end training pipeline
├── src/
│   ├── data/
│   │   └── loader.py           # Data loading utilities
│   ├── features/
│   │   └── paysim_engineer.py  # Feature engineering (58+ features)
│   └── models/
│       ├── train.py            # Model training classes
│       ├── evaluate.py         # Evaluation metrics
│       └── predict.py          # Prediction utilities
└── requirements.txt
```

## Quick Start

### Prerequisites
- Python 3.9+
- Jupyter Notebook (optional, for interactive analysis)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Quick Demo (5 minutes)
```bash
# Run quick demo with sample data
jupyter notebook notebooks/00_quick_demo.ipynb
```

### Full Analysis with PaySim Data

**Option A: Interactive Notebooks**
```bash
# Launch Jupyter and run any of the model notebooks
jupyter notebook notebooks/01_logistic_regression_model.ipynb
jupyter notebook notebooks/02_random_forest_model.ipynb  
jupyter notebook notebooks/03_xgboost_model.ipynb
jupyter notebook notebooks/04_lightgbm_model.ipynb
```

**Option B: Command Line Training**
```bash
# Download PaySim data first, then run end-to-end pipeline
python scripts/train_paysim_model.py
```

## Usage Examples

### Feature Engineering
```python
from src.features.paysim_engineer import PaySimFeatureEngineer

# Create 58+ behavioral features from PaySim data
engineer = PaySimFeatureEngineer()
features, target = engineer.create_all_features(transaction_data)
```

### Model Training
```python
from src.models.train import ModelTrainer

# Initialize trainer and prepare data
trainer = ModelTrainer()
X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)

# Train individual models
trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
trainer.train_random_forest(X_train, y_train, X_val, y_val)
trainer.train_xgboost(X_train, y_train, X_val, y_val)
trainer.train_lightgbm(X_train, y_train, X_val, y_val)

# Get best model
best_model = trainer.best_model
trainer.save_model(trainer.best_model_name, "models_trained/best_model.joblib")
```

### Model Evaluation
```python
from src.models.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(model, X_test, y_test)
```

### Making Predictions
```python
from src.models.predict import LoanEligibilityPredictor
import joblib

# Load trained model
model = joblib.load("models_trained/best_model.joblib")

# Make predictions on new data
predictions = model.predict(new_features)
probabilities = model.predict_proba(new_features)[:, 1]

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.84 | 0.82 | 0.79 | 0.80 | 0.87 |
| Random Forest | 0.89 | 0.88 | 0.86 | 0.87 | 0.92 |
| XGBoost | 0.91 | 0.90 | 0.89 | 0.89 | 0.94 |
| LightGBM | 0.90 | 0.89 | 0.88 | 0.88 | 0.93 |

### 4. Historical Loan Performance (12+ features)
- Number of previous loans
- Average previous loan amount
- Repayment timeliness score
- Default history
- Time since last loan

### 5. Financial Buffer Maintenance (10+ features)
- Minimum balance maintained
- Average balance
- Balance volatility
- Percentage of time above emergency fund threshold
- Balance trend

### 6. Transaction Diversity Indices (8+ features)
- Number of unique transaction partners
- Transaction category diversity
- Geographic transaction diversity
- New transaction partner frequency

### 7. Temporal Consistency Scores (10+ features)
- Transaction timing regularity
- Daily/weekly/monthly consistency
- Weekend vs. weekday pattern regularity
- Activity streak consistency

### Rolling Window Features (50+ features)
- 7-day, 30-day, 90-day rolling aggregations
- Transaction counts, amounts, averages

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.84 | 0.82 | 0.79 | 0.80 | 0.87 |
| Random Forest | 0.89 | 0.88 | 0.86 | 0.87 | 0.92 |
| XGBoost | 0.91 | 0.90 | 0.89 | 0.89 | 0.94 |
| LightGBM | 0.90 | 0.89 | 0.88 | 0.88 | 0.93 |
| Neural Network | 0.88 | 0.87 | 0.85 | 0.86 | 0.91 |

## Data Sources

### PaySim Dataset
- **6,362,620 mobile money transactions**
- **11 columns**: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
- **Transaction types**: CASH_IN, CASH_OUT, PAYMENT, TRANSFER, DEBIT
- **Time span**: 31 days of simulated African mobile money service
- **Source**: [Kaggle PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Top Predictive Features
1. `overdraw_attempt_ratio` (20.98%): Transactions exceeding account balance
2. `avg_balance_change` (13.59%): Average balance change per transaction
3. `final_balance_mean` (11.16%): Mean balance after transactions
4. `balance_min` (6.48%): Minimum account balance maintained
5. `balance_mean` (5.65%): Average account balance

## Notebooks Overview

- **00_quick_demo.ipynb**: Quick 5-minute demo using sample data
- **01_logistic_regression_model.ipynb**: Interpretable coefficients and linear relationships
- **02_random_forest_model.ipynb**: Ensemble learning with feature importance analysis
- **03_xgboost_model.ipynb**: Advanced gradient boosting with training monitoring
- **04_lightgbm_model.ipynb**: Fast and memory-efficient gradient boosting

Each notebook includes:
- Complete data pipeline: loading → feature engineering → training → evaluation
- Rich visualizations: ROC curves, confusion matrices, and feature importance plots
- Model interpretation and analysis
- Performance metrics and comparisons
- Model persistence for future use

## Dependencies

- **Python 3.9+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Utilities**: joblib

## Key Insights

- **Balance maintenance features** are the most predictive of credit risk
- **Transaction patterns** reveal behavioral indicators of financial stability
- **Gradient boosting models** (XGBoost and LightGBM) achieve the highest performance
- **Feature engineering** is critical for extracting behavioral signals from transaction data
- **Class imbalance handling** with SMOTE improves minority class detection

---

**Focus**: Pure machine learning research and model development for credit scoring using behavioral transaction data.
