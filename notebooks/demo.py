"""Quick Demo of the ML Loan Eligibility Platform

This script demonstrates:
1. Loading data
2. Feature engineering
3. Training a simple model
4. Making predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

print("=" * 80)
print("ML Loan Eligibility Platform - Quick Demo")
print("=" * 80)
print()

# 1. Load Data
print("Step 1: Loading data...")
transactions = pd.read_csv("data/raw/mobile_money_transactions.csv")
airtime = pd.read_csv("data/raw/airtime_purchases.csv")
loans = pd.read_csv("data/raw/loan_history.csv")
target = pd.read_csv("data/raw/loan_eligibility.csv")

print(f"  ✓ Loaded {len(transactions)} transactions")
print(f"  ✓ Loaded {len(airtime)} airtime purchases")
print(f"  ✓ Loaded {len(loans)} loan records")
print(f"  ✓ Loaded {len(target)} users with labels")
print()

# 2. Simple Feature Engineering
print("Step 2: Engineering features...")

# Transaction features per user
trans_features = transactions.groupby('user_id').agg({
    'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
}).reset_index()
trans_features.columns = ['user_id', 'txn_count', 'txn_total', 'txn_mean',
                         'txn_std', 'txn_min', 'txn_max']

# Incoming vs outgoing
incoming = transactions[transactions['type'] == 'incoming'].groupby('user_id')['amount'].agg([
    ('income_total', 'sum'),
    ('income_avg', 'mean'),
    ('income_count', 'count')
]).reset_index()

outgoing = transactions[transactions['type'] == 'outgoing'].groupby('user_id')['amount'].agg([
    ('expense_total', 'sum'),
    ('expense_avg', 'mean'),
    ('expense_count', 'count')
]).reset_index()

# Balance features
balance_features = transactions.groupby('user_id')['balance'].agg([
    ('balance_avg', 'mean'),
    ('balance_min', 'min'),
    ('balance_max', 'max'),
    ('balance_std', 'std')
]).reset_index()

# Airtime features
airtime_features = airtime.groupby('user_id')['amount'].agg([
    ('airtime_total', 'sum'),
    ('airtime_avg', 'mean'),
    ('airtime_count', 'count')
]).reset_index()

# Loan features
loan_features = loans.groupby('user_id').agg({
    'loan_amount': [('prev_loans_count', 'count'), ('prev_loans_avg', 'mean')],
    'is_default': [('default_count', 'sum')]
}).reset_index()
loan_features.columns = ['user_id', 'prev_loans_count', 'prev_loans_avg', 'default_count']

# Merge all features
features = trans_features.merge(incoming, on='user_id', how='left')
features = features.merge(outgoing, on='user_id', how='left')
features = features.merge(balance_features, on='user_id', how='left')
features = features.merge(airtime_features, on='user_id', how='left')
features = features.merge(loan_features, on='user_id', how='left')

# Fill missing values
features = features.fillna(0)

print(f"  ✓ Created {len(features.columns) - 1} features")
print()

# 3. Prepare data for training
print("Step 3: Preparing data for training...")
data = features.merge(target, on='user_id')
X = data.drop(columns=['user_id', 'is_eligible'])
y = data['is_eligible']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✓ Training set: {len(X_train)} samples")
print(f"  ✓ Test set: {len(X_test)} samples")
print(f"  ✓ Positive rate in training: {y_train.mean():.1%}")
print()

# 4. Train model
print("Step 4: Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("  ✓ Model trained")
print()

# 5. Evaluate
print("Step 5: Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"  ✓ Test Accuracy: {accuracy:.2%}")
print(f"  ✓ Test ROC-AUC: {auc:.4f}")
print()

print("Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['Not Eligible', 'Eligible']))
print()

# 6. Feature importance
print("Top 10 Most Important Features:")
print("-" * 60)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importances.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")
print()

# 7. Make sample predictions
print("Sample Predictions:")
print("-" * 60)
sample_users = X_test.head(5)
sample_preds = model.predict_proba(sample_users)[:, 1]
sample_decisions = (sample_preds >= 0.5).astype(int)

for i in range(len(sample_users)):
    decision = "APPROVE" if sample_decisions[i] else "REJECT"
    prob = sample_preds[i]
    print(f"  Application {i+1}: {decision:8s} (Probability: {prob:.2%})")
print()

print("=" * 80)
print("Demo Complete!")
print("=" * 80)
print()
print("Next steps:")
print("  1. Run full training: python scripts/train_model.py")
print("  2. Start API server: uvicorn src.api.app:app --reload")
print("  3. Visit API docs: http://localhost:8000/docs")
print()
