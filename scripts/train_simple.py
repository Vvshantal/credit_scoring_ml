"""Simplified training script - works without XGBoost/LightGBM"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

print("=" * 80)
print("ML Loan Eligibility Platform - Simplified Training")
print("=" * 80)
print()

# Load data
print("Step 1: Loading data...")
transactions = pd.read_csv("data/raw/mobile_money_transactions.csv")
airtime = pd.read_csv("data/raw/airtime_purchases.csv")
loans = pd.read_csv("data/raw/loan_history.csv")
target = pd.read_csv("data/raw/loan_eligibility.csv")
print(f"  ✓ Data loaded: {len(transactions)} transactions, {len(target)} users")
print()

# Feature engineering
print("Step 2: Engineering features...")

# Transaction features
trans_features = transactions.groupby('user_id').agg({
    'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
    'balance': ['mean', 'min', 'max', 'std']
}).reset_index()
trans_features.columns = ['user_id'] + [f'{col[0]}_{col[1]}' for col in trans_features.columns[1:]]

# Incoming/outgoing
incoming = transactions[transactions['type'] == 'incoming'].groupby('user_id')['amount'].agg([
    ('income_total', 'sum'),
    ('income_avg', 'mean'),
    ('income_count', 'count'),
    ('income_std', 'std')
]).reset_index()

outgoing = transactions[transactions['type'] == 'outgoing'].groupby('user_id')['amount'].agg([
    ('expense_total', 'sum'),
    ('expense_avg', 'mean'),
    ('expense_count', 'count'),
    ('expense_std', 'std')
]).reset_index()

# Category diversity
category_diversity = transactions.groupby('user_id')['category'].nunique().reset_index()
category_diversity.columns = ['user_id', 'category_diversity']

# Merchant diversity
merchant_diversity = transactions.groupby('user_id')['merchant_id'].nunique().reset_index()
merchant_diversity.columns = ['user_id', 'merchant_diversity']

# Airtime features
airtime_features = airtime.groupby('user_id')['amount'].agg([
    ('airtime_total', 'sum'),
    ('airtime_avg', 'mean'),
    ('airtime_count', 'count'),
    ('airtime_std', 'std')
]).reset_index()

# Loan features
loan_features = loans.groupby('user_id').agg({
    'loan_amount': [('prev_loans_count', 'count'), ('prev_loans_avg', 'mean'), ('prev_loans_total', 'sum')],
    'is_default': [('default_count', 'sum'), ('default_rate', 'mean')]
}).reset_index()
loan_features.columns = ['user_id', 'prev_loans_count', 'prev_loans_avg', 'prev_loans_total', 'default_count', 'default_rate']

# Merge all features
features = trans_features.merge(incoming, on='user_id', how='left')
features = features.merge(outgoing, on='user_id', how='left')
features = features.merge(category_diversity, on='user_id', how='left')
features = features.merge(merchant_diversity, on='user_id', how='left')
features = features.merge(airtime_features, on='user_id', how='left')
features = features.merge(loan_features, on='user_id', how='left')

# Fill missing values
features = features.fillna(0)

# Create some derived features
features['income_expense_ratio'] = features['income_total'] / (features['expense_total'] + 1)
features['balance_range'] = features['balance_max'] - features['balance_min']
features['avg_balance_income_ratio'] = features['balance_mean'] / (features['income_avg'] + 1)

print(f"  ✓ Created {len(features.columns) - 1} features")
print()

# Prepare data
print("Step 3: Preparing data...")
data = features.merge(target, on='user_id')

# Drop non-numeric columns
columns_to_drop = ['user_id', 'is_eligible']
# Also drop any other non-numeric columns that might have been merged
for col in data.columns:
    if data[col].dtype == 'object':
        columns_to_drop.append(col)

X = data.drop(columns=columns_to_drop)
y = data['is_eligible']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  ✓ Class distribution: {y_train.value_counts().to_dict()}")
print()

# Apply SMOTE
print("Step 4: Applying SMOTE for class balance...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"  ✓ After SMOTE: {len(X_train_smote)} samples")
print(f"  ✓ Class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")
print()

# Train models
results = {}

print("Step 5: Training models...")
print()

# Logistic Regression
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_smote, y_train_smote)
lr_pred = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'recall': recall_score(y_test, lr_pred),
    'f1': f1_score(y_test, lr_pred),
    'roc_auc': roc_auc_score(y_test, lr_proba)
}
print(f"    ✓ Accuracy: {results['Logistic Regression']['accuracy']:.2%}, ROC-AUC: {results['Logistic Regression']['roc_auc']:.4f}")

# Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'f1': f1_score(y_test, rf_pred),
    'roc_auc': roc_auc_score(y_test, rf_proba)
}
print(f"    ✓ Accuracy: {results['Random Forest']['accuracy']:.2%}, ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")
print()

# Model comparison
print("=" * 80)
print("Model Comparison")
print("=" * 80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
print("-" * 80)

for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
          f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f}")

print("=" * 80)
print()

# Select best model
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
best_model = rf if best_model_name == 'Random Forest' else lr

print(f"Best Model: {best_model_name}")
print()

# Feature importance (Random Forest)
if best_model_name == 'Random Forest':
    print("Top 15 Most Important Features:")
    print("-" * 60)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in importances.head(15).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")
    print()

# Save model
print("Step 6: Saving model...")
Path("models_trained").mkdir(exist_ok=True)

joblib.dump(best_model, "models_trained/best_model.joblib")
joblib.dump(list(X.columns), "models_trained/feature_names.joblib")

print(f"  ✓ Model saved: models_trained/best_model.joblib")
print(f"  ✓ Features saved: models_trained/feature_names.joblib")
print()

# Test predictions
print("Sample Predictions:")
print("-" * 60)
sample_proba = best_model.predict_proba(X_test.head(10))[:, 1]
sample_pred = (sample_proba >= 0.5).astype(int)

for i in range(10):
    decision = "APPROVE" if sample_pred[i] else "REJECT"
    print(f"  Application {i+1}: {decision:8s} (Probability: {sample_proba[i]:.2%})")
print()

print("=" * 80)
print("Training Complete!")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {results[best_model_name]['accuracy']:.2%}")
print(f"Test ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"\nModel saved to: models_trained/best_model.joblib")
