"""Test the trained model with predictions"""

import joblib
import pandas as pd
import numpy as np

print("=" * 80)
print("Testing Loan Eligibility Predictions")
print("=" * 80)
print()

# Load model and feature names
print("Loading model...")
model = joblib.load("models_trained/best_model.joblib")
feature_names = joblib.load("models_trained/feature_names.joblib")
print(f"  ✓ Model loaded: {type(model).__name__}")
print(f"  ✓ Features: {len(feature_names)} features")
print()

# Load some real data for testing
print("Loading test data...")
transactions = pd.read_csv("data/raw/mobile_money_transactions.csv")
target = pd.read_csv("data/raw/loan_eligibility.csv")

# Pick a few users to test
test_users = target.sample(5, random_state=42)['user_id'].tolist()
print(f"  ✓ Testing {len(test_users)} users")
print()

print("Making Predictions:")
print("=" * 80)

for user_id in test_users:
    # Get user's actual eligibility
    actual = target[target['user_id'] == user_id]['is_eligible'].values[0]

    # Get user transactions
    user_trans = transactions[transactions['user_id'] == user_id]

    # Simple feature calculation (matching training)
    features = {
        'amount_count': len(user_trans),
        'amount_sum': user_trans['amount'].sum(),
        'amount_mean': user_trans['amount'].mean(),
        'amount_std': user_trans['amount'].std() or 0,
        'amount_min': user_trans['amount'].min(),
        'amount_max': user_trans['amount'].max(),
        'balance_mean': user_trans['balance'].mean(),
        'balance_min': user_trans['balance'].min(),
        'balance_max': user_trans['balance'].max(),
        'balance_std': user_trans['balance'].std() or 0,
    }

    # Incoming/outgoing
    incoming = user_trans[user_trans['type'] == 'incoming']
    outgoing = user_trans[user_trans['type'] == 'outgoing']

    features['income_total'] = incoming['amount'].sum()
    features['income_avg'] = incoming['amount'].mean() if len(incoming) > 0 else 0
    features['income_count'] = len(incoming)
    features['income_std'] = incoming['amount'].std() if len(incoming) > 0 else 0

    features['expense_total'] = outgoing['amount'].sum()
    features['expense_avg'] = outgoing['amount'].mean() if len(outgoing) > 0 else 0
    features['expense_count'] = len(outgoing)
    features['expense_std'] = outgoing['amount'].std() if len(outgoing) > 0 else 0

    features['category_diversity'] = user_trans['category'].nunique()
    features['merchant_diversity'] = user_trans['merchant_id'].nunique()

    # Airtime (simplified - set to 0 for now)
    features.update({
        'airtime_total': 0,
        'airtime_avg': 0,
        'airtime_count': 0,
        'airtime_std': 0,
    })

    # Loan features (set to 0)
    features.update({
        'prev_loans_count': 0,
        'prev_loans_avg': 0,
        'prev_loans_total': 0,
        'default_count': 0,
        'default_rate': 0,
    })

    # Derived features
    features['income_expense_ratio'] = features['income_total'] / (features['expense_total'] + 1)
    features['balance_range'] = features['balance_max'] - features['balance_min']
    features['avg_balance_income_ratio'] = features['balance_mean'] / (features['income_avg'] + 1)

    # Create DataFrame with correct feature order
    X = pd.DataFrame([features])[feature_names]

    # Make prediction
    probability = model.predict_proba(X)[0, 1]
    prediction = "APPROVE" if probability >= 0.5 else "REJECT"
    actual_label = "ELIGIBLE" if actual == 1 else "NOT ELIGIBLE"

    correct = "✓" if (prediction == "APPROVE" and actual == 1) or (prediction == "REJECT" and actual == 0) else "✗"

    print(f"\nUser: {user_id}")
    print(f"  Prediction: {prediction} ({probability:.1%} confidence)")
    print(f"  Actual: {actual_label}")
    print(f"  Result: {correct} {'CORRECT' if correct == '✓' else 'INCORRECT'}")
    print(f"  Transactions: {features['amount_count']}")
    print(f"  Avg Income: ${features['income_avg']:.2f}")
    print(f"  Avg Expense: ${features['expense_avg']:.2f}")
    print(f"  Balance: ${features['balance_mean']:.2f}")

print()
print("=" * 80)
print("Prediction Test Complete!")
print("=" * 80)
