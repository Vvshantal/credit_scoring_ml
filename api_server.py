"""Simplified FastAPI server for loan eligibility predictions"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

# Initialize FastAPI
app = FastAPI(
    title="ML Loan Eligibility Platform",
    description="AI-powered loan eligibility assessment",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
print("Loading model...")
model = joblib.load("models_trained/best_model.joblib")
feature_names = joblib.load("models_trained/feature_names.joblib")
print(f"✓ Model loaded: {type(model).__name__}")
print(f"✓ Features: {len(feature_names)}")

# Load sample data for lookups
transactions = pd.read_csv("data/raw/mobile_money_transactions.csv")
airtime = pd.read_csv("data/raw/airtime_purchases.csv")
loans_data = pd.read_csv("data/raw/loan_history.csv")
eligibility_data = pd.read_csv("data/raw/loan_eligibility.csv")

# Schemas
class ApplicationRequest(BaseModel):
    user_id: str
    requested_amount: float
    loan_purpose: Optional[str] = "business"
    contact_email: str
    contact_phone: str

class ApplicationResponse(BaseModel):
    application_id: str
    status: str
    message: str
    created_at: str

class DecisionResponse(BaseModel):
    application_id: str
    decision: str
    probability: float
    confidence: float
    requested_amount: float
    recommended_amount: float
    explanation: dict
    timestamp: str

def extract_features(user_id: str):
    """Extract features for a user from transaction data"""

    user_trans = transactions[transactions['user_id'] == user_id]

    if len(user_trans) == 0:
        # Return default features for unknown users
        return {name: 0.0 for name in feature_names}

    # Calculate features
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

    # Airtime
    user_airtime = airtime[airtime['user_id'] == user_id]
    features['airtime_total'] = user_airtime['amount'].sum() if len(user_airtime) > 0 else 0
    features['airtime_avg'] = user_airtime['amount'].mean() if len(user_airtime) > 0 else 0
    features['airtime_count'] = len(user_airtime)
    features['airtime_std'] = user_airtime['amount'].std() if len(user_airtime) > 0 else 0

    # Loan history
    user_loans = loans_data[loans_data['user_id'] == user_id]
    features['prev_loans_count'] = len(user_loans)
    features['prev_loans_avg'] = user_loans['loan_amount'].mean() if len(user_loans) > 0 else 0
    features['prev_loans_total'] = user_loans['loan_amount'].sum() if len(user_loans) > 0 else 0
    features['default_count'] = user_loans['is_default'].sum() if len(user_loans) > 0 else 0
    features['default_rate'] = user_loans['is_default'].mean() if len(user_loans) > 0 else 0

    # Derived features
    features['income_expense_ratio'] = features['income_total'] / (features['expense_total'] + 1)
    features['balance_range'] = features['balance_max'] - features['balance_min']
    features['avg_balance_income_ratio'] = features['balance_mean'] / (features['income_avg'] + 1)

    # Get credit_score and monthly_income from eligibility data if available
    user_eligibility = eligibility_data[eligibility_data['user_id'] == user_id]
    if len(user_eligibility) > 0:
        features['credit_score'] = user_eligibility['credit_score'].values[0]
        features['monthly_income'] = user_eligibility['monthly_income'].values[0]
    else:
        # Estimate for unknown users
        features['credit_score'] = 50.0
        features['monthly_income'] = features.get('income_total', 0) / 6  # Approximate monthly from 6-month data

    return features

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ML Loan Eligibility Platform API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/applications", response_model=ApplicationResponse)
async def submit_application(request: ApplicationRequest):
    """Submit a loan application"""

    application_id = str(uuid.uuid4())

    return ApplicationResponse(
        application_id=application_id,
        status="received",
        message="Application received and is being processed",
        created_at=datetime.now().isoformat()
    )

@app.get("/decisions/{application_id}")
async def get_decision(application_id: str):
    """Get decision for an application (mock for demo)"""

    # Mock decision
    return {
        "application_id": application_id,
        "decision": "approved",
        "probability": 0.75,
        "confidence": 0.65,
        "requested_amount": 5000.0,
        "recommended_amount": 5000.0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/evaluate", response_model=DecisionResponse)
async def evaluate_application(user_id: str, requested_amount: float):
    """Evaluate a loan application in real-time"""

    try:
        # Extract features
        features = extract_features(user_id)

        # Create DataFrame with correct feature order
        X = pd.DataFrame([features])[feature_names]

        # Make prediction
        probability = float(model.predict_proba(X)[0, 1])

        # Determine decision
        if probability >= 0.85:
            decision = "auto_approve"
            confidence = probability
        elif probability <= 0.35:
            decision = "auto_reject"
            confidence = 1 - probability
        else:
            decision = "manual_review"
            confidence = abs(probability - 0.5) * 2

        # Calculate recommended amount
        if probability >= 0.8:
            recommended_amount = requested_amount
        elif probability >= 0.6:
            recommended_amount = requested_amount * 0.75
        elif probability >= 0.4:
            recommended_amount = requested_amount * 0.50
        else:
            recommended_amount = min(requested_amount * 0.25, 1000)

        # Get top features for explanation
        feature_values = X.iloc[0].to_dict()
        top_features = sorted(
            [(k, v) for k, v in feature_values.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        explanation = {
            "top_factors": [
                {"feature": f, "value": float(v)} for f, v in top_features
            ],
            "income_total": float(features.get('income_total', 0)),
            "expense_total": float(features.get('expense_total', 0)),
            "balance_avg": float(features.get('balance_mean', 0)),
            "transaction_count": int(features.get('amount_count', 0))
        }

        application_id = str(uuid.uuid4())

        return DecisionResponse(
            application_id=application_id,
            decision=decision,
            probability=probability,
            confidence=confidence,
            requested_amount=requested_amount,
            recommended_amount=recommended_amount,
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/users")
async def list_users():
    """List available users for testing"""
    users = transactions['user_id'].unique()[:20].tolist()
    return {"users": users, "count": len(users)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("Starting ML Loan Eligibility Platform API Server")
    print("="*80)
    print(f"Model: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")
    print(f"Sample users available: {len(transactions['user_id'].unique())}")
    print("\nServer will be available at:")
    print("  - API: http://localhost:8000")
    print("  - Docs: http://localhost:8000/docs")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
