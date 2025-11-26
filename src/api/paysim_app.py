"""FastAPI application using PaySim credit risk model."""

from datetime import datetime
from typing import Optional, List
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib


# ============== Pydantic Models ==============

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    total_customers: int


class CustomerListResponse(BaseModel):
    users: List[str]
    total: int


class EvaluationResponse(BaseModel):
    application_id: str
    decision: str
    probability: float
    confidence: float
    requested_amount: float
    recommended_amount: float
    explanation: dict
    timestamp: datetime


class CustomerFeaturesResponse(BaseModel):
    customer_id: str
    features: dict
    risk_score: float


# ============== Initialize App ==============

app = FastAPI(
    title="PaySim Credit Risk API",
    description="Credit risk assessment using PaySim mobile money transaction data",
    version="2.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
scaler = None
feature_names = None
customer_features = None


def load_model():
    """Load trained model, scaler, and features."""
    global model, scaler, feature_names, customer_features

    model_dir = Path("models_trained")

    try:
        model = joblib.load(model_dir / "paysim_credit_model.joblib")
        scaler = joblib.load(model_dir / "paysim_scaler.joblib")
        feature_names = joblib.load(model_dir / "paysim_feature_names.joblib")
        print(f"Model loaded with {len(feature_names)} features")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        model = None

    # Load pre-computed customer features
    features_path = Path("data/features/paysim_features_sample.csv")
    if features_path.exists():
        customer_features = pd.read_csv(features_path)
        print(f"Loaded features for {len(customer_features)} customers")
    else:
        customer_features = None
        print("Warning: Customer features not found")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    load_model()


# ============== Endpoints ==============

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "PaySim Credit Risk API",
        "version": "2.0.0",
        "endpoints": ["/health", "/users", "/evaluate", "/customer/{customer_id}"]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now(),
        model_loaded=model is not None,
        total_customers=len(customer_features) if customer_features is not None else 0,
    )


@app.get("/users", response_model=CustomerListResponse, tags=["Customers"])
async def list_users(
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0),
    risky_only: bool = Query(default=False),
):
    """List available customers for evaluation."""
    if customer_features is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded")

    df = customer_features.copy()

    if risky_only and 'is_risky' in df.columns:
        df = df[df['is_risky'] == 1]

    users = df['nameOrig'].iloc[offset:offset + limit].tolist()

    return CustomerListResponse(
        users=users,
        total=len(customer_features),
    )


@app.get("/customer/{customer_id}", response_model=CustomerFeaturesResponse, tags=["Customers"])
async def get_customer_features(customer_id: str):
    """Get features for a specific customer."""
    if customer_features is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded")

    customer = customer_features[customer_features['nameOrig'] == customer_id]

    if customer.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    customer_row = customer.iloc[0]

    # Calculate risk score
    risk_score = 0.0
    if model is not None and scaler is not None:
        features = customer_row[feature_names].values.reshape(1, -1)
        features_scaled = scaler.transform(features)
        risk_score = float(model.predict_proba(features_scaled)[0][1])

    # Build feature dict
    feature_dict = {
        "income_total": float(customer_row.get('income_total', 0)),
        "income_mean": float(customer_row.get('income_mean', 0)),
        "expense_total": float(customer_row.get('expense_total', 0)),
        "expense_mean": float(customer_row.get('expense_mean', 0)),
        "balance_mean": float(customer_row.get('balance_mean', 0)),
        "balance_min": float(customer_row.get('balance_min', 0)),
        "total_transactions": int(customer_row.get('total_transactions', 0)),
        "unique_recipients": int(customer_row.get('unique_recipients', 0)),
        "overdraw_attempt_ratio": float(customer_row.get('overdraw_attempt_ratio', 0)),
        "zero_balance_freq": float(customer_row.get('zero_balance_freq', 0)),
    }

    return CustomerFeaturesResponse(
        customer_id=customer_id,
        features=feature_dict,
        risk_score=risk_score,
    )


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_customer(
    user_id: str = Query(..., description="Customer ID"),
    requested_amount: float = Query(..., description="Requested loan amount"),
):
    """Evaluate a customer for loan eligibility."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if customer_features is None:
        raise HTTPException(status_code=503, detail="Customer data not loaded")

    # Find customer
    customer = customer_features[customer_features['nameOrig'] == user_id]

    if customer.empty:
        raise HTTPException(status_code=404, detail=f"Customer {user_id} not found")

    customer_row = customer.iloc[0]

    # Get features and predict
    features = customer_row[feature_names].values.reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predict
    probability = float(model.predict_proba(features_scaled)[0][1])
    risk_score = probability

    # Decision logic
    if risk_score < 0.3:
        decision = "auto_approve"
        confidence = 1 - risk_score
        recommended_amount = requested_amount
    elif risk_score < 0.6:
        decision = "manual_review"
        confidence = 0.5
        recommended_amount = requested_amount * 0.7
    else:
        decision = "auto_reject"
        confidence = risk_score
        recommended_amount = 0

    # Build explanation
    explanation = {
        "risk_score": round(risk_score, 4),
        "income_total": float(customer_row.get('income_total', 0)),
        "expense_total": float(customer_row.get('expense_total', 0)),
        "balance_avg": float(customer_row.get('balance_mean', 0)),
        "transaction_count": int(customer_row.get('total_transactions', 0)),
        "overdraw_ratio": float(customer_row.get('overdraw_attempt_ratio', 0)),
        "zero_balance_freq": float(customer_row.get('zero_balance_freq', 0)),
        "top_risk_factors": get_top_risk_factors(customer_row, feature_names),
    }

    return EvaluationResponse(
        application_id=str(uuid.uuid4()),
        decision=decision,
        probability=1 - risk_score,  # Approval probability
        confidence=confidence,
        requested_amount=requested_amount,
        recommended_amount=recommended_amount,
        explanation=explanation,
        timestamp=datetime.now(),
    )


def get_top_risk_factors(customer_row, feature_names, top_n=5):
    """Get top risk factors for a customer."""
    # Feature importance from model (if Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        top_factors = []
        for feat, imp in feature_importance[:top_n]:
            value = float(customer_row.get(feat, 0))
            top_factors.append({
                "feature": feat,
                "importance": round(imp, 4),
                "value": round(value, 2),
            })
        return top_factors

    return []


@app.get("/model/features", tags=["Model"])
async def get_model_features():
    """Get list of features used by the model."""
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "features": feature_names,
        "count": len(feature_names),
    }


@app.get("/model/importance", tags=["Model"])
async def get_feature_importance():
    """Get feature importance from the model."""
    if model is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_list = [
            {"feature": feat, "importance": float(imp)}
            for feat, imp in sorted(zip(feature_names, importances),
                                   key=lambda x: x[1], reverse=True)
        ]
        return {"feature_importance": importance_list}

    return {"feature_importance": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
