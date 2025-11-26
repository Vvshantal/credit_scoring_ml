"""API schemas for the ML Loan Eligibility Platform."""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator


class ApplicationRequest(BaseModel):
    """Loan application request schema."""

    user_id: str = Field(..., description="Unique user identifier")
    requested_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_purpose: Optional[str] = Field(None, description="Purpose of the loan")
    contact_email: str = Field(..., description="Contact email")
    contact_phone: str = Field(..., description="Contact phone number")

    @validator("requested_amount")
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Loan amount must be positive")
        return v


class ApplicationResponse(BaseModel):
    """Loan application response schema."""

    application_id: str
    status: str
    message: str
    created_at: datetime


class DecisionResponse(BaseModel):
    """Decision response schema."""

    application_id: str
    decision: str
    probability: float
    confidence: float
    requested_amount: float
    recommended_amount: Optional[float] = None
    explanation: Optional[Dict] = None
    timestamp: datetime


class ApplicationStatus(BaseModel):
    """Application status schema."""

    application_id: str
    status: str
    decision: Optional[str] = None
    probability: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class ExplanationResponse(BaseModel):
    """Explanation response schema."""

    application_id: str
    decision: str
    probability: float
    top_factors: List[Dict[str, float]]
    model_version: str


class FeedbackRequest(BaseModel):
    """Feedback request schema."""

    application_id: str
    actual_outcome: str = Field(..., description="Actual loan outcome (approved, rejected, defaulted, repaid)")
    feedback_text: Optional[str] = Field(None, description="Additional feedback")


class FeedbackResponse(BaseModel):
    """Feedback response schema."""

    application_id: str
    message: str
    recorded_at: datetime


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    timestamp: datetime
    version: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    """Metrics response schema."""

    total_applications: int
    approval_rate: float
    avg_processing_time: float
    model_accuracy: Optional[float] = None
    timestamp: datetime
