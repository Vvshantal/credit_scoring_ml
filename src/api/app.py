"""FastAPI application for the ML Loan Eligibility Platform."""

from datetime import datetime
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    ApplicationRequest,
    ApplicationResponse,
    DecisionResponse,
    ApplicationStatus,
    ExplanationResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
)
from ..models.predict import LoanEligibilityPredictor
from ..decision_engine.engine import DecisionEngine
from ..utils.config import config
from ..utils.logging import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="ML Loan Eligibility Platform API",
    description="API for ML-driven loan eligibility assessment",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = get_logger(__name__)

# Global state (in production, use proper dependency injection)
predictor: Optional[LoanEligibilityPredictor] = None
decision_engine: Optional[DecisionEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global predictor, decision_engine

    logger.info("starting_api_server")

    # Load model (in production, load from model registry)
    try:
        predictor = LoanEligibilityPredictor(
            model_path=f"{config.model.model_path}/best_model.joblib",
            threshold=config.model.prediction_threshold,
        )
        decision_engine = DecisionEngine(predictor)
        logger.info("services_initialized")
    except Exception as e:
        logger.error("failed_to_initialize_services", error=str(e))


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "ML Loan Eligibility Platform API", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        timestamp=datetime.now(),
        version="1.0.0",
        model_loaded=predictor is not None,
    )


@app.post("/applications", response_model=ApplicationResponse, tags=["Applications"])
async def submit_application(request: ApplicationRequest):
    """Submit a new loan application.

    This endpoint receives a loan application and returns an application ID
    for tracking purposes.
    """
    try:
        # Generate application ID
        application_id = str(uuid.uuid4())

        logger.info(
            "application_submitted",
            application_id=application_id,
            user_id=request.user_id,
            amount=request.requested_amount,
        )

        # In production, save to database
        # For now, return success response

        return ApplicationResponse(
            application_id=application_id,
            status="received",
            message="Application received and is being processed",
            created_at=datetime.now(),
        )

    except Exception as e:
        logger.error("application_submission_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process application",
        )


@app.get("/applications/{application_id}", response_model=ApplicationStatus, tags=["Applications"])
async def get_application_status(application_id: str):
    """Get application status.

    Retrieve the current status and decision for a loan application.
    """
    # In production, fetch from database
    # For now, return mock response

    return ApplicationStatus(
        application_id=application_id,
        status="processed",
        decision="approved",
        probability=0.87,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@app.get("/decisions/{application_id}", response_model=DecisionResponse, tags=["Decisions"])
async def get_decision(application_id: str):
    """Get loan decision for an application.

    Returns the detailed decision including probability, confidence,
    and recommended loan amount.
    """
    # In production, fetch decision from database
    # For now, return mock response

    return DecisionResponse(
        application_id=application_id,
        decision="approved",
        probability=0.87,
        confidence=0.72,
        requested_amount=5000.0,
        recommended_amount=5000.0,
        explanation={"top_factors": ["income_stability", "repayment_history"]},
        timestamp=datetime.now(),
    )


@app.post("/evaluate", response_model=DecisionResponse, tags=["Decisions"])
async def evaluate_application(
    user_id: str,
    requested_amount: float,
    features: dict,
):
    """Evaluate a loan application in real-time.

    This endpoint performs real-time evaluation without saving the application.
    Useful for "what-if" scenarios.
    """
    if predictor is None or decision_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    try:
        # Generate temporary application ID
        application_id = str(uuid.uuid4())

        # Make decision
        decision = decision_engine.make_decision(
            application_id=application_id,
            features=features,
            requested_amount=requested_amount,
        )

        logger.info(
            "application_evaluated",
            application_id=application_id,
            decision=decision.decision_type.value,
        )

        return DecisionResponse(
            application_id=application_id,
            decision=decision.decision_type.value,
            probability=decision.probability,
            confidence=decision.confidence,
            requested_amount=requested_amount,
            recommended_amount=decision.recommended_amount,
            explanation=decision.explanation,
            timestamp=decision.timestamp,
        )

    except Exception as e:
        logger.error("evaluation_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to evaluate application",
        )


@app.get("/explanations/{application_id}", response_model=ExplanationResponse, tags=["Explanations"])
async def get_explanation(application_id: str):
    """Get explanation for a loan decision.

    Returns the top factors that influenced the decision.
    """
    # In production, generate explanation using SHAP/LIME
    # For now, return mock response

    return ExplanationResponse(
        application_id=application_id,
        decision="approved",
        probability=0.87,
        top_factors=[
            {"income_stability": 0.25},
            {"repayment_history": 0.18},
            {"transaction_regularity": 0.15},
        ],
        model_version="1.0.0",
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on loan outcome.

    This endpoint collects actual loan outcomes for model retraining
    and performance monitoring.
    """
    logger.info(
        "feedback_received",
        application_id=request.application_id,
        outcome=request.actual_outcome,
    )

    # In production, save to database for model retraining

    return FeedbackResponse(
        application_id=request.application_id,
        message="Feedback recorded successfully",
        recorded_at=datetime.now(),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get platform metrics.

    Returns key performance indicators for the platform.
    """
    # In production, fetch from monitoring database
    # For now, return mock metrics

    return MetricsResponse(
        total_applications=15234,
        approval_rate=0.67,
        avg_processing_time=142.5,
        model_accuracy=0.89,
        timestamp=datetime.now(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        path=str(request.url),
        method=request.method,
        error=str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        workers=config.api.workers,
    )
