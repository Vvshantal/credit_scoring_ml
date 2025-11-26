"""Decision engine for loan eligibility platform."""

from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
from ..models.predict import LoanEligibilityPredictor
from ..utils.logging import LoggerMixin
from ..utils.config import config


class DecisionType(str, Enum):
    """Decision types."""
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    MANUAL_REVIEW = "manual_review"


class LoanDecision(LoggerMixin):
    """Loan decision result."""

    def __init__(
        self,
        application_id: str,
        decision_type: DecisionType,
        probability: float,
        confidence: float,
        explanation: Optional[Dict] = None,
        loan_amount: Optional[float] = None,
        recommended_amount: Optional[float] = None,
    ):
        """Initialize loan decision.

        Args:
            application_id: Application ID
            decision_type: Type of decision
            probability: Prediction probability
            confidence: Confidence score
            explanation: Explanation dictionary
            loan_amount: Requested loan amount
            recommended_amount: Recommended loan amount
        """
        self.application_id = application_id
        self.decision_type = decision_type
        self.probability = probability
        self.confidence = confidence
        self.explanation = explanation or {}
        self.loan_amount = loan_amount
        self.recommended_amount = recommended_amount
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "application_id": self.application_id,
            "decision": self.decision_type.value,
            "probability": self.probability,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "requested_amount": self.loan_amount,
            "recommended_amount": self.recommended_amount,
            "timestamp": self.timestamp.isoformat(),
        }


class DecisionEngine(LoggerMixin):
    """Decision engine for loan applications."""

    def __init__(
        self,
        predictor: LoanEligibilityPredictor,
        auto_approve_threshold: Optional[float] = None,
        auto_reject_threshold: Optional[float] = None,
        max_loan_amount: Optional[float] = None,
        min_loan_amount: Optional[float] = None,
    ):
        """Initialize decision engine.

        Args:
            predictor: Loan eligibility predictor
            auto_approve_threshold: Threshold for automatic approval
            auto_reject_threshold: Threshold for automatic rejection
            max_loan_amount: Maximum loan amount
            min_loan_amount: Minimum loan amount
        """
        self.predictor = predictor
        self.auto_approve_threshold = auto_approve_threshold or config.decision_engine.auto_approve_threshold
        self.auto_reject_threshold = auto_reject_threshold or config.decision_engine.auto_reject_threshold
        self.max_loan_amount = max_loan_amount or config.decision_engine.max_loan_amount
        self.min_loan_amount = min_loan_amount or config.decision_engine.min_loan_amount

        self.logger.info(
            "decision_engine_initialized",
            auto_approve_threshold=self.auto_approve_threshold,
            auto_reject_threshold=self.auto_reject_threshold,
        )

    def make_decision(
        self,
        application_id: str,
        features: Dict[str, float],
        requested_amount: float,
    ) -> LoanDecision:
        """Make a loan decision.

        Args:
            application_id: Application ID
            features: Application features
            requested_amount: Requested loan amount

        Returns:
            Loan decision
        """
        # Get prediction
        result = self.predictor.predict_single(features)

        probability = result["probability"]
        confidence = result["confidence"]

        # Determine decision type
        if probability >= self.auto_approve_threshold:
            decision_type = DecisionType.AUTO_APPROVE
        elif probability <= self.auto_reject_threshold:
            decision_type = DecisionType.AUTO_REJECT
        else:
            decision_type = DecisionType.MANUAL_REVIEW

        # Calculate recommended amount based on probability
        recommended_amount = self._calculate_recommended_amount(
            requested_amount,
            probability,
        )

        # Create decision
        decision = LoanDecision(
            application_id=application_id,
            decision_type=decision_type,
            probability=probability,
            confidence=confidence,
            loan_amount=requested_amount,
            recommended_amount=recommended_amount,
        )

        self.logger.info(
            "decision_made",
            application_id=application_id,
            decision=decision_type.value,
            probability=probability,
        )

        return decision

    def _calculate_recommended_amount(
        self,
        requested_amount: float,
        probability: float,
    ) -> float:
        """Calculate recommended loan amount.

        Args:
            requested_amount: Requested amount
            probability: Approval probability

        Returns:
            Recommended amount
        """
        # Validate requested amount
        if requested_amount < self.min_loan_amount:
            return self.min_loan_amount
        if requested_amount > self.max_loan_amount:
            requested_amount = self.max_loan_amount

        # Adjust amount based on probability
        if probability >= 0.8:
            # High probability - approve full amount
            return requested_amount
        elif probability >= 0.6:
            # Medium-high probability - approve 75%
            return requested_amount * 0.75
        elif probability >= 0.4:
            # Medium probability - approve 50%
            return requested_amount * 0.50
        else:
            # Low probability - recommend minimum
            return self.min_loan_amount

    def apply_business_rules(
        self,
        decision: LoanDecision,
        applicant_data: Dict[str, Any],
    ) -> LoanDecision:
        """Apply business rules to modify decision.

        Args:
            decision: Initial decision
            applicant_data: Applicant data

        Returns:
            Modified decision
        """
        # Example business rules
        if "previous_defaults" in applicant_data:
            if applicant_data["previous_defaults"] > 0:
                # Automatic rejection for previous defaults
                decision.decision_type = DecisionType.AUTO_REJECT
                decision.explanation["rejection_reason"] = "Previous defaults found"

        if "monthly_income" in applicant_data:
            # Ensure loan payment doesn't exceed 40% of monthly income
            monthly_payment = decision.recommended_amount / 12  # Assuming 1 year term
            max_payment = applicant_data["monthly_income"] * 0.4

            if monthly_payment > max_payment:
                decision.recommended_amount = max_payment * 12

        self.logger.info(
            "business_rules_applied",
            application_id=decision.application_id,
        )

        return decision
