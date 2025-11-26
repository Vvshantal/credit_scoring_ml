"""Prediction module for the ML Loan Eligibility Platform."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import pandas as pd
import numpy as np
from ..utils.logging import LoggerMixin


class LoanEligibilityPredictor(LoggerMixin):
    """Predictor for loan eligibility assessment."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """Initialize predictor.

        Args:
            model_path: Path to trained model
            threshold: Prediction threshold
        """
        self.model = None
        self.threshold = threshold
        self.feature_names: Optional[List[str]] = None

        if model_path:
            self.load_model(model_path)

        self.logger.info("predictor_initialized", threshold=threshold)

    def load_model(self, model_path: str) -> None:
        """Load trained model from disk.

        Args:
            model_path: Path to saved model
        """
        self.model = joblib.load(model_path)
        self.logger.info("model_loaded", path=model_path)

    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Predict loan eligibility.

        Args:
            features: Feature matrix

        Returns:
            Binary predictions (0 = reject, 1 = approve)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Get probability predictions
        probabilities = self.predict_proba(features)

        # Apply threshold
        predictions = (probabilities >= self.threshold).astype(int)

        self.logger.info(
            "predictions_made",
            num_predictions=len(predictions),
            approval_rate=predictions.mean(),
        )

        return predictions

    def predict_proba(
        self,
        features: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Predict loan eligibility probabilities.

        Args:
            features: Feature matrix

        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Make probability predictions
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[:, 1]
        else:
            # For models without predict_proba (e.g., neural networks)
            probabilities = self.model.predict(features).flatten()

        return probabilities

    def predict_with_confidence(
        self,
        features: Union[pd.DataFrame, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Predict with confidence scores.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with predictions, probabilities, and confidence
        """
        probabilities = self.predict_proba(features)
        predictions = (probabilities >= self.threshold).astype(int)

        # Calculate confidence (distance from threshold)
        confidence = np.abs(probabilities - self.threshold)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "confidence": confidence,
        }

    def predict_single(
        self,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Predict for a single application.

        Args:
            features: Dictionary of feature values

        Returns:
            Prediction result
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Make prediction
        result = self.predict_with_confidence(df)

        return {
            "approved": bool(result["predictions"][0]),
            "probability": float(result["probabilities"][0]),
            "confidence": float(result["confidence"][0]),
            "decision": self._get_decision_category(result["probabilities"][0]),
        }

    def _get_decision_category(
        self,
        probability: float,
        auto_approve_threshold: float = 0.85,
        auto_reject_threshold: float = 0.35,
    ) -> str:
        """Categorize decision based on probability.

        Args:
            probability: Predicted probability
            auto_approve_threshold: Threshold for automatic approval
            auto_reject_threshold: Threshold for automatic rejection

        Returns:
            Decision category
        """
        if probability >= auto_approve_threshold:
            return "auto_approve"
        elif probability <= auto_reject_threshold:
            return "auto_reject"
        else:
            return "manual_review"

    def batch_predict(
        self,
        features_list: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Predict for multiple applications.

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame(features_list)

        # Make predictions
        results = self.predict_with_confidence(df)

        # Format results
        predictions = []
        for i in range(len(df)):
            predictions.append({
                "approved": bool(results["predictions"][i]),
                "probability": float(results["probabilities"][i]),
                "confidence": float(results["confidence"][i]),
                "decision": self._get_decision_category(results["probabilities"][i]),
            })

        return predictions
