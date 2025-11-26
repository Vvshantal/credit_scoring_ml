"""Model evaluation and fairness module."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from ..utils.logging import LoggerMixin


class ModelEvaluator(LoggerMixin):
    """Evaluator for model performance and fairness."""

    def __init__(self):
        """Initialize evaluator."""
        self.logger.info("model_evaluator_initialized")

    def evaluate_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        groups: np.ndarray,
        group_names: Optional[Dict] = None,
    ) -> Dict:
        """Evaluate fairness across demographic groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            groups: Group membership array
            group_names: Optional mapping of group IDs to names

        Returns:
            Fairness metrics dictionary
        """
        unique_groups = np.unique(groups)
        fairness_metrics = {}

        for group_id in unique_groups:
            group_name = group_names.get(group_id, group_id) if group_names else group_id
            mask = groups == group_id

            # Calculate metrics for this group
            group_approval_rate = y_pred[mask].mean()
            group_default_rate = y_true[mask].mean() if y_true[mask].sum() > 0 else 0
            group_fpr = ((y_pred[mask] == 1) & (y_true[mask] == 0)).sum() / max((y_true[mask] == 0).sum(), 1)
            group_fnr = ((y_pred[mask] == 0) & (y_true[mask] == 1)).sum() / max((y_true[mask] == 1).sum(), 1)

            fairness_metrics[group_name] = {
                "approval_rate": float(group_approval_rate),
                "default_rate": float(group_default_rate),
                "false_positive_rate": float(group_fpr),
                "false_negative_rate": float(group_fnr),
                "sample_size": int(mask.sum()),
            }

        # Calculate disparate impact
        approval_rates = [m["approval_rate"] for m in fairness_metrics.values()]
        disparate_impact = min(approval_rates) / max(approval_rates) if max(approval_rates) > 0 else 0

        fairness_metrics["overall"] = {
            "disparate_impact": float(disparate_impact),
            "passes_80_rule": disparate_impact >= 0.8,
        }

        self.logger.info("fairness_evaluation_complete", metrics=fairness_metrics)

        return fairness_metrics


class ModelInterpreter(LoggerMixin):
    """Interpreter for model explanations using SHAP and LIME."""

    def __init__(self, model, feature_names: List[str]):
        """Initialize interpreter.

        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
        self.logger.info("model_interpreter_initialized", features=len(feature_names))

    def initialize_shap(self, X_background: np.ndarray, max_samples: int = 100):
        """Initialize SHAP explainer.

        Args:
            X_background: Background dataset for SHAP
            max_samples: Maximum samples for background
        """
        # Use subset if dataset is large
        if len(X_background) > max_samples:
            X_background = X_background[np.random.choice(len(X_background), max_samples, replace=False)]

        self.shap_explainer = shap.TreeExplainer(self.model)
        self.logger.info("shap_initialized", background_samples=len(X_background))

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values for predictions.

        Args:
            X: Feature matrix

        Returns:
            SHAP values
        """
        if self.shap_explainer is None:
            self.initialize_shap(X)

        shap_values = self.shap_explainer.shap_values(X)
        return shap_values

    def explain_prediction_shap(self, X: np.ndarray, index: int = 0) -> Dict:
        """Explain a single prediction using SHAP.

        Args:
            X: Feature matrix
            index: Index of prediction to explain

        Returns:
            Explanation dictionary
        """
        shap_values = self.get_shap_values(X[index:index+1])

        # Get feature contributions
        contributions = dict(zip(self.feature_names, shap_values[0]))
        sorted_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "top_features": sorted_contributions[:10],
            "all_contributions": contributions,
        }

    def initialize_lime(self, X_train: np.ndarray, mode: str = "classification"):
        """Initialize LIME explainer.

        Args:
            X_train: Training data for LIME
            mode: 'classification' or 'regression'
        """
        self.lime_explainer = LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            mode=mode,
            random_state=42,
        )
        self.logger.info("lime_initialized")

    def explain_prediction_lime(self, X: np.ndarray, index: int = 0, num_features: int = 10) -> Dict:
        """Explain a single prediction using LIME.

        Args:
            X: Feature matrix
            index: Index of prediction to explain
            num_features: Number of top features to show

        Returns:
            Explanation dictionary
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_lime() first.")

        explanation = self.lime_explainer.explain_instance(
            X[index],
            self.model.predict_proba,
            num_features=num_features,
        )

        return {
            "top_features": explanation.as_list(),
            "score": explanation.score,
        }
