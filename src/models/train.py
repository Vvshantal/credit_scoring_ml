"""Model training module for the ML Loan Eligibility Platform."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
from ..utils.logging import LoggerMixin


class ModelTrainer(LoggerMixin):
    """Model trainer for loan eligibility prediction."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize model trainer.

        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.logger.info(
            "model_trainer_initialized",
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        apply_smote: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Prepare data for model training.

        Args:
            X: Feature DataFrame
            y: Target variable
            apply_smote: Whether to apply SMOTE for class imbalance

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp,
        )

        # Apply SMOTE to training data if requested
        if apply_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            self.logger.info(
                "smote_applied",
                original_samples=len(y_temp) * (1 - self.val_size),
                resampled_samples=len(y_train),
            )

        self.logger.info(
            "data_prepared",
            train_samples=len(y_train),
            val_samples=len(y_val),
            test_samples=len(y_test),
            train_class_dist=y_train.value_counts().to_dict(),
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs,
    ) -> Tuple[LogisticRegression, Dict[str, float]]:
        """Train Logistic Regression model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional parameters for LogisticRegression

        Returns:
            Tuple of (trained model, metrics dict)
        """
        self.logger.info("training_logistic_regression")

        # Default parameters
        params = {
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 1000,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)

        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = self._evaluate_model(model, X_val, y_val)

        self.models["logistic_regression"] = model
        self.results["logistic_regression"] = metrics

        self.logger.info("logistic_regression_trained", metrics=metrics)

        return model, metrics

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs,
    ) -> Tuple[RandomForestClassifier, Dict[str, float]]:
        """Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional parameters for RandomForestClassifier

        Returns:
            Tuple of (trained model, metrics dict)
        """
        self.logger.info("training_random_forest")

        # Default parameters
        params = {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = self._evaluate_model(model, X_val, y_val)

        self.models["random_forest"] = model
        self.results["random_forest"] = metrics

        self.logger.info("random_forest_trained", metrics=metrics)

        return model, metrics

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional parameters for XGBClassifier

        Returns:
            Tuple of (trained model, metrics dict)
        """
        self.logger.info("training_xgboost")

        # Default parameters
        params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": -1,
            "eval_metric": "auc",
        }
        params.update(kwargs)

        # Train model with early stopping
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        metrics = self._evaluate_model(model, X_val, y_val)

        self.models["xgboost"] = model
        self.results["xgboost"] = metrics

        self.logger.info("xgboost_trained", metrics=metrics)

        return model, metrics

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        **kwargs,
    ) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
        """Train LightGBM model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional parameters for LGBMClassifier

        Returns:
            Tuple of (trained model, metrics dict)
        """
        self.logger.info("training_lightgbm")

        # Default parameters
        params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        params.update(kwargs)

        # Train model
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        # Evaluate
        metrics = self._evaluate_model(model, X_val, y_val)

        self.models["lightgbm"] = model
        self.results["lightgbm"] = metrics

        self.logger.info("lightgbm_trained", metrics=metrics)

        return model, metrics

    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            model: Trained model
            X: Features
            y: True labels

        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_pred_proba),
            "pr_auc": average_precision_score(y, y_pred_proba),
        }

        return metrics

    def optimize_hyperparameters(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna.

        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials

        Returns:
            Best hyperparameters
        """
        self.logger.info(
            "optimizing_hyperparameters",
            model_type=model_type,
            n_trials=n_trials,
        )

        def objective(trial):
            if model_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
                model = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=-1)

            elif model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = xgb.XGBClassifier(**params, random_state=self.random_state, n_jobs=-1)

            elif model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                }
                model = lgb.LGBMClassifier(**params, random_state=self.random_state, n_jobs=-1, verbose=-1)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)

            return auc

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.logger.info(
            "hyperparameter_optimization_complete",
            model_type=model_type,
            best_auc=study.best_value,
            best_params=study.best_params,
        )

        return study.best_params

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Train all baseline models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary of model results
        """
        self.logger.info("training_all_models")

        # Train each model
        self.train_logistic_regression(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)

        # Find best model
        best_auc = 0
        for model_name, metrics in self.results.items():
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                self.best_model_name = model_name
                self.best_model = self.models[model_name]

        self.logger.info(
            "all_models_trained",
            best_model=self.best_model_name,
            best_auc=best_auc,
        )

        return self.results

    def save_model(
        self,
        model_name: str,
        save_path: str,
    ) -> None:
        """Save a trained model to disk.

        Args:
            model_name: Name of the model to save
            save_path: Path to save the model
        """
        if model_name not in self.models:
            self.logger.error("model_not_found", model_name=model_name)
            return

        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.models[model_name], save_path)

        self.logger.info(
            "model_saved",
            model_name=model_name,
            path=save_path,
        )

    def load_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
    ) -> Any:
        """Load a trained model from disk.

        Args:
            model_path: Path to the saved model
            model_name: Name to assign to the loaded model

        Returns:
            Loaded model
        """
        model = joblib.load(model_path)

        if model_name:
            self.models[model_name] = model

        self.logger.info(
            "model_loaded",
            path=model_path,
            model_name=model_name,
        )

        return model

    def generate_comparison_report(self) -> str:
        """Generate model comparison report.

        Returns:
            Comparison report as string
        """
        if not self.results:
            return "No models have been trained yet."

        report_lines = [
            "Model Comparison Report",
            "=" * 80,
            f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10} {'PR-AUC':<10}",
            "-" * 80,
        ]

        for model_name, metrics in self.results.items():
            report_lines.append(
                f"{model_name:<20} "
                f"{metrics['accuracy']:<10.4f} "
                f"{metrics['precision']:<10.4f} "
                f"{metrics['recall']:<10.4f} "
                f"{metrics['f1_score']:<10.4f} "
                f"{metrics['roc_auc']:<10.4f} "
                f"{metrics['pr_auc']:<10.4f}"
            )

        report_lines.append("-" * 80)
        if self.best_model_name:
            report_lines.append(f"Best Model: {self.best_model_name}")

        return "\n".join(report_lines)
