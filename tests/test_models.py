"""Tests for model training module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.train import ModelTrainer
from src.models.predict import LoanEligibilityPredictor


@pytest.fixture
def sample_dataset():
    """Create sample classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y_series = pd.Series(y)

    return X_df, y_series


def test_model_trainer_initialization():
    """Test model trainer initialization."""
    trainer = ModelTrainer(test_size=0.2, random_state=42)
    assert trainer.test_size == 0.2
    assert trainer.random_state == 42


def test_data_preparation(sample_dataset):
    """Test data preparation."""
    X, y = sample_dataset
    trainer = ModelTrainer()

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X, y, apply_smote=False
    )

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0


def test_logistic_regression_training(sample_dataset):
    """Test logistic regression training."""
    X, y = sample_dataset
    trainer = ModelTrainer()

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X, y, apply_smote=False
    )

    model, metrics = trainer.train_logistic_regression(
        X_train, y_train, X_val, y_val
    )

    assert model is not None
    assert 'accuracy' in metrics
    assert 'roc_auc' in metrics
    assert metrics['accuracy'] > 0


def test_random_forest_training(sample_dataset):
    """Test random forest training."""
    X, y = sample_dataset
    trainer = ModelTrainer()

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X, y, apply_smote=False
    )

    model, metrics = trainer.train_random_forest(
        X_train, y_train, X_val, y_val
    )

    assert model is not None
    assert metrics['accuracy'] > 0


def test_predictor_initialization():
    """Test predictor initialization."""
    predictor = LoanEligibilityPredictor(threshold=0.6)
    assert predictor.threshold == 0.6


def test_prediction_with_confidence(sample_dataset):
    """Test prediction with confidence."""
    X, y = sample_dataset
    trainer = ModelTrainer()

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X, y, apply_smote=False
    )

    model, _ = trainer.train_logistic_regression(
        X_train, y_train, X_val, y_val
    )

    predictor = LoanEligibilityPredictor(threshold=0.5)
    predictor.model = model

    result = predictor.predict_with_confidence(X_test[:10])

    assert 'predictions' in result
    assert 'probabilities' in result
    assert 'confidence' in result
    assert len(result['predictions']) == 10
