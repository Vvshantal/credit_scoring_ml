"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from src.features.engineer import FeatureEngineer
from src.features.selector import FeatureSelector


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n_users = 10
    n_transactions = 100

    data = {
        'user_id': np.random.choice([f'user_{i}' for i in range(n_users)], n_transactions),
        'transaction_date': pd.date_range('2023-01-01', periods=n_transactions, freq='D'),
        'amount': np.random.uniform(10, 1000, n_transactions),
        'type': np.random.choice(['incoming', 'outgoing'], n_transactions),
        'balance': np.random.uniform(100, 5000, n_transactions),
    }

    return pd.DataFrame(data)


def test_feature_engineer_initialization():
    """Test feature engineer initialization."""
    engineer = FeatureEngineer(rolling_windows=[7, 30])
    assert engineer.rolling_windows == [7, 30]


def test_income_stability_features(sample_transaction_data):
    """Test income stability feature creation."""
    engineer = FeatureEngineer()
    features = engineer.create_income_stability_features(sample_transaction_data)

    assert 'avg_transaction_amount_incoming' in features.columns
    assert 'income_consistency' in features.columns
    assert len(features) <= sample_transaction_data['user_id'].nunique()


def test_rolling_window_features(sample_transaction_data):
    """Test rolling window feature creation."""
    engineer = FeatureEngineer(rolling_windows=[7])
    features = engineer.create_rolling_window_features(sample_transaction_data)

    assert 'rolling_7d_transaction_count' in features.columns
    assert 'rolling_7d_total_amount' in features.columns


def test_feature_selector_initialization():
    """Test feature selector initialization."""
    selector = FeatureSelector(target_features=20)
    assert selector.target_features == 20


def test_feature_selector_correlation(sample_transaction_data):
    """Test correlation analysis."""
    # Create some numeric features
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
    })

    selector = FeatureSelector()
    corr_matrix, to_drop = selector.calculate_correlation_matrix(X, threshold=0.95)

    assert corr_matrix.shape == (3, 3)
    assert isinstance(to_drop, list)


def test_feature_selector_mutual_information():
    """Test mutual information calculation."""
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
    })
    y = pd.Series(np.random.choice([0, 1], 100))

    selector = FeatureSelector()
    mi_scores = selector.calculate_mutual_information(X, y)

    assert len(mi_scores) == 2
    assert 'mi_score' in mi_scores.columns
