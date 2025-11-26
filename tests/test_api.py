"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ML Loan Eligibility Platform API" in response.json()["message"]


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data


def test_submit_application():
    """Test application submission."""
    application_data = {
        "user_id": "test_user_123",
        "requested_amount": 5000.0,
        "loan_purpose": "business",
        "contact_email": "test@example.com",
        "contact_phone": "+1234567890"
    }

    response = client.post("/applications", json=application_data)
    assert response.status_code == 200
    data = response.json()
    assert "application_id" in data
    assert data["status"] == "received"


def test_get_application_status():
    """Test getting application status."""
    application_id = "test_app_123"
    response = client.get(f"/applications/{application_id}")
    assert response.status_code == 200
    data = response.json()
    assert "application_id" in data
    assert "status" in data


def test_get_decision():
    """Test getting decision."""
    application_id = "test_app_123"
    response = client.get(f"/decisions/{application_id}")
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "probability" in data


def test_submit_feedback():
    """Test submitting feedback."""
    feedback_data = {
        "application_id": "test_app_123",
        "actual_outcome": "repaid",
        "feedback_text": "Loan was repaid on time"
    }

    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_get_metrics():
    """Test getting metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_applications" in data
    assert "approval_rate" in data
