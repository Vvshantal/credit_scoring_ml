"""Database models and connection for the ML Loan Eligibility Platform."""

from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..utils.config import config

# Create database engine
engine = create_engine(
    config.database.url,
    pool_size=config.database.pool_size,
    max_overflow=config.database.max_overflow,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class Application(Base):
    """Loan application model."""

    __tablename__ = "applications"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    requested_amount = Column(Float, nullable=False)
    loan_purpose = Column(String, nullable=True)
    contact_email = Column(String, nullable=False)
    contact_phone = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Decision(Base):
    """Loan decision model."""

    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String, nullable=False, index=True)
    decision_type = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    recommended_amount = Column(Float, nullable=True)
    explanation = Column(JSON, nullable=True)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)


class Feedback(Base):
    """Loan outcome feedback model."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String, nullable=False, index=True)
    actual_outcome = Column(String, nullable=False)
    feedback_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class AuditLog(Base):
    """Audit log model."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.now)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
