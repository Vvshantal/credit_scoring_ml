# Multi-stage build for ML Loan Eligibility Platform

# Stage 1: Base image with Python dependencies
FROM python:3.9-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application image
FROM base as app

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY models_trained/ ./models_trained/

# Create necessary directories
RUN mkdir -p logs data/raw data/processed data/features

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
