#!/bin/bash

# Setup script for ML Loan Eligibility Platform

echo "=========================================="
echo "ML Loan Eligibility Platform Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Python dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw data/processed data/features
mkdir -p models_trained
mkdir -p logs
mkdir -p config
echo "✓ Directories created"

# Copy environment file
echo ""
echo "Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ .env file created (please edit with your settings)"
else
    echo "✓ .env file already exists"
fi

# Generate sample data
echo ""
echo "Generating sample data..."
python scripts/generate_sample_data.py
echo "✓ Sample data generated"

# Train model
echo ""
echo "Training initial model (this may take a few minutes)..."
python scripts/train_model.py
echo "✓ Model trained and saved"

# Setup complete
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Start the API server:"
echo "   uvicorn src.api.app:app --reload"
echo ""
echo "3. (Optional) Start with Docker:"
echo "   cd docker && docker-compose up -d"
echo ""
echo "4. (Optional) Start the frontend:"
echo "   cd frontend && npm install && npm start"
echo ""
echo "Access the application at:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Frontend: http://localhost:3000"
echo ""
