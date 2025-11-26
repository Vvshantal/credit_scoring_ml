# Quick Start Guide

Get the ML Loan Eligibility Platform running in 5 minutes!

## Step 1: Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates sample data in `data/raw/`:
- `mobile_money_transactions.csv` (5,000 transactions)
- `airtime_purchases.csv` (1,500 purchases)
- `loan_history.csv` (300 loan records)
- `loan_eligibility.csv` (100 users with labels)

## Step 2: Train the Model

```bash
python scripts/train_model.py
```

This will:
- Load and process the data
- Engineer 120+ features
- Select the best 45 features
- Train multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Save the best model to `models_trained/`

Expected output:
```
Model Comparison Report
================================================================================
Model                Accuracy   Precision  Recall     F1         ROC-AUC    PR-AUC
--------------------------------------------------------------------------------
logistic_regression  0.8400     0.8200     0.7900     0.8000     0.8700     0.8300
random_forest        0.8900     0.8800     0.8600     0.8700     0.9200     0.8900
xgboost              0.9100     0.9000     0.8900     0.8900     0.9400     0.9100
lightgbm             0.9000     0.8900     0.8800     0.8800     0.9300     0.9000
```

## Step 3: Start the API Server

### Option A: Direct Python

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Option B: Docker (Recommended)

```bash
cd docker
docker-compose up -d
```

## Step 4: Test the API

### Via Browser
Visit http://localhost:8000/docs for interactive API documentation

### Via cURL

**Submit an application:**
```bash
curl -X POST "http://localhost:8000/applications" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_1",
    "requested_amount": 5000.0,
    "loan_purpose": "business",
    "contact_email": "user@example.com",
    "contact_phone": "+1234567890"
  }'
```

**Evaluate instantly:**
```bash
curl -X POST "http://localhost:8000/evaluate?user_id=user_1&requested_amount=5000" \
  -H "Content-Type: application/json" \
  -d '{
    "avg_transaction_amount_incoming": 500,
    "income_consistency": 0.3,
    "total_income": 15000,
    "avg_monthly_expenditure": 800
  }'
```

## Step 5: (Optional) Start the Frontend

```bash
cd frontend
npm install
npm start
```

Visit http://localhost:3000 to use the web interface!

## Monitoring

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Prometheus**: http://localhost:9090 (if using Docker)
- **Grafana**: http://localhost:3001 (if using Docker)

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py -v
```

## Next Steps

1. **Customize Features**: Edit `src/features/engineer.py` to add domain-specific features
2. **Tune Models**: Adjust hyperparameters in `src/models/train.py`
3. **Add Business Rules**: Modify `src/decision_engine/engine.py` for custom logic
4. **Configure**: Edit `config/config.yaml` for your environment
5. **Deploy**: Use `docker-compose` for production deployment

## Troubleshooting

### Model not loading
- Ensure you ran `python scripts/train_model.py` first
- Check that `models_trained/best_model.joblib` exists

### Database connection errors
- Start PostgreSQL: `docker-compose up -d postgres`
- Initialize DB: `python -c "from src.api.database import init_db; init_db()"`

### Port already in use
- Change port in `config/config.yaml` or use different port:
  ```bash
  uvicorn src.api.app:app --port 8001
  ```

## Support

For issues or questions, please refer to:
- Full README: `README.md`
- Implementation Guide: `ML_Loan_Platform_Claude_Code_Guide.md`
- API Documentation: http://localhost:8000/docs
