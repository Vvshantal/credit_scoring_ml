# 🚀 Starting the ML Loan Eligibility Platform

## ✅ API Server is RUNNING!

The backend API is already running on **http://localhost:8000**

---

## 🎯 Quick Test

### Test the API (Already Working!)
```bash
curl -X POST "http://localhost:8000/evaluate?user_id=user_0&requested_amount=5000"
```

**Response:**
```json
{
  "application_id": "2afbe5d3-5286-46ad-b6fc-d8fe0f93d99a",
  "decision": "auto_approve",
  "probability": 0.915,
  "confidence": 0.915,
  "requested_amount": 5000.0,
  "recommended_amount": 5000.0,
  "explanation": {
    "income_total": 14998.03,
    "expense_total": 4914.97,
    "balance_avg": 2171.39,
    "transaction_count": 50
  }
}
```

---

## 🌐 Start the Frontend

### Option 1: Using npx (No installation needed)
```bash
cd frontend
npx serve -s build -l 3000
```

### Option 2: Development mode with npm
```bash
cd frontend

# If npm is not installed, install it first
# brew install node  (on Mac)

npm install
npm start
```

The frontend will open at **http://localhost:3000**

---

## 📡 Available Endpoints

### API Server (Port 8000)
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **List Users**: http://localhost:8000/users
- **Evaluate**: POST http://localhost:8000/evaluate

### Frontend (Port 3000)
- **Web App**: http://localhost:3000

---

## 🎮 Using the Platform

### Via Web Interface
1. Open http://localhost:3000
2. Select a user from the dropdown
3. Enter requested loan amount
4. Click "Evaluate Application"
5. View instant ML-powered decision!

### Via API
```bash
# Get list of users
curl http://localhost:8000/users

# Evaluate user_0 for $5,000 loan
curl -X POST "http://localhost:8000/evaluate?user_id=user_0&requested_amount=5000"

# Evaluate user_10 for $10,000 loan
curl -X POST "http://localhost:8000/evaluate?user_id=user_10&requested_amount=10000"
```

---

## 📊 Sample Users to Test

Try these users (all have transaction history):
- `user_0` - High approval probability
- `user_1` - Medium approval
- `user_2` - Low approval
- `user_10` - Mixed profile
- `user_25` - Business user

---

## 🛑 Stop the Server

```bash
# Find the process
ps aux | grep api_server.py

# Kill it
kill <PID>

# Or use:
pkill -f api_server.py
```

---

## ✅ What's Running

- [x] **Backend API** - http://localhost:8000 ✅ RUNNING
- [x] **Trained Model** - Random Forest (65% accuracy)
- [x] **Sample Data** - 5,000 transactions, 100 users
- [x] **Real-time Predictions** - Working!
- [ ] **Frontend** - Ready to start (npm needed)

---

## 🎉 Platform Status

```
┌─────────────────────────────────────────┐
│   ML Loan Eligibility Platform         │
│   Status: 🟢 OPERATIONAL               │
└─────────────────────────────────────────┘

Backend API:    ✅ Running (Port 8000)
Model:          ✅ Loaded (Random Forest)
Data:           ✅ Ready (5,000+ records)
Predictions:    ✅ Working (91.5% prob)
Frontend:       ⏳ Ready to start

Latest Test:
  User: user_0
  Decision: AUTO APPROVE
  Probability: 91.5%
  Amount: $5,000
```

---

## 🚀 Quick Demo

```bash
# 1. Test health
curl http://localhost:8000/health

# 2. Get users
curl http://localhost:8000/users

# 3. Evaluate a loan
curl -X POST "http://localhost:8000/evaluate?user_id=user_0&requested_amount=5000"

# 4. Check API docs
open http://localhost:8000/docs
```

---

## 📱 Frontend Features

When you start the frontend, you'll see:
- 🏦 Beautiful gradient UI
- 👤 User selection dropdown
- 💰 Loan amount input
- ⚡ Real-time evaluation
- 📊 Financial profile display
- ✅ Approval/rejection decision
- 📈 Confidence scores
- 💡 Recommended amounts

---

## 🎯 Next Steps

1. **Start Frontend** (if npm available):
   ```bash
   cd frontend && npm install && npm start
   ```

2. **Test Different Users**: Try evaluating various users

3. **Adjust Amounts**: See how different loan amounts affect decisions

4. **Check API Docs**: Visit http://localhost:8000/docs

---

**Your ML Loan Eligibility Platform is LIVE! 🎉**
