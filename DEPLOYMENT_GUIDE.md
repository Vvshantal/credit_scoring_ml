# Free Hosting Options for ML Loan Eligibility Platform

## Recommended Free Hosting Solutions

### Option 1: Render (Recommended - Easiest)

**Best for**: Full-stack deployment with minimal configuration

**Free Tier Includes:**
- 750 hours/month free (enough for 1 web service)
- Auto-deploy from Git
- HTTPS/SSL certificates
- Custom domains

**Steps:**

1. **Prepare Backend for Deployment**
   - Create `requirements.txt`:
     ```bash
     cd /Users/rtv-lpt-237/dev/personal-projects/credit_score
     pip freeze > requirements.txt
     ```

2. **Create `render.yaml`** (in project root):
   ```yaml
   services:
     - type: web
       name: loan-api
       env: python
       buildCommand: "pip install -r requirements.txt"
       startCommand: "uvicorn api_server:app --host 0.0.0.0 --port $PORT"
       envVars:
         - key: PYTHON_VERSION
           value: 3.11.0

     - type: web
       name: loan-frontend
       env: static
       buildCommand: "cd frontend && npm install && npm run build"
       staticPublishPath: frontend/build
   ```

3. **Deploy:**
   - Push code to GitHub
   - Connect to Render (https://render.com)
   - Select your repository
   - Render will auto-deploy both services

**Pros:**
- Zero configuration needed
- Automatic HTTPS
- Good for both API and frontend
- Free tier is generous

**Cons:**
- Services sleep after 15 min of inactivity (takes ~30s to wake up)

---

### Option 2: Railway (Good Alternative)

**Free Tier:**
- $5 credit/month (renewable)
- Auto-deploy from Git
- Built-in databases

**Steps:**

1. **Push to GitHub/GitLab**
2. **Connect Railway** (https://railway.app)
3. **Add Environment Variables:**
   ```
   PORT=8000
   ```
4. **Deploy both services**

**Pros:**
- Faster cold starts than Render
- Better for ML models
- Good dashboard

**Cons:**
- $5/month credit runs out with heavy usage

---

### Option 3: Vercel (Frontend) + Render/Railway (Backend)

**Best for**: Separate frontend and backend deployment

**Frontend on Vercel:**
1. Push frontend to GitHub
2. Connect to Vercel (https://vercel.com)
3. Set build settings:
   - Framework: Create React App
   - Build command: `npm run build`
   - Output directory: `build`

**Backend on Render/Railway** (see above)

**Update Frontend API URL:**
In `frontend/src/App.js`, change:
```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Then use:
fetch(`${API_URL}/users`)
```

**Pros:**
- Vercel has excellent React hosting
- Unlimited bandwidth
- Global CDN

---

### Option 4: Hugging Face Spaces (Great for ML)

**Best for**: Showcasing ML models

**Free Tier:**
- 2 CPU cores
- 16 GB RAM
- Perfect for ML models

**Steps:**

1. **Create Space** at https://huggingface.co/spaces
2. **Choose Gradio or Streamlit** (or custom Docker)
3. **Upload your model and code**
4. **Create `app.py`** using Gradio:

```python
import gradio as gr
import joblib
import pandas as pd

model = joblib.load("models_trained/best_model.joblib")
feature_names = joblib.load("models_trained/feature_names.joblib")

def predict_loan(user_id, requested_amount):
    # Your prediction logic
    # Return approval decision
    pass

iface = gr.Interface(
    fn=predict_loan,
    inputs=[
        gr.Textbox(label="User ID"),
        gr.Number(label="Requested Amount (UGX)")
    ],
    outputs="text",
    title="Uganda Loan Eligibility Assessment"
)

iface.launch()
```

**Pros:**
- Perfect for ML models
- Fast inference
- Great for portfolio

---

### Option 5: PythonAnywhere (Backend) + Netlify (Frontend)

**Backend on PythonAnywhere:**
- Free tier: 1 web app
- Good for Flask/FastAPI
- URL: `yourusername.pythonanywhere.com`

**Frontend on Netlify:**
- Drag-and-drop deployment
- Or connect GitHub
- Automatic HTTPS

---

## Recommended Setup for Your Project

### Best Free Setup:

```
Frontend: Vercel (https://vercel.com)
Backend: Render (https://render.com)
```

**Why?**
- Both have generous free tiers
- Excellent performance
- Easy deployment
- Automatic HTTPS
- Good uptime

---

## Step-by-Step Deployment (Vercel + Render)

### Step 1: Prepare Your Code

1. **Create `.gitignore`** (if not exists):
```
venv/
__pycache__/
*.pyc
.env
node_modules/
.DS_Store
```

2. **Initialize Git** (if not done):
```bash
cd /Users/rtv-lpt-237/dev/personal-projects/credit_score
git init
git add .
git commit -m "Initial commit"
```

3. **Create GitHub Repository**:
```bash
# On GitHub, create new repo: credit-score-platform
git remote add origin https://github.com/YOUR_USERNAME/credit-score-platform.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy Backend to Render

1. Go to https://render.com and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `uganda-loan-api`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free

5. Click "Create Web Service"

6. Copy the URL (e.g., `https://uganda-loan-api.onrender.com`)

### Step 3: Deploy Frontend to Vercel

1. Go to https://vercel.com and sign up
2. Click "Add New Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Create React App
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`

5. Add Environment Variable:
   - **Name**: `REACT_APP_API_URL`
   - **Value**: `https://uganda-loan-api.onrender.com`

6. Click "Deploy"

### Step 4: Update Frontend Code

Before deploying, update `frontend/src/App.js`:

```javascript
// At the top of the file
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Update all fetch calls:
fetch(`${API_URL}/users`)
fetch(`${API_URL}/evaluate?user_id=${formData.user_id}&requested_amount=${formData.requested_amount}`)
```

---

## Alternative: All-in-One Docker Deployment

### Deploy to Fly.io (Free Tier)

1. **Install Fly CLI**:
```bash
brew install flyctl
```

2. **Create `Dockerfile`**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. **Create `fly.toml`**:
```toml
app = "uganda-loan-platform"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
```

4. **Deploy**:
```bash
fly launch
fly deploy
```

---

## Cost Comparison

| Platform | Free Tier | Best For | Limitations |
|----------|-----------|----------|-------------|
| **Render** | 750 hrs/month | Full-stack apps | Sleeps after 15 min |
| **Railway** | $5 credit/month | Quick deploys | Credit expires |
| **Vercel** | Unlimited | React/Next.js | No backend |
| **Netlify** | 100 GB bandwidth | Static sites | No backend |
| **Fly.io** | 3 VMs free | Docker apps | 256 MB RAM |
| **HuggingFace** | 16 GB RAM | ML models | Public only |
| **PythonAnywhere** | 1 web app | Python APIs | Limited CPU |

---

## Recommended: Render Setup

**Quickest deployment:**

1. **Create `requirements.txt`**:
```bash
pip freeze > requirements.txt
```

2. **Push to GitHub**

3. **Deploy to Render** (both services):
   - Backend: Python web service
   - Frontend: Static site from `frontend/build`

4. **Your app will be live at**:
   - API: `https://your-app.onrender.com`
   - Frontend: `https://your-app-frontend.onrender.com`

**Total time**: ~10 minutes
**Cost**: $0

---

## Ready to Deploy?

Run these commands to prepare:

```bash
# 1. Create requirements.txt
cd /Users/rtv-lpt-237/dev/personal-projects/credit_score
source venv/bin/activate
pip freeze > requirements.txt

# 2. Build frontend
cd frontend
npm run build

# 3. Test production build locally
npx serve -s build -l 3000

# 4. Initialize git and push
cd ..
git init
git add .
git commit -m "Uganda Loan Eligibility Platform"
# Create repo on GitHub, then:
git remote add origin YOUR_GITHUB_URL
git push -u origin main
```

Then follow Render or Vercel deployment steps above.
