# Quick Deployment Guide

## Fixed Issues
✓ Currency input now properly displays "UGX" prefix without overlap
✓ Frontend configured for production deployment
✓ API URL is environment-aware

## Free Hosting - Recommended Option

### Deploy to Render (Free - 5 minutes)

**Step 1: Push to GitHub**
```bash
git init
git add .
git commit -m "Uganda Loan Platform"
# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/loan-platform.git
git push -u origin main
```

**Step 2: Deploy Backend**
1. Go to https://render.com (sign up free)
2. Click "New +" → "Web Service"
3. Connect GitHub repository
4. Settings:
   - **Name**: `uganda-loan-api`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
   - **Free tier**: Yes

5. Click "Create Web Service"
6. **Copy the URL** (e.g., `https://uganda-loan-api.onrender.com`)

**Step 3: Deploy Frontend**
1. On Render, click "New +" → "Static Site"
2. Connect same repository
3. Settings:
   - **Name**: `uganda-loan-frontend`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `frontend/build`
   - **Environment Variable**:
     - Name: `REACT_APP_API_URL`
     - Value: `https://uganda-loan-api.onrender.com` (from step 2)

4. Click "Create Static Site"

**Done!** Your app is live at both URLs.

---

## Alternative: Vercel (Frontend) + Render (Backend)

**Backend (Render)**: Same as above

**Frontend (Vercel)**:
1. Go to https://vercel.com
2. Import GitHub repo
3. Settings:
   - **Framework**: Create React App
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
   - **Environment Variable**:
     - `REACT_APP_API_URL` = Your Render backend URL

4. Deploy

---

## Test Locally Before Deploying

**Backend:**
```bash
source venv/bin/activate
python api_server.py
# Visit http://localhost:8000/docs
```

**Frontend:**
```bash
cd frontend
npm start
# Visit http://localhost:3000
```

---

## What You Get (Free Tier)

**Render:**
- 750 hours/month free
- Automatic HTTPS
- Auto-deploy from GitHub
- Sleeps after 15 min inactivity (wakes in ~30s)

**Vercel:**
- Unlimited deployments
- Global CDN
- Automatic HTTPS
- Always on (no sleeping)

---

## Cost: $0/month

Both options are 100% free for your usage levels.

**Recommended**: Render for both (simpler setup)
**Best Performance**: Vercel (frontend) + Render (backend)

---

## Files Ready for Deployment

✓ `requirements.txt` - Python dependencies
✓ `.gitignore` - Excludes unnecessary files
✓ `frontend/src/App.js` - Environment-aware API calls
✓ `api_server.py` - Production-ready FastAPI server
✓ Uganda-specific data included

---

## Deploy Now

```bash
# 1. Initialize git
git init

# 2. Add all files
git add .

# 3. Commit
git commit -m "Uganda Financial Services Loan Platform"

# 4. Create GitHub repo and push
# (Follow GitHub instructions)

# 5. Deploy to Render
# (Follow steps above)
```

**Time to deploy**: ~5-10 minutes
**Monthly cost**: $0
