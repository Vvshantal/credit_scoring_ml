# Deploy to Vercel + Render - Step by Step

## Prerequisites
✓ Git initialized and committed
✓ Code ready for deployment
✓ GitHub account
✓ Vercel account (free)
✓ Render account (free)

---

## Step 1: Push to GitHub

### 1.1 Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `uganda-loan-platform` (or your choice)
3. Description: "ML-powered loan eligibility platform for Uganda"
4. **Important**: Make it Public (required for free Vercel deployment)
5. Do NOT initialize with README (we already have one)
6. Click "Create repository"

### 1.2 Push Your Code
```bash
# Copy these commands from GitHub (they'll appear after creating the repo)
git remote add origin https://github.com/YOUR_USERNAME/uganda-loan-platform.git
git branch -M main
git push -u origin main
```

**Verify**: Visit your GitHub repo URL to confirm all files are uploaded.

---

## Step 2: Deploy Backend to Render

### 2.1 Sign Up for Render
1. Go to https://render.com
2. Sign up with GitHub (recommended) or email
3. Authorize Render to access your GitHub repos

### 2.2 Create Web Service
1. Click **"New +"** button (top right)
2. Select **"Web Service"**
3. Choose your GitHub repository: `uganda-loan-platform`

### 2.3 Configure Backend Service
Fill in these settings:

**Basic Info:**
- **Name**: `uganda-loan-api`
- **Region**: Oregon (US West)
- **Branch**: `main`
- **Root Directory**: Leave empty
- **Runtime**: Python 3

**Build & Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

**Plan:**
- Select **"Free"** plan (0 USD/month)

**Advanced (Optional):**
- **Health Check Path**: `/health`
- **Auto-Deploy**: Yes (enabled by default)

### 2.4 Deploy
1. Click **"Create Web Service"**
2. Wait 2-3 minutes for deployment (watch the logs)
3. Once deployed, you'll see: ✓ **Live** status

### 2.5 Copy Backend URL
**IMPORTANT**: Copy your backend URL. It will look like:
```
https://uganda-loan-api.onrender.com
```

**Test your API**: Visit `https://uganda-loan-api.onrender.com/docs` to see the API documentation.

---

## Step 3: Deploy Frontend to Vercel

### 3.1 Sign Up for Vercel
1. Go to https://vercel.com
2. Sign up with GitHub (recommended)
3. Authorize Vercel to access your repos

### 3.2 Import Project
1. Click **"Add New..."** → **"Project"**
2. Import your GitHub repository: `uganda-loan-platform`
3. Click **"Import"**

### 3.3 Configure Frontend
**Framework Preset**: Vercel should auto-detect Create React App. If not:
- Framework Preset: `Create React App`
- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `build`

**Environment Variables:**
Click **"Environment Variables"** section and add:

| Name | Value |
|------|-------|
| `REACT_APP_API_URL` | `https://uganda-loan-api.onrender.com` |

⚠️ **Replace with YOUR Render URL from Step 2.5**

### 3.4 Deploy
1. Click **"Deploy"**
2. Wait 1-2 minutes for build and deployment
3. Once complete, you'll see: 🎉 **Congratulations**

### 3.5 Visit Your Live Site
Vercel will give you a URL like:
```
https://uganda-loan-platform.vercel.app
```

Click the URL to see your live application!

---

## Step 4: Test Your Deployment

### 4.1 Test Backend (API)
```bash
# Replace with your Render URL
curl https://uganda-loan-api.onrender.com/health
curl https://uganda-loan-api.onrender.com/users

# Test evaluation
curl -X POST "https://uganda-loan-api.onrender.com/evaluate?user_id=user_0&requested_amount=5000000"
```

### 4.2 Test Frontend
1. Visit your Vercel URL: `https://uganda-loan-platform.vercel.app`
2. You should see the Uganda Financial Services interface
3. Select a user (e.g., user_6)
4. Enter loan amount: 8,000,000
5. Click "Submit Application"
6. Wait for ML prediction results

**Expected**: You should see an instant decision with:
- Approval status (APPROVED/REJECTED/REVIEW)
- Confidence level
- Recommended amount
- Financial profile

---

## Step 5: Set Up Auto-Deploy (Optional)

Both Vercel and Render support automatic deployments:

**How it works:**
- Push changes to GitHub `main` branch
- Both services automatically redeploy
- Takes 2-3 minutes per deployment

**Test auto-deploy:**
```bash
# Make a small change
echo "# Version 1.0.1" >> README.md

# Commit and push
git add README.md
git commit -m "Update version"
git push

# Watch deployments in Vercel and Render dashboards
```

---

## Step 6: Configure Custom Domain (Optional)

### Vercel Custom Domain
1. Go to your project in Vercel
2. Settings → Domains
3. Add your domain
4. Follow DNS configuration instructions

### Render Custom Domain
1. Go to your web service in Render
2. Settings → Custom Domains
3. Add your domain
4. Configure DNS (add CNAME record)

---

## Deployment URLs

After deployment, save these URLs:

**Backend API:**
```
https://uganda-loan-api.onrender.com
https://uganda-loan-api.onrender.com/docs  (API Documentation)
```

**Frontend:**
```
https://uganda-loan-platform.vercel.app
```

---

## Troubleshooting

### Backend Issues

**Problem**: "Application failed to start"
- Check logs in Render dashboard
- Verify `requirements.txt` has all dependencies
- Ensure Python version is 3.11

**Problem**: API returns 500 errors
- Check model files are committed (`models_trained/`)
- Verify data files exist (`data/raw/`)
- Check logs for specific error messages

**Problem**: Slow cold starts
- Free tier sleeps after 15 min of inactivity
- First request after sleep takes ~30 seconds
- Subsequent requests are instant

### Frontend Issues

**Problem**: "Failed to fetch" errors
- Verify `REACT_APP_API_URL` environment variable is set correctly in Vercel
- Check backend is deployed and responding
- Test backend URL directly in browser

**Problem**: CORS errors
- Backend already has CORS enabled for all origins
- If issues persist, check Render logs

**Problem**: Blank page
- Check browser console for errors
- Verify build completed successfully in Vercel
- Check build logs in Vercel dashboard

---

## Monitoring

### Render Dashboard
- View logs: Render dashboard → Logs
- Check health: Status indicator
- Resource usage: Free tier usage stats

### Vercel Dashboard
- View deployments: Project → Deployments
- Analytics: Project → Analytics (free basic analytics)
- Logs: Click on any deployment → Build logs

---

## Costs

**Total Monthly Cost**: $0

**Render Free Tier:**
- 750 hours/month compute time
- Automatic HTTPS
- Auto-deploy from Git
- Services sleep after 15 min inactivity

**Vercel Free Tier:**
- Unlimited deployments
- 100 GB bandwidth/month
- Automatic HTTPS
- Global CDN

---

## Next Steps

1. **Share Your Links**: Send your Vercel URL to stakeholders
2. **Monitor Usage**: Check dashboards regularly
3. **Add Features**: Push changes to GitHub for auto-deploy
4. **Scale Up**: Upgrade to paid tiers when needed

---

## Quick Command Reference

```bash
# View local git status
git status

# Make changes and deploy
git add .
git commit -m "Your message"
git push

# View remote URL
git remote -v

# Check recent commits
git log --oneline -5
```

---

## Support

**Render Issues**: https://render.com/docs
**Vercel Issues**: https://vercel.com/docs

**Platform Status:**
- Render Status: https://status.render.com
- Vercel Status: https://www.vercel-status.com

---

## Success Checklist

After deployment, verify:

- [ ] Backend API is live and responding at `/health`
- [ ] API documentation accessible at `/docs`
- [ ] Frontend loads without errors
- [ ] Can select users from dropdown
- [ ] Can submit loan applications
- [ ] ML predictions return successfully
- [ ] Currency displays as UGX
- [ ] Professional banking theme visible
- [ ] Auto-deploy works on git push

---

**Congratulations!** 🎉

Your Uganda Financial Services Loan Eligibility Platform is now live and accessible worldwide at:
- **Frontend**: https://uganda-loan-platform.vercel.app
- **API**: https://uganda-loan-api.onrender.com

**Total deployment time**: ~10 minutes
**Total cost**: $0/month
