# Deployment Guide - Vercel Free Tier

## Quick Start (5 minutes)

### Step 1: Push to GitHub
✅ Already done! Your code is on branch: `claude/ml-learning-project-review-011CUQDg3GD8qcbpr1ocz5w9`

### Step 2: Deploy to Vercel

1. **Go to Vercel**
   - Visit [vercel.com](https://vercel.com)
   - Click "Sign Up" or "Login"
   - Choose "Continue with GitHub"

2. **Import Project**
   - Click "Add New..." → "Project"
   - Select your repository: `santiagoa58/machine-learning-intro`
   - Click "Import"

3. **Configure Project**
   - **Framework Preset**: Next.js (should auto-detect)
   - **Root Directory**: `web-app` ← **IMPORTANT! Change from default `.`**
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)
   - **Install Command**: `npm install` (default)

4. **Environment Variables** (Optional for now)
   - None required for initial deployment
   - Skip this section

5. **Deploy**
   - Click "Deploy"
   - Wait 2-3 minutes for build to complete
   - You'll get a URL like: `https://machine-learning-intro-xyz.vercel.app`

---

## Important Settings

### Root Directory Configuration
⚠️ **Critical**: The Next.js app is in the `web-app/` subdirectory, not the root.

Make sure to set:
- **Root Directory**: `web-app`

### Build Settings (Auto-configured)
- **Framework**: Next.js
- **Build Command**: `npm run build`
- **Output Directory**: `.next`
- **Install Command**: `npm install`
- **Node Version**: 18.x or higher (auto-detected)

---

## After Deployment

### Verify Deployment
1. Visit your Vercel URL
2. You should see the homepage with:
   - "Machine Learning Introduction" title
   - Philosophy section
   - Available algorithms
   - Documentation links (will work after next step)

### Get Your URL
- Vercel provides: `https://[project-name]-[hash].vercel.app`
- You can also set a custom domain (optional)

### Enable Playwright Access
Once deployed, Playwright will be able to access the live site at your Vercel URL.

---

## Troubleshooting

### Build Fails
**Error**: `Cannot find module 'next'`
- **Fix**: Ensure "Root Directory" is set to `web-app`

**Error**: Google Fonts timeout
- **Fix**: Already handled (using system fonts)

### 404 on Documentation Links
- **Expected**: Documentation pages not yet implemented
- Will be added in next tasks

### Deployment Branch
If you want to deploy from a different branch:
1. Go to Project Settings → Git
2. Change "Production Branch" to your desired branch
3. Or keep `main` and merge your changes there

---

## Vercel Free Tier Limits

✅ **What's Included (Free)**:
- Unlimited personal projects
- Unlimited deployments
- 100 GB bandwidth/month
- Serverless Functions (12,000 hrs/month)
- Edge Functions (500,000 invocations/month)
- Analytics (basic)
- Preview deployments for all branches
- SSL/HTTPS automatic
- Global CDN

🎉 **Perfect for this project!**

---

## Continuous Deployment

Once connected, Vercel automatically:
- ✅ Deploys every push to production branch
- ✅ Creates preview deployments for pull requests
- ✅ Runs build and tests
- ✅ Provides deployment URLs instantly

---

## Next Steps

After successful deployment:

1. ✅ **Share the URL** - Test the live site
2. ⏳ **Set up Playwright** - We'll configure automated UI testing
3. ⏳ **Add Documentation Pages** - Implement `/docs/*` routes
4. ⏳ **Integrate Compass Template** - Add sleek design (Sprint 1)

---

## Custom Domain (Optional)

Want a custom domain like `ml-learn.com`?

1. **Buy domain** (from Namecheap, Google Domains, etc.)
2. **In Vercel**:
   - Project Settings → Domains
   - Add your domain
   - Follow DNS configuration steps
3. **Done!** Vercel handles SSL automatically

---

## Monitoring & Analytics

### Built-in Analytics (Free)
- Go to your project dashboard
- Click "Analytics" tab
- See: Page views, visitors, top pages

### Deployment Logs
- Click any deployment
- View build logs, function logs
- Debug issues in real-time

---

## Quick Commands Reference

```bash
# Local development (in web-app/)
npm run dev          # Start dev server (localhost:3000)
npm run build        # Test production build
npm run start        # Run production build locally
npm run lint         # Check for errors

# After changing code
git add .
git commit -m "Your message"
git push             # Auto-deploys to Vercel!
```

---

## Status

- [x] Next.js app created
- [x] Code pushed to GitHub
- [ ] Deployed to Vercel (waiting for you!)
- [ ] Playwright configured
- [ ] Documentation pages added

---

**Ready to deploy?** Head to [vercel.com](https://vercel.com) and follow Step 2 above!

Once deployed, share the URL and we'll set up Playwright testing next.
