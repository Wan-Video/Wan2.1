# Deployment Guide

## Frontend (Vercel)

### Prerequisites
- Vercel account
- GitHub repository

### Steps

1. **Push to GitHub**
   ```bash
   git push origin main
   ```

2. **Import to Vercel**
   - Go to https://vercel.com
   - Click "New Project"
   - Import your repository
   - Select `apps/web` as the root directory

3. **Configure Environment Variables**

   Add these in Vercel dashboard:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGci...
   SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
   NEXT_PUBLIC_API_URL=https://your-api-url.com
   NEXT_PUBLIC_APP_URL=https://your-app.vercel.app
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait for build to complete
   - Visit your deployment URL

### Custom Domain (Optional)

1. Go to Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed
4. Update `NEXT_PUBLIC_APP_URL` to your domain

---

## Backend (Modal)

Modal provides serverless Python deployment with GPU support.

### Prerequisites
- Modal account
- Modal CLI installed: `pip install modal`

### Steps

1. **Install Modal CLI**
   ```bash
   pip install modal
   modal setup
   ```

2. **Create Modal App**

   Create `apps/api/modal_deploy.py`:
   ```python
   import modal

   stub = modal.Stub("wan-pwa-api")

   image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

   @stub.function(image=image)
   @modal.asgi_app()
   def fastapi_app():
       from main import app
       return app
   ```

3. **Set Secrets**
   ```bash
   modal secret create wan-secrets \
     SUPABASE_URL=https://xxxxx.supabase.co \
     SUPABASE_SERVICE_ROLE_KEY=eyJhbGci... \
     REPLICATE_API_TOKEN=r8_xxxxx
   ```

4. **Deploy**
   ```bash
   cd apps/api
   modal deploy modal_deploy.py
   ```

5. **Get URL**
   - Modal will provide a URL like `https://your-app--modal.com`
   - Update frontend `NEXT_PUBLIC_API_URL` to this URL

---

## Backend (Railway - Alternative)

Railway is simpler but doesn't have GPU support (uses Replicate API instead).

### Steps

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Create Project**
   ```bash
   cd apps/api
   railway init
   ```

3. **Add Environment Variables**
   ```bash
   railway variables set SUPABASE_URL=https://xxxxx.supabase.co
   railway variables set SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
   railway variables set REPLICATE_API_TOKEN=r8_xxxxx
   ```

4. **Deploy**
   ```bash
   railway up
   ```

5. **Get URL**
   ```bash
   railway domain
   ```

---

## Database (Supabase)

Database is already set up in Supabase - no deployment needed!

### Production Checklist

- [ ] Run migrations in production project
- [ ] Enable RLS (Row Level Security)
- [ ] Configure Auth providers (email, Google, GitHub)
- [ ] Set up storage buckets with proper policies
- [ ] Enable database backups
- [ ] Set up monitoring and alerts

---

## Redis (Upstash) - Optional

For background jobs and caching.

### Steps

1. **Create Upstash Account**
   - Go to https://upstash.com
   - Create a Redis database

2. **Get Connection String**
   - Copy the connection URL

3. **Update Environment**
   ```
   REDIS_URL=redis://...
   CELERY_BROKER_URL=redis://...
   ```

---

## CI/CD with GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          working-directory: apps/web

  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Deploy to Modal
        run: |
          pip install modal
          modal token set --token-id ${{ secrets.MODAL_TOKEN_ID }} --token-secret ${{ secrets.MODAL_TOKEN_SECRET }}
          cd apps/api && modal deploy modal_deploy.py
```

---

## Environment Variables Checklist

### Frontend (Vercel)
- [ ] `NEXT_PUBLIC_SUPABASE_URL`
- [ ] `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- [ ] `SUPABASE_SERVICE_ROLE_KEY`
- [ ] `NEXT_PUBLIC_API_URL`
- [ ] `NEXT_PUBLIC_APP_URL`

### Backend (Modal/Railway)
- [ ] `SUPABASE_URL`
- [ ] `SUPABASE_SERVICE_ROLE_KEY`
- [ ] `REPLICATE_API_TOKEN`
- [ ] `ALLOWED_ORIGINS`
- [ ] `REDIS_URL` (optional)

---

## Post-Deployment

1. **Test the Application**
   - Sign up a test user
   - Generate a test video
   - Check credit deduction
   - Verify video download

2. **Monitor**
   - Set up Sentry for error tracking
   - Monitor Vercel analytics
   - Check Supabase usage

3. **Scale**
   - Adjust Vercel plan if needed
   - Scale Modal functions based on usage
   - Upgrade Supabase plan for production

---

## Troubleshooting

### Build Failures
- Check environment variables are set
- Verify all dependencies in package.json
- Check build logs for specific errors

### API Errors
- Verify Supabase connection
- Check Replicate API token
- Review CORS settings

### Database Issues
- Ensure migrations have run
- Check RLS policies
- Verify user permissions
