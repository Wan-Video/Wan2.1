# Setup Guide

## Quick Start (5 Minutes)

### 1. Prerequisites

- Node.js 18+ installed
- Python 3.10+ installed
- Supabase account (free tier)
- Replicate account ($10 free credit)

### 2. Clone and Install

```bash
git clone <your-repo-url>
cd wan-pwa
npm install
```

### 3. Get Credentials

#### Supabase Setup (3 minutes)

1. Go to https://supabase.com
2. Create new project: "wan-pwa"
3. Wait ~2 minutes for provisioning
4. Go to Settings → API
5. Copy these 4 values:
   - Project URL → `NEXT_PUBLIC_SUPABASE_URL`
   - anon public → `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - service_role → `SUPABASE_SERVICE_ROLE_KEY`
   - JWT Secret → `SUPABASE_JWT_SECRET`

#### Replicate Setup (2 minutes)

1. Go to https://replicate.com
2. Sign up with GitHub
3. Go to https://replicate.com/account/api-tokens
4. Create token → Copy → `REPLICATE_API_TOKEN`

### 4. Configure Environment

#### Frontend (.env.local)

```bash
cd apps/web
cp .env.example .env.local
```

Edit `.env.local`:
```bash
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGci...
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

#### Backend (.env)

```bash
cd ../api
cp .env.example .env
```

Edit `.env`:
```bash
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...
REPLICATE_API_TOKEN=r8_xxxxx
ALLOWED_ORIGINS=http://localhost:3000
```

### 5. Database Setup

```bash
# From project root
cd packages/db

# Run migration in Supabase dashboard:
# 1. Go to SQL Editor in Supabase
# 2. Create new query
# 3. Copy contents of migrations/001_initial_schema.sql
# 4. Run query
```

### 6. Start Development

```bash
# Terminal 1: Frontend
cd apps/web
npm run dev
# → http://localhost:3000

# Terminal 2: Backend
cd apps/api
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000
```

### 7. Test the App

1. Open http://localhost:3000
2. Click "Get Started"
3. Sign up with email
4. Browse prompt templates
5. Generate your first video!

---

## Troubleshooting

### "Module not found" errors

```bash
# Clear caches and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Supabase connection errors

Check:
- URLs don't have trailing slashes
- Keys are complete (very long strings)
- Project is fully provisioned (not still "Setting up")

### API not starting

```bash
# Check Python version
python --version  # Should be 3.10+

# Try with full path
python3 -m uvicorn main:app --reload
```

### Database migration fails

- Make sure you're using the SQL Editor in Supabase dashboard
- Check that UUID extension is enabled
- Verify you're in the correct project

---

## Next Steps

- [ ] Customize prompt templates in `apps/web/src/lib/prompts/templates.ts`
- [ ] Add your logo in `apps/web/public/icons/`
- [ ] Setup GitHub Actions for CI/CD
- [ ] Deploy to production (see DEPLOYMENT.md)

---

## Project Structure

```
wan-pwa/
├── apps/
│   ├── web/              # Next.js frontend
│   │   ├── src/
│   │   │   ├── app/      # Pages & layouts
│   │   │   ├── components/ # React components
│   │   │   └── lib/      # Utilities & hooks
│   │   └── public/       # Static assets
│   │
│   └── api/              # FastAPI backend
│       ├── routes/       # API endpoints
│       ├── core/         # Business logic
│       └── models/       # Pydantic models
│
├── packages/
│   └── db/               # Database schema
│       └── migrations/   # SQL migrations
│
└── turbo.json            # Monorepo config
```

---

## Need Help?

- Check [README.md](./README.md) for features overview
- See [DEPLOYMENT.md](./DEPLOYMENT.md) for production setup
- Open an issue on GitHub
