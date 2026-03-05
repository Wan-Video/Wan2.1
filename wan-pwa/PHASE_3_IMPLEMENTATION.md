# Phase 3 Implementation - Backend Integration & Polish

## Overview

Phase 3 closes the critical integration gaps between the frontend, backend, database, and Replicate API. This document details all implemented changes and how to test them.

## ✅ Completed Features

### 1. Database Integration

**What Changed:**
- Generation records now created BEFORE calling Replicate
- Credits deducted atomically using database function
- Job IDs properly tracked for status polling
- Automatic refunds on failures

**Files Modified:**
- `packages/db/migrations/002_credit_system.sql` - New migration with credit functions
- `apps/api/routes/generation.py` - Complete rewrite of generation flow

**How It Works:**

```python
# Flow for Text-to-Video generation:
1. Check user has sufficient credits
2. Create generation record (status: "queued")
3. Start Replicate job
4. Update record with job_id (status: "processing")
5. Deduct credits using database function
6. Return generation_id to client
7. (Webhook) Update record when complete
```

**Testing:**
```bash
# 1. Run migration in Supabase SQL Editor
# Copy contents of packages/db/migrations/002_credit_system.sql

# 2. Test credit deduction
curl -X POST http://localhost:8000/api/generation/text-to-video \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Test video",
    "model": "t2v-14B",
    "resolution": "720p"
  }'

# 3. Check database
# generations table should have new record
# credits should be deducted
# credit_transactions should have deduction entry
```

### 2. Webhook Handler

**What Changed:**
- Created `/api/webhooks/replicate` endpoint
- HMAC signature verification
- Automatic status updates from Replicate
- Refund credits on failures

**Files Created:**
- `apps/api/routes/webhooks.py` - Webhook handler

**How It Works:**
```python
# When Replicate completes a prediction:
1. Replicate sends POST to /api/webhooks/replicate
2. Verify HMAC signature
3. Find generation by job_id
4. Update status, progress, video_url
5. If failed, trigger refund
```

**Setup:**
```bash
# 1. Deploy API
modal deploy apps/api/main.py

# 2. Get webhook URL
# https://your-app--modal.run/api/webhooks/replicate

# 3. Register webhook with Replicate
curl -X POST https://api.replicate.com/v1/webhooks \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app--modal.run/api/webhooks/replicate",
    "events": ["predictions.completed", "predictions.failed"],
    "secret": "your-webhook-secret"
  }'

# 4. Add secret to environment
# In Modal: modal secret create wan-secrets REPLICATE_WEBHOOK_SECRET=wh_sec_xxxxx
# In .env: REPLICATE_WEBHOOK_SECRET=wh_sec_xxxxx
```

**Testing:**
```bash
# Test webhook endpoint
curl -X POST http://localhost:8000/api/webhooks/replicate \
  -H "Content-Type: application/json" \
  -H "Webhook-Signature: test-signature" \
  -d '{
    "id": "test-job-id",
    "status": "succeeded",
    "output": "https://example.com/video.mp4"
  }'
```

### 3. Credit System Functions

**What Changed:**
- Added `deduct_credits()` - Atomic credit deduction with transaction logging
- Added `add_credits()` - Add credits with transaction logging
- Added `refund_credits()` - Automatic refunds for failed generations
- Added `credit_transactions` table for audit trail

**Database Functions:**
```sql
-- Deduct credits (called by API)
SELECT deduct_credits(
  'user-uuid',  -- p_user_id
  20,           -- p_amount
  'gen-uuid'    -- p_gen_id (optional)
);

-- Add credits (for purchases)
SELECT add_credits(
  'user-uuid',     -- p_user_id
  100,             -- p_amount
  'purchase',      -- p_type
  'Bought 100 credits'  -- p_description
);

-- Refund credits (automatic on failure)
SELECT refund_credits('gen-uuid');
```

**Testing:**
```sql
-- Test deduction
SELECT deduct_credits('test-user-id', 10, NULL);

-- Verify transaction logged
SELECT * FROM credit_transactions WHERE user_id = 'test-user-id';

-- Test refund
SELECT refund_credits('test-generation-id');
```

### 4. Frontend Error Handling

**What Changed:**
- Added `sonner` for toast notifications
- Created `Providers` component with Toaster
- Added validation schemas with Zod
- Created `useCredits` hook for credit management

**Files Created:**
- `apps/web/src/components/providers.tsx` - Toast provider
- `apps/web/src/lib/validation/generation.ts` - Zod schemas
- `apps/web/src/lib/hooks/use-credits.ts` - Credit management hook

**Usage Example:**
```tsx
import { toast } from "sonner"
import { useCredits } from "@/lib/hooks/use-credits"

function GenerationForm() {
  const { credits, optimisticDeduct } = useCredits(userId)

  const handleGenerate = async () => {
    try {
      const response = await fetch('/api/generation/text-to-video', {
        method: 'POST',
        body: JSON.stringify(formData)
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail)
      }

      // Optimistically update credits
      optimisticDeduct(cost)

      toast.success('Generation started!', {
        description: 'Your video is being generated. Check History for progress.'
      })

    } catch (error) {
      toast.error('Generation failed', {
        description: error.message,
        action: {
          label: 'Retry',
          onClick: () => handleGenerate()
        }
      })
    }
  }
}
```

### 5. Image Upload Component

**What Changed:**
- Created drag-and-drop image upload
- Client-side validation (file type, size)
- Preview functionality
- Integration ready for I2V

**Files Created:**
- `apps/web/src/components/generation/image-upload.tsx`

**Usage:**
```tsx
import { ImageUpload } from "@/components/generation/image-upload"

function I2VForm() {
  const [inputImage, setInputImage] = useState<File | null>(null)

  return (
    <ImageUpload
      onImageSelect={(file) => setInputImage(file)}
      onImageRemove={() => setInputImage(null)}
      maxSizeMB={10}
    />
  )
}
```

**Testing:**
1. Drag image file onto upload area
2. Verify preview shows
3. Try uploading non-image file (should show error toast)
4. Try uploading 15MB file (should show size error)

### 6. Form Validation

**What Changed:**
- Added Zod schemas for T2V and I2V
- Validation for prompt length, model selection, resolution
- Credit cost calculator

**Schemas:**
```typescript
import { textToVideoSchema, calculateCreditCost } from '@/lib/validation/generation'

// Validate form data
const result = textToVideoSchema.safeParse(formData)
if (!result.success) {
  // Show validation errors
  console.log(result.error.issues)
}

// Calculate cost
const cost = calculateCreditCost('t2v-14B', '720p')  // Returns 20
```

### 7. Settings Page

**What Changed:**
- Created basic settings page structure
- Placeholders for Profile, Billing, API Keys

**Files Created:**
- `apps/web/src/app/dashboard/settings/page.tsx`

**TODO:**
- Implement profile editing
- Add billing/payment integration
- Create API key management

## 🔧 Environment Variables

### Backend (New)
```bash
# Add to apps/api/.env
REPLICATE_WEBHOOK_SECRET=wh_sec_xxxxxxxxxxxxx
```

### Frontend (No Changes)
```bash
# Existing .env.local variables still apply
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🧪 Testing Checklist

### Backend Integration
- [ ] Create generation → Record appears in database
- [ ] Credits deduct correctly (20 for 720p, 10 for 480p)
- [ ] job_id saved to generation record
- [ ] Status updates via polling work
- [ ] Webhook updates status automatically
- [ ] Video URL saved on completion
- [ ] Failed generations trigger refund
- [ ] Credit transactions logged correctly

### Frontend
- [ ] Toast notifications show on success/error
- [ ] Form validation prevents invalid submissions
- [ ] Credit balance displays correctly
- [ ] Low credit warning shows when < 5 credits
- [ ] Image upload accepts valid files
- [ ] Image upload rejects invalid files
- [ ] Settings page loads without errors

### End-to-End
- [ ] Sign up → Receive 100 free credits
- [ ] Generate video → Credits deduct
- [ ] Poll status → Updates show progress
- [ ] Video completes → URL available for download
- [ ] Try with 0 credits → Prevented with error message

## 📊 Database Changes

### New Table: `credit_transactions`
```sql
CREATE TABLE credit_transactions (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  amount INTEGER NOT NULL,
  type TEXT NOT NULL,  -- 'deduction', 'purchase', 'refund'
  generation_id UUID REFERENCES generations(id),
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### New Columns: `generations`
- `job_id TEXT` - Replicate prediction ID
- `progress INTEGER` - Progress percentage (0-100)
- `error_message TEXT` - Error details if failed

### New Functions
- `deduct_credits(user_id, amount, gen_id)` - Atomic deduction
- `add_credits(user_id, amount, type, description)` - Add credits
- `refund_credits(gen_id)` - Refund failed generation

## 🚀 Deployment Steps

### 1. Database Migration
```bash
# In Supabase SQL Editor:
# 1. Go to SQL Editor
# 2. Create new query
# 3. Paste contents of packages/db/migrations/002_credit_system.sql
# 4. Run query
# 5. Verify tables and functions created
```

### 2. Backend Deployment
```bash
cd apps/api

# Update environment variables
# Add REPLICATE_WEBHOOK_SECRET to Modal secrets or .env

# Deploy
modal deploy main.py

# Note the webhook URL
# https://your-app--modal.run
```

### 3. Register Webhook
```bash
# Set environment variables
export REPLICATE_API_TOKEN="your-token"
export WEBHOOK_SECRET="wh_sec_$(openssl rand -hex 32)"

# Register webhook
curl -X POST https://api.replicate.com/v1/webhooks \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"https://your-app--modal.run/api/webhooks/replicate\",
    \"events\": [\"predictions.completed\", \"predictions.failed\"],
    \"secret\": \"$WEBHOOK_SECRET\"
  }"

# Save webhook secret to environment
# Add REPLICATE_WEBHOOK_SECRET=$WEBHOOK_SECRET to your deployment
```

### 4. Frontend Deployment
```bash
cd apps/web

# No new variables needed
# Deploy to Vercel
vercel deploy --prod
```

## 🐛 Known Issues & Limitations

### 1. Polling Fallback
**Issue:** If webhook fails, polling never stops
**Solution:** Add max polling attempts (implement in Phase 4)

### 2. Race Condition
**Issue:** Multiple concurrent requests could bypass credit check
**Solution:** Database function ensures atomic operation, but add rate limiting

### 3. No Retry Logic
**Issue:** Failed generations can't be retried
**Solution:** Add retry button in history (implement in Phase 4)

### 4. Storage Costs
**Issue:** No cleanup of old videos/images
**Solution:** Implement lifecycle policies (implement in Phase 4)

### 5. No Cancel Button
**Issue:** Users can't stop in-progress generations
**Solution:** Add cancel endpoint (implement in Phase 4)

## 📈 Metrics to Monitor

### Backend
- Generation success rate (target: > 95%)
- Average completion time (target: < 5 minutes)
- Webhook delivery rate (target: > 99%)
- Credit deduction accuracy (target: 100%)

### Frontend
- Form validation error rate
- Toast notification engagement
- Image upload success rate
- Credit check effectiveness

## 🔜 Next Steps (Phase 4)

### High Priority
1. **Payment Integration** - Stripe for credit purchases
2. **Retry Logic** - Retry failed generations
3. **Cancel Function** - Stop in-progress generations
4. **Video Player** - In-app preview instead of download-only

### Medium Priority
5. **Batch Operations** - Multi-delete, bulk download
6. **Admin Panel** - Usage monitoring, user management
7. **Rate Limiting** - Prevent API abuse
8. **Caching** - Redis for status queries

### Low Priority
9. **Analytics** - Track generation patterns
10. **Social Features** - Share videos, favorites
11. **Advanced Editing** - VACE integration
12. **API for Developers** - REST + SDKs

## 📚 Additional Resources

### Documentation
- [Replicate Webhooks](https://replicate.com/docs/webhooks)
- [Supabase RPC Functions](https://supabase.com/docs/guides/database/functions)
- [Sonner Toast Library](https://sonner.emilkowal.ski/)
- [Zod Validation](https://zod.dev/)

### Code Examples
- Database functions: `packages/db/migrations/002_credit_system.sql`
- Webhook handler: `apps/api/routes/webhooks.py`
- Credit hook: `apps/web/src/lib/hooks/use-credits.ts`
- Validation: `apps/web/src/lib/validation/generation.ts`

## 🤝 Support

For issues or questions:
1. Check this documentation
2. Review SETUP.md and DEPLOYMENT.md
3. Check database logs in Supabase
4. Review API logs in Modal
5. Open GitHub issue with logs and reproduction steps

---

**Phase 3 Status:** ✅ Complete
**Ready for Testing:** Yes
**Ready for Production:** Pending testing and webhook registration
