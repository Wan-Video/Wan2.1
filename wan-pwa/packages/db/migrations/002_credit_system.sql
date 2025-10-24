-- Add credit transaction log (for audit trail)
CREATE TABLE IF NOT EXISTS public.credit_transactions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  amount INTEGER NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('deduction', 'purchase', 'refund')),
  generation_id UUID REFERENCES public.generations(id),
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for user queries
CREATE INDEX IF NOT EXISTS idx_credit_transactions_user ON public.credit_transactions(user_id, created_at DESC);

-- Enable RLS
ALTER TABLE public.credit_transactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own transactions"
  ON public.credit_transactions FOR SELECT
  USING (auth.uid() = user_id);

-- Update deduct_credits function to log transaction
CREATE OR REPLACE FUNCTION deduct_credits(p_user_id UUID, p_amount INTEGER, p_gen_id UUID DEFAULT NULL)
RETURNS VOID AS $$
BEGIN
  -- Deduct credits atomically
  UPDATE public.users
  SET credits = credits - p_amount, updated_at = NOW()
  WHERE id = p_user_id AND credits >= p_amount;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Insufficient credits';
  END IF;

  -- Log transaction
  INSERT INTO public.credit_transactions (user_id, amount, type, generation_id, description)
  VALUES (p_user_id, -p_amount, 'deduction', p_gen_id, 'Video generation');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to add credits (for purchases/refunds)
CREATE OR REPLACE FUNCTION add_credits(p_user_id UUID, p_amount INTEGER, p_type TEXT, p_description TEXT DEFAULT NULL)
RETURNS VOID AS $$
BEGIN
  -- Add credits
  UPDATE public.users
  SET credits = credits + p_amount, updated_at = NOW()
  WHERE id = p_user_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'User not found';
  END IF;

  -- Log transaction
  INSERT INTO public.credit_transactions (user_id, amount, type, description)
  VALUES (p_user_id, p_amount, p_type, p_description);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to refund credits
CREATE OR REPLACE FUNCTION refund_credits(p_gen_id UUID)
RETURNS VOID AS $$
DECLARE
  v_user_id UUID;
  v_credits_used INTEGER;
BEGIN
  -- Get generation details
  SELECT user_id, credits_used INTO v_user_id, v_credits_used
  FROM public.generations
  WHERE id = p_gen_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Generation not found';
  END IF;

  -- Refund credits
  UPDATE public.users
  SET credits = credits + v_credits_used, updated_at = NOW()
  WHERE id = v_user_id;

  -- Log refund transaction
  INSERT INTO public.credit_transactions (user_id, amount, type, generation_id, description)
  VALUES (v_user_id, v_credits_used, 'refund', p_gen_id, 'Generation failed - refund');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Add job_id column to generations if not exists
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='generations' AND column_name='job_id') THEN
    ALTER TABLE public.generations ADD COLUMN job_id TEXT;
    CREATE INDEX idx_generations_job_id ON public.generations(job_id);
  END IF;
END $$;

-- Add progress column for tracking
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='generations' AND column_name='progress') THEN
    ALTER TABLE public.generations ADD COLUMN progress INTEGER DEFAULT 0;
  END IF;
END $$;

-- Add error_message column
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_name='generations' AND column_name='error_message') THEN
    ALTER TABLE public.generations ADD COLUMN error_message TEXT;
  END IF;
END $$;
