# Database Package

This package contains database schema and migrations for the Wan2.1 PWA.

## Setup

1. Go to your Supabase dashboard
2. Navigate to SQL Editor
3. Create a new query
4. Copy the contents of `migrations/001_initial_schema.sql`
5. Run the query

## Schema

### Tables

#### users
- Stores user profile data
- Extends Supabase auth.users
- Tracks credits and subscription tier

#### generations
- Stores video generation requests and results
- Links to users and tracks status
- Stores prompts, settings, and output URLs

#### credit_transactions
- Tracks all credit additions and deductions
- Provides audit trail for user credits

### Storage

#### images bucket
- Stores uploaded images for Image-to-Video generation
- Publicly accessible
- Organized by user ID

## Row Level Security (RLS)

All tables have RLS enabled to ensure users can only access their own data:

- Users can read/update their own profile
- Users can view/create their own generations
- Users can view their own transactions

## Migrations

Migrations should be run in order:
1. `001_initial_schema.sql` - Core schema
2. `002_seed_data.sql` - Optional seed data

## Indexes

Indexes are created on:
- `generations.user_id`
- `generations.created_at`
- `credit_transactions.user_id`
- `credit_transactions.created_at`

These optimize common queries for user data and history.
