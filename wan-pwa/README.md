# Wan2.1 PWA - AI Video Generation Platform

A production-ready Progressive Web App for AI-powered video generation using Wan2.1 models.

## Features

- 🎨 Smart Prompt Engineering: 50+ templates with context-aware suggestions
- 🎬 Video Generation: Text-to-Video and Image-to-Video
- 📱 Progressive Web App: Installable, offline-capable
- 🔐 Authentication: Supabase Auth with OAuth support
- 💳 Credit System: Freemium model with usage tracking
- ⚡ Real-time Progress: WebSocket-based generation tracking
- 🎯 Template Library: Categorized prompts (Cinematic, Animation, Realistic)
- 📥 Download & Share: Export videos to device

## Tech Stack

### Frontend
- Framework: Next.js 15 (App Router)
- UI: shadcn/ui + Tailwind CSS
- State: Zustand
- Forms: React Hook Form + Zod
- PWA: next-pwa

### Backend
- API: FastAPI (Python)
- GPU: Replicate / Modal
- Queue: Celery + Redis

### Database
- DB: Supabase (Postgres)
- Auth: Supabase Auth
- Storage: Supabase Storage
- Cache: Upstash Redis

## Project Structure

```
wan-pwa/
├── apps/
│   ├── web/          # Next.js frontend
│   └── api/          # FastAPI backend
├── packages/
│   ├── ui/           # Shared UI components
│   ├── db/           # Database schema & migrations
│   └── types/        # Shared TypeScript types
├── turbo.json        # Monorepo build config
└── package.json      # Root dependencies
```

## Prerequisites

- Node.js 18+
- Python 3.10+
- npm 9+
- Supabase account
- Replicate account

## Setup

See [SETUP.md](./SETUP.md) for detailed instructions.

## Quick Start

```bash
# Clone & Install
git clone <your-repo>
cd wan-pwa
npm install

# Start all services
npm run dev

# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

## Development Commands

```bash
npm run dev          # Start all services
npm run build        # Build all packages
npm run lint         # Lint all packages
npm run test         # Run all tests
npm run clean        # Clean build artifacts
npm run format       # Format code with Prettier
```

## Deployment

- Frontend: Vercel
- Backend: Modal or Railway
- Database: Supabase

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed instructions.

## Project Roadmap

- [x] Monorepo setup
- [ ] Authentication flows
- [ ] Prompt template system
- [ ] T2V generation
- [ ] I2V generation
- [ ] Batch processing
- [ ] Payment integration
- [ ] Mobile app

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md)

## License

MIT License - see [LICENSE](./LICENSE)
