import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <div className="max-w-4xl text-center">
        <h1 className="mb-4 text-6xl font-bold tracking-tight">
          Wan2.1 <span className="text-primary">PWA</span>
        </h1>
        <p className="mb-8 text-xl text-muted-foreground">
          AI-Powered Video Generation Platform
        </p>
        <p className="mb-12 text-lg">
          Create stunning videos with Text-to-Video and Image-to-Video using state-of-the-art
          Wan2.1 models. 50+ prompt templates, real-time generation tracking, and installable PWA
          experience.
        </p>

        <div className="flex flex-col gap-4 sm:flex-row sm:justify-center">
          <Button asChild size="lg">
            <Link href="/auth/signup">Get Started</Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link href="/dashboard">View Dashboard</Link>
          </Button>
        </div>

        <div className="mt-16 grid gap-8 sm:grid-cols-3">
          <div className="rounded-lg border p-6">
            <h3 className="mb-2 text-lg font-semibold">Text-to-Video</h3>
            <p className="text-sm text-muted-foreground">
              Generate videos from text prompts using advanced AI models
            </p>
          </div>
          <div className="rounded-lg border p-6">
            <h3 className="mb-2 text-lg font-semibold">Image-to-Video</h3>
            <p className="text-sm text-muted-foreground">
              Animate images into dynamic video sequences
            </p>
          </div>
          <div className="rounded-lg border p-6">
            <h3 className="mb-2 text-lg font-semibold">50+ Templates</h3>
            <p className="text-sm text-muted-foreground">
              Pre-built prompt templates for every creative need
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
