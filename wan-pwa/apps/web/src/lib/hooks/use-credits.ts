"use client"

import { useState, useEffect } from "react"
import { createClient } from "@/lib/supabase/client"

export function useCredits(userId: string | undefined) {
  const [credits, setCredits] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchCredits = async () => {
    if (!userId) {
      setCredits(null)
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      const supabase = createClient()
      const { data, error: fetchError } = await supabase
        .from("users")
        .select("credits")
        .eq("id", userId)
        .single()

      if (fetchError) throw fetchError

      setCredits(data?.credits || 0)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch credits")
      setCredits(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCredits()
  }, [userId])

  const refreshCredits = () => {
    fetchCredits()
  }

  const optimisticDeduct = (amount: number) => {
    setCredits((prev) => (prev !== null ? Math.max(0, prev - amount) : null))
  }

  return {
    credits,
    loading,
    error,
    refreshCredits,
    optimisticDeduct,
  }
}
