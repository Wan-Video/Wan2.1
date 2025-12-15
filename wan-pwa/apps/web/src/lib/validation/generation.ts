import { z } from "zod"

export const textToVideoSchema = z.object({
  prompt: z
    .string()
    .min(10, "Prompt must be at least 10 characters")
    .max(500, "Prompt must be under 500 characters"),
  negative_prompt: z.string().max(200, "Negative prompt must be under 200 characters").optional(),
  model: z.enum(["t2v-14B", "t2v-1.3B"], {
    required_error: "Please select a model",
  }),
  resolution: z.enum(["480p", "720p"], {
    required_error: "Please select a resolution",
  }),
  duration: z.number().int().min(1).max(10).default(5),
  seed: z.number().int().optional(),
})

export const imageToVideoSchema = z.object({
  prompt: z
    .string()
    .min(10, "Prompt must be at least 10 characters")
    .max(500, "Prompt must be under 500 characters"),
  negative_prompt: z.string().max(200, "Negative prompt must be under 200 characters").optional(),
  image_url: z.string().url("Please provide a valid image URL"),
  model: z.enum(["i2v-14B"], {
    required_error: "Please select a model",
  }),
  resolution: z.enum(["480p", "720p"], {
    required_error: "Please select a resolution",
  }),
  duration: z.number().int().min(1).max(10).default(5),
  seed: z.number().int().optional(),
})

export type TextToVideoInput = z.infer<typeof textToVideoSchema>
export type ImageToVideoInput = z.infer<typeof imageToVideoSchema>

// Credit cost calculator
export function calculateCreditCost(model: string, resolution: string): number {
  const costs: Record<string, number> = {
    "t2v-14B-720p": 20,
    "t2v-14B-480p": 10,
    "t2v-1.3B-480p": 5,
    "i2v-14B-720p": 25,
    "i2v-14B-480p": 15,
  }

  const key = `${model}-${resolution}`
  return costs[key] || 10
}
