export type PromptCategory =
  | "cinematic"
  | "animation"
  | "realistic"
  | "abstract"
  | "nature"
  | "people"
  | "animals"

export interface PromptTemplate {
  id: string
  title: string
  category: PromptCategory
  prompt: string
  negativePrompt?: string
  tags: string[]
  featured?: boolean
}

export const promptTemplates: PromptTemplate[] = [
  // Cinematic
  {
    id: "cinematic-1",
    title: "Epic Movie Scene",
    category: "cinematic",
    prompt:
      "Cinematic wide shot of a lone hero standing on a cliff edge at sunset, dramatic lighting, volumetric fog, epic scale, film grain, shallow depth of field",
    negativePrompt:
      "blurry, low quality, static, overexposed, ugly, deformed, amateur, cartoon",
    tags: ["cinematic", "hero", "sunset", "epic"],
    featured: true,
  },
  {
    id: "cinematic-2",
    title: "Noir Detective",
    category: "cinematic",
    prompt:
      "Film noir style detective walking through rain-soaked city streets at night, neon lights reflecting on wet pavement, high contrast lighting, moody atmosphere",
    negativePrompt: "bright colors, daytime, cheerful, low quality",
    tags: ["noir", "detective", "rain", "night"],
  },
  {
    id: "cinematic-3",
    title: "Space Opera",
    category: "cinematic",
    prompt:
      "Epic space battle with massive starships, laser beams, explosions, nebula in background, cinematic camera movement, lens flares",
    negativePrompt: "static, low quality, cartoon, unrealistic",
    tags: ["space", "battle", "sci-fi", "epic"],
  },
  {
    id: "cinematic-4",
    title: "Medieval Battle",
    category: "cinematic",
    prompt:
      "Epic medieval battle scene, knights charging on horseback, castle siege, dramatic sky, volumetric dust and smoke, cinematic composition",
    tags: ["medieval", "battle", "knights", "epic"],
  },
  {
    id: "cinematic-5",
    title: "Dystopian City",
    category: "cinematic",
    prompt:
      "Futuristic dystopian cityscape, towering megastructures, neon signs, flying vehicles, rain, cyberpunk aesthetic, cinematic drone shot",
    tags: ["cyberpunk", "dystopia", "future", "city"],
    featured: true,
  },

  // Animation
  {
    id: "animation-1",
    title: "Pixar Style Character",
    category: "animation",
    prompt:
      "Cute cartoon character in Pixar animation style, expressive eyes, dynamic pose, colorful environment, warm lighting, high quality 3D render",
    negativePrompt: "realistic, photorealistic, ugly, deformed, low quality",
    tags: ["pixar", "cute", "3d", "character"],
    featured: true,
  },
  {
    id: "animation-2",
    title: "Studio Ghibli Landscape",
    category: "animation",
    prompt:
      "Beautiful countryside landscape in Studio Ghibli animation style, rolling hills, traditional Japanese houses, cherry blossoms, peaceful atmosphere",
    tags: ["ghibli", "landscape", "peaceful", "japan"],
  },
  {
    id: "animation-3",
    title: "Cartoon Animals",
    category: "animation",
    prompt:
      "Adorable cartoon animals playing in a magical forest, vibrant colors, playful animation, Disney-style, whimsical atmosphere",
    tags: ["animals", "cartoon", "forest", "playful"],
  },
  {
    id: "animation-4",
    title: "Anime Action",
    category: "animation",
    prompt:
      "Dynamic anime-style action sequence, character with special powers, energy effects, speed lines, dramatic camera angles, vibrant colors",
    tags: ["anime", "action", "powers", "dynamic"],
  },
  {
    id: "animation-5",
    title: "Claymation Scene",
    category: "animation",
    prompt:
      "Charming claymation-style scene with textured characters, stop-motion aesthetic, warm lighting, cozy atmosphere, handcrafted feel",
    tags: ["claymation", "stop-motion", "handcrafted", "cozy"],
  },

  // Realistic
  {
    id: "realistic-1",
    title: "Nature Documentary",
    category: "realistic",
    prompt:
      "National Geographic style wildlife footage, majestic lion on African savanna at golden hour, cinematic camera work, 4K quality, natural lighting",
    negativePrompt: "cartoon, animated, low quality, static",
    tags: ["wildlife", "nature", "documentary", "africa"],
    featured: true,
  },
  {
    id: "realistic-2",
    title: "Urban Photography",
    category: "realistic",
    prompt:
      "Photorealistic urban street scene, busy city intersection, people walking, cars moving, realistic lighting and shadows, documentary style",
    tags: ["urban", "street", "documentary", "people"],
  },
  {
    id: "realistic-3",
    title: "Portrait Cinematic",
    category: "realistic",
    prompt:
      "Cinematic portrait of a person, shallow depth of field, professional lighting, emotional expression, film grain, anamorphic lens look",
    tags: ["portrait", "cinematic", "emotional", "professional"],
  },
  {
    id: "realistic-4",
    title: "Architectural Tour",
    category: "realistic",
    prompt:
      "Architectural visualization, modern building exterior, smooth camera movement, golden hour lighting, photorealistic materials and textures",
    tags: ["architecture", "building", "modern", "professional"],
  },
  {
    id: "realistic-5",
    title: "Underwater World",
    category: "realistic",
    prompt:
      "Photorealistic underwater scene, colorful coral reef, tropical fish swimming, sunlight rays penetrating water, documentary quality",
    tags: ["underwater", "ocean", "reef", "documentary"],
  },

  // Abstract
  {
    id: "abstract-1",
    title: "Liquid Art",
    category: "abstract",
    prompt:
      "Abstract liquid art, flowing colorful fluids, paint mixing, organic shapes, macro photography style, vibrant colors, smooth motion",
    tags: ["abstract", "liquid", "colorful", "organic"],
  },
  {
    id: "abstract-2",
    title: "Geometric Motion",
    category: "abstract",
    prompt:
      "Abstract geometric shapes morphing and rotating, neon colors, symmetrical patterns, mathematical precision, hypnotic motion",
    tags: ["geometric", "abstract", "neon", "patterns"],
  },
  {
    id: "abstract-3",
    title: "Particle System",
    category: "abstract",
    prompt:
      "Abstract particle system, millions of particles forming complex patterns, flowing and dissolving, ethereal atmosphere, dark background",
    tags: ["particles", "abstract", "ethereal", "complex"],
  },
  {
    id: "abstract-4",
    title: "Fractal Zoom",
    category: "abstract",
    prompt:
      "Infinite fractal zoom, mathematical patterns, psychedelic colors, recursive geometry, mesmerizing motion, high detail",
    tags: ["fractal", "mathematical", "psychedelic", "infinite"],
  },
  {
    id: "abstract-5",
    title: "Digital Glitch",
    category: "abstract",
    prompt:
      "Digital glitch art aesthetic, datamoshing effects, chromatic aberration, pixel sorting, cyberpunk colors, corrupted data visualization",
    tags: ["glitch", "digital", "cyberpunk", "corrupted"],
  },

  // Nature
  {
    id: "nature-1",
    title: "Mountain Landscape",
    category: "nature",
    prompt:
      "Majestic mountain landscape with snow-capped peaks, alpine meadow with wildflowers, flowing stream, dramatic clouds, sunrise lighting",
    tags: ["mountain", "landscape", "alpine", "sunrise"],
  },
  {
    id: "nature-2",
    title: "Ocean Waves",
    category: "nature",
    prompt:
      "Powerful ocean waves crashing on rocky shore, sea spray, dramatic sky, slow motion, natural beauty, coastal scenery",
    tags: ["ocean", "waves", "coastal", "dramatic"],
  },
  {
    id: "nature-3",
    title: "Forest Path",
    category: "nature",
    prompt:
      "Peaceful forest path with sunlight filtering through trees, dappled light, morning mist, lush vegetation, serene atmosphere",
    tags: ["forest", "path", "peaceful", "sunlight"],
  },
  {
    id: "nature-4",
    title: "Desert Sunset",
    category: "nature",
    prompt:
      "Vast desert landscape at sunset, sand dunes, warm golden light, long shadows, clear sky transitioning to night, peaceful solitude",
    tags: ["desert", "sunset", "dunes", "peaceful"],
  },
  {
    id: "nature-5",
    title: "Waterfall Paradise",
    category: "nature",
    prompt:
      "Stunning tropical waterfall, crystal clear water, lush green vegetation, rainbow in mist, natural pool, paradise setting",
    tags: ["waterfall", "tropical", "paradise", "rainbow"],
  },

  // People
  {
    id: "people-1",
    title: "Dance Performance",
    category: "people",
    prompt:
      "Professional dancer performing contemporary dance, fluid movements, dramatic lighting, stage performance, emotional expression, elegant choreography",
    tags: ["dance", "performance", "elegant", "artistic"],
  },
  {
    id: "people-2",
    title: "Street Musician",
    category: "people",
    prompt:
      "Street musician playing guitar on urban street corner, passersby, natural lighting, documentary style, authentic moment, city atmosphere",
    tags: ["music", "street", "urban", "documentary"],
  },
  {
    id: "people-3",
    title: "Chef at Work",
    category: "people",
    prompt:
      "Professional chef preparing gourmet dish, kitchen environment, precise movements, steam and sizzling, cinematic close-up shots, culinary artistry",
    tags: ["chef", "cooking", "culinary", "professional"],
  },
  {
    id: "people-4",
    title: "Athlete Training",
    category: "people",
    prompt:
      "Athlete training intensely, gym environment, dynamic movements, sweat details, determination, motivational atmosphere, dramatic lighting",
    tags: ["athlete", "training", "fitness", "motivation"],
  },
  {
    id: "people-5",
    title: "Fashion Runway",
    category: "people",
    prompt:
      "Fashion model walking down runway, haute couture clothing, professional lighting, confident stride, fashion show atmosphere, elegant presentation",
    tags: ["fashion", "runway", "model", "elegant"],
  },

  // Animals
  {
    id: "animals-1",
    title: "Bird in Flight",
    category: "animals",
    prompt:
      "Majestic eagle soaring through mountain valley, wings spread, slow motion, natural habitat, cloudy sky, wildlife cinematography",
    tags: ["bird", "eagle", "flight", "wildlife"],
  },
  {
    id: "animals-2",
    title: "Playful Dolphins",
    category: "animals",
    prompt:
      "Pod of dolphins jumping and playing in ocean waves, underwater and above water shots, sunlight, joyful energy, marine life",
    tags: ["dolphins", "ocean", "playful", "marine"],
  },
  {
    id: "animals-3",
    title: "Tiger Hunt",
    category: "animals",
    prompt:
      "Bengal tiger stalking through jungle, intense focus, powerful movements, dappled sunlight through canopy, predator instincts, wildlife drama",
    tags: ["tiger", "jungle", "predator", "wildlife"],
  },
  {
    id: "animals-4",
    title: "Butterfly Metamorphosis",
    category: "animals",
    prompt:
      "Time-lapse of butterfly emerging from chrysalis, delicate wings unfurling, macro detail, natural beauty, transformation process",
    tags: ["butterfly", "metamorphosis", "macro", "nature"],
  },
  {
    id: "animals-5",
    title: "Wolf Pack",
    category: "animals",
    prompt:
      "Wolf pack moving through snowy forest, coordinated movement, winter landscape, misty breath, wild beauty, pack dynamics",
    tags: ["wolf", "pack", "winter", "forest"],
  },

  // Additional templates to reach 50+
  {
    id: "cinematic-6",
    title: "Car Chase",
    category: "cinematic",
    prompt:
      "High-speed car chase through city streets, dynamic camera angles, motion blur, tire smoke, dramatic pursuit, action movie style",
    tags: ["cars", "chase", "action", "speed"],
  },
  {
    id: "animation-6",
    title: "Magical Transformation",
    category: "animation",
    prompt:
      "Magical girl transformation sequence, sparkles and light effects, anime style, dynamic poses, colorful energy, enchanting atmosphere",
    tags: ["magic", "transformation", "anime", "sparkles"],
  },
  {
    id: "realistic-6",
    title: "Concert Performance",
    category: "realistic",
    prompt:
      "Live concert performance, crowd energy, stage lights, musicians performing, photorealistic, dynamic camera work, electric atmosphere",
    tags: ["concert", "music", "performance", "crowd"],
  },
  {
    id: "abstract-7",
    title: "Smoke Art",
    category: "abstract",
    prompt:
      "Colorful smoke wisps and tendrils, black background, fluid motion, ethereal patterns, vibrant colors mixing, hypnotic movement",
    tags: ["smoke", "abstract", "colorful", "ethereal"],
  },
  {
    id: "nature-6",
    title: "Northern Lights",
    category: "nature",
    prompt:
      "Aurora borealis dancing across night sky, green and purple lights, snowy landscape below, stars visible, magical natural phenomenon",
    tags: ["aurora", "northern lights", "night", "magical"],
  },
]

export function getTemplatesByCategory(category: PromptCategory): PromptTemplate[] {
  return promptTemplates.filter((t) => t.category === category)
}

export function getFeaturedTemplates(): PromptTemplate[] {
  return promptTemplates.filter((t) => t.featured)
}

export function getTemplateById(id: string): PromptTemplate | undefined {
  return promptTemplates.find((t) => t.id === id)
}

export function searchTemplates(query: string): PromptTemplate[] {
  const lowerQuery = query.toLowerCase()
  return promptTemplates.filter(
    (t) =>
      t.title.toLowerCase().includes(lowerQuery) ||
      t.prompt.toLowerCase().includes(lowerQuery) ||
      t.tags.some((tag) => tag.toLowerCase().includes(lowerQuery))
  )
}
