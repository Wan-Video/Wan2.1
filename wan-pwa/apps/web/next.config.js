const withPWA = require("next-pwa")({
  dest: "public",
  register: true,
  skipWaiting: true,
  disable: process.env.NODE_ENV === "development",
})

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ["supabase.co", "replicate.delivery"],
  },
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
}

module.exports = withPWA(nextConfig)
