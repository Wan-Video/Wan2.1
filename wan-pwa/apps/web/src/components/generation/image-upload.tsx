"use client"

import { useState, useCallback } from "react"
import { Upload, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"

interface ImageUploadProps {
  onImageSelect: (file: File) => void
  onImageRemove: () => void
  maxSizeMB?: number
}

export function ImageUpload({ onImageSelect, onImageRemove, maxSizeMB = 10 }: ImageUploadProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const validateAndProcessFile = useCallback(
    (file: File) => {
      // Validate file type
      if (!file.type.startsWith("image/")) {
        toast.error("Invalid file type", {
          description: "Please upload an image file (PNG, JPG, WEBP)",
        })
        return false
      }

      // Validate file size
      const maxSizeBytes = maxSizeMB * 1024 * 1024
      if (file.size > maxSizeBytes) {
        toast.error("File too large", {
          description: `Image must be under ${maxSizeMB}MB. Current size: ${(file.size / 1024 / 1024).toFixed(2)}MB`,
        })
        return false
      }

      // Create preview
      const reader = new FileReader()
      reader.onload = () => setPreview(reader.result as string)
      reader.readAsDataURL(file)

      onImageSelect(file)
      return true
    },
    [maxSizeMB, onImageSelect]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      setIsDragging(false)

      const file = e.dataTransfer.files[0]
      if (file) {
        validateAndProcessFile(file)
      }
    },
    [validateAndProcessFile]
  )

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        validateAndProcessFile(file)
      }
    },
    [validateAndProcessFile]
  )

  const handleRemove = () => {
    setPreview(null)
    onImageRemove()
  }

  if (preview) {
    return (
      <div className="relative">
        <img src={preview} alt="Upload preview" className="w-full rounded-lg object-cover" />
        <Button
          type="button"
          variant="destructive"
          size="icon"
          className="absolute right-2 top-2"
          onClick={handleRemove}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      className={`rounded-lg border-2 border-dashed p-8 text-center transition cursor-pointer ${
        isDragging
          ? "border-primary bg-primary/5"
          : "border-muted hover:border-primary/50 hover:bg-muted/50"
      }`}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleFileInput}
        className="hidden"
        id="image-upload"
      />
      <label htmlFor="image-upload" className="cursor-pointer">
        <Upload className="mx-auto mb-4 h-8 w-8 text-muted-foreground" />
        <p className="font-medium">Drop an image here</p>
        <p className="mt-1 text-sm text-muted-foreground">
          or click to browse (PNG, JPG, WEBP • Max {maxSizeMB}MB)
        </p>
      </label>
    </div>
  )
}
