from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class TextToVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(None, max_length=2000)
    model: Literal["t2v-14B", "t2v-1.3B"] = "t2v-14B"
    resolution: Literal["480p", "720p"] = "720p"
    duration: int = Field(5, ge=1, le=10)
    seed: Optional[int] = None


class ImageToVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    image_url: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = Field(None, max_length=2000)
    model: Literal["i2v-14B"] = "i2v-14B"
    resolution: Literal["480p", "720p"] = "720p"
    duration: int = Field(5, ge=1, le=10)
    seed: Optional[int] = None


class GenerationResponse(BaseModel):
    id: str
    status: Literal["pending", "processing", "completed", "failed"]
    video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    credits_used: int


class GenerationStatus(BaseModel):
    id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: int = Field(0, ge=0, le=100)
    video_url: Optional[str] = None
    error: Optional[str] = None
    logs: Optional[str] = None
