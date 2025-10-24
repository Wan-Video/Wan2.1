from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional
from models.generation import (
    TextToVideoRequest,
    ImageToVideoRequest,
    GenerationResponse,
    GenerationStatus,
)
from services.replicate_service import ReplicateService
from services.credit_service import CreditService
from core.supabase import get_supabase
from datetime import datetime

router = APIRouter()


async def get_user_id(authorization: Optional[str] = Header(None)) -> str:
    """Extract user ID from authorization header"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.replace("Bearer ", "")
    supabase = get_supabase()

    try:
        user = supabase.auth.get_user(token)
        return user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/text-to-video", response_model=GenerationResponse)
async def generate_text_to_video(
    request: TextToVideoRequest, user_id: str = Depends(get_user_id)
):
    """Generate video from text prompt"""

    # Calculate credit cost
    cost = CreditService.calculate_cost(request.model, request.resolution)

    # Check and deduct credits
    has_credits = await CreditService.deduct_credits(
        user_id, cost, f"T2V generation: {request.model} @ {request.resolution}"
    )

    if not has_credits:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    try:
        # Start generation via Replicate
        prediction_id = await ReplicateService.generate_text_to_video(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model=request.model,
            resolution=request.resolution,
            duration=request.duration,
            seed=request.seed,
        )

        # Store generation record in database
        supabase = get_supabase()
        generation = (
            supabase.table("generations")
            .insert(
                {
                    "id": prediction_id,
                    "user_id": user_id,
                    "type": "text-to-video",
                    "prompt": request.prompt,
                    "model": request.model,
                    "resolution": request.resolution,
                    "status": "pending",
                    "credits_used": cost,
                }
            )
            .execute()
        )

        return GenerationResponse(
            id=prediction_id,
            status="pending",
            created_at=datetime.utcnow(),
            credits_used=cost,
        )

    except Exception as e:
        # Refund credits on error
        await CreditService.add_credits(user_id, cost, "Refund: Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image-to-video", response_model=GenerationResponse)
async def generate_image_to_video(
    request: ImageToVideoRequest, user_id: str = Depends(get_user_id)
):
    """Generate video from image"""

    # Calculate credit cost
    cost = CreditService.calculate_cost(request.model, request.resolution)

    # Check and deduct credits
    has_credits = await CreditService.deduct_credits(
        user_id, cost, f"I2V generation: {request.model} @ {request.resolution}"
    )

    if not has_credits:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    try:
        # Start generation via Replicate
        prediction_id = await ReplicateService.generate_image_to_video(
            prompt=request.prompt,
            image_url=request.image_url,
            negative_prompt=request.negative_prompt,
            resolution=request.resolution,
            duration=request.duration,
            seed=request.seed,
        )

        # Store generation record
        supabase = get_supabase()
        supabase.table("generations").insert(
            {
                "id": prediction_id,
                "user_id": user_id,
                "type": "image-to-video",
                "prompt": request.prompt,
                "image_url": request.image_url,
                "model": request.model,
                "resolution": request.resolution,
                "status": "pending",
                "credits_used": cost,
            }
        ).execute()

        return GenerationResponse(
            id=prediction_id,
            status="pending",
            created_at=datetime.utcnow(),
            credits_used=cost,
        )

    except Exception as e:
        # Refund credits on error
        await CreditService.add_credits(user_id, cost, "Refund: Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{generation_id}", response_model=GenerationStatus)
async def get_generation_status(generation_id: str, user_id: str = Depends(get_user_id)):
    """Get status of a generation"""

    # Verify ownership
    supabase = get_supabase()
    generation = (
        supabase.table("generations")
        .select("*")
        .eq("id", generation_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    if not generation.data:
        raise HTTPException(status_code=404, detail="Generation not found")

    # Get status from Replicate
    status = await ReplicateService.get_prediction_status(generation_id)

    # Update database with latest status
    update_data = {"status": status["status"]}
    if status["output"]:
        update_data["video_url"] = status["output"]
    if status["error"]:
        update_data["error"] = status["error"]

    supabase.table("generations").update(update_data).eq("id", generation_id).execute()

    # Map status to progress percentage
    progress_map = {"pending": 0, "processing": 50, "succeeded": 100, "failed": 0}

    return GenerationStatus(
        id=generation_id,
        status=status["status"],
        progress=progress_map.get(status["status"], 0),
        video_url=status["output"],
        error=status["error"],
        logs=status["logs"],
    )


@router.get("/history")
async def get_generation_history(user_id: str = Depends(get_user_id)):
    """Get user's generation history"""

    supabase = get_supabase()
    result = (
        supabase.table("generations")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )

    return {"generations": result.data}
