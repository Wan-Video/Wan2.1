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
    supabase = get_supabase()

    # Calculate credit cost
    cost = CreditService.calculate_cost(request.model, request.resolution)

    # Check if user has sufficient credits
    credits_result = await CreditService.get_user_credits(user_id)
    if credits_result < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. You need {cost} credits but have {credits_result}.",
        )

    # Create generation record BEFORE calling Replicate
    generation_record = (
        supabase.table("generations")
        .insert(
            {
                "user_id": user_id,
                "type": "text-to-video",
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "model": request.model,
                "resolution": request.resolution,
                "status": "queued",
                "credits_used": cost,
                "progress": 0,
            }
        )
        .execute()
    )

    if not generation_record.data:
        raise HTTPException(status_code=500, detail="Failed to create generation record")

    generation_id = generation_record.data[0]["id"]

    try:
        # Start generation via Replicate
        job_id = await ReplicateService.generate_text_to_video(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model=request.model,
            resolution=request.resolution,
            duration=request.duration,
            seed=request.seed,
        )

        # Update generation with job_id and status
        supabase.table("generations").update(
            {"job_id": job_id, "status": "processing", "progress": 10}
        ).eq("id", generation_id).execute()

        # Deduct credits using database function
        try:
            supabase.rpc(
                "deduct_credits", {"p_user_id": user_id, "p_amount": cost, "p_gen_id": generation_id}
            ).execute()
        except Exception as credit_error:
            # Rollback: delete generation record
            supabase.table("generations").delete().eq("id", generation_id).execute()
            raise HTTPException(status_code=402, detail="Failed to deduct credits")

        return GenerationResponse(
            id=generation_id,
            status="processing",
            created_at=datetime.utcnow(),
            credits_used=cost,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Mark generation as failed
        supabase.table("generations").update(
            {"status": "failed", "error_message": str(e), "progress": 0}
        ).eq("id", generation_id).execute()

        # Refund credits if they were deducted
        try:
            supabase.rpc("refund_credits", {"p_gen_id": generation_id}).execute()
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/image-to-video", response_model=GenerationResponse)
async def generate_image_to_video(
    request: ImageToVideoRequest, user_id: str = Depends(get_user_id)
):
    """Generate video from image"""
    supabase = get_supabase()

    # Calculate credit cost
    cost = CreditService.calculate_cost(request.model, request.resolution)

    # Check if user has sufficient credits
    credits_result = await CreditService.get_user_credits(user_id)
    if credits_result < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. You need {cost} credits but have {credits_result}.",
        )

    # Create generation record BEFORE calling Replicate
    generation_record = (
        supabase.table("generations")
        .insert(
            {
                "user_id": user_id,
                "type": "image-to-video",
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "image_url": request.image_url,
                "model": request.model,
                "resolution": request.resolution,
                "status": "queued",
                "credits_used": cost,
                "progress": 0,
            }
        )
        .execute()
    )

    if not generation_record.data:
        raise HTTPException(status_code=500, detail="Failed to create generation record")

    generation_id = generation_record.data[0]["id"]

    try:
        # Start generation via Replicate
        job_id = await ReplicateService.generate_image_to_video(
            prompt=request.prompt,
            image_url=request.image_url,
            negative_prompt=request.negative_prompt,
            resolution=request.resolution,
            duration=request.duration,
            seed=request.seed,
        )

        # Update generation with job_id and status
        supabase.table("generations").update(
            {"job_id": job_id, "status": "processing", "progress": 10}
        ).eq("id", generation_id).execute()

        # Deduct credits using database function
        try:
            supabase.rpc(
                "deduct_credits", {"p_user_id": user_id, "p_amount": cost, "p_gen_id": generation_id}
            ).execute()
        except Exception as credit_error:
            # Rollback: delete generation record
            supabase.table("generations").delete().eq("id", generation_id).execute()
            raise HTTPException(status_code=402, detail="Failed to deduct credits")

        return GenerationResponse(
            id=generation_id,
            status="processing",
            created_at=datetime.utcnow(),
            credits_used=cost,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Mark generation as failed
        supabase.table("generations").update(
            {"status": "failed", "error_message": str(e), "progress": 0}
        ).eq("id", generation_id).execute()

        # Refund credits if they were deducted
        try:
            supabase.rpc("refund_credits", {"p_gen_id": generation_id}).execute()
        except:
            pass

        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


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

    gen_data = generation.data

    # If generation is already completed, return cached data
    if gen_data.get("status") in ["completed", "failed"]:
        return GenerationStatus(
            id=generation_id,
            status=gen_data["status"],
            progress=gen_data.get("progress", 100 if gen_data["status"] == "completed" else 0),
            video_url=gen_data.get("video_url"),
            error=gen_data.get("error_message"),
            logs=None,
        )

    # Get live status from Replicate using job_id
    job_id = gen_data.get("job_id")
    if not job_id:
        # No job_id yet, return queued status
        return GenerationStatus(
            id=generation_id,
            status="queued",
            progress=0,
            video_url=None,
            error=None,
            logs=None,
        )

    try:
        replicate_status = await ReplicateService.get_prediction_status(job_id)

        # Update database with latest status
        update_data = {}

        # Map Replicate status to our status
        status_map = {
            "starting": "processing",
            "processing": "processing",
            "succeeded": "completed",
            "failed": "failed",
            "canceled": "failed",
        }

        new_status = status_map.get(replicate_status["status"], "processing")
        update_data["status"] = new_status

        # Update progress
        if new_status == "processing":
            update_data["progress"] = 50
        elif new_status == "completed":
            update_data["progress"] = 100
        elif new_status == "failed":
            update_data["progress"] = 0

        # Save video URL if completed
        if replicate_status.get("output"):
            video_url = replicate_status["output"]
            if isinstance(video_url, list):
                video_url = video_url[0]
            update_data["video_url"] = video_url
            update_data["completed_at"] = datetime.utcnow().isoformat()

        # Save error if failed
        if replicate_status.get("error"):
            update_data["error_message"] = replicate_status["error"]

        # Update database
        supabase.table("generations").update(update_data).eq("id", generation_id).execute()

        return GenerationStatus(
            id=generation_id,
            status=new_status,
            progress=update_data.get("progress", 0),
            video_url=update_data.get("video_url"),
            error=update_data.get("error_message"),
            logs=replicate_status.get("logs"),
        )

    except Exception as e:
        # If Replicate call fails, return database status
        return GenerationStatus(
            id=generation_id,
            status=gen_data["status"],
            progress=gen_data.get("progress", 0),
            video_url=gen_data.get("video_url"),
            error=gen_data.get("error_message"),
            logs=None,
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
