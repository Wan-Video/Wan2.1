from fastapi import APIRouter, HTTPException, Header, Request
import hmac
import hashlib
import json
import os
from core.supabase import get_supabase
from datetime import datetime

router = APIRouter()


@router.post("/replicate")
async def replicate_webhook(request: Request, webhook_signature: str = Header(None, alias="Webhook-Signature")):
    """
    Handle Replicate completion webhook

    This endpoint receives push notifications from Replicate when predictions complete,
    eliminating the need for constant polling.
    """

    # Read raw body for signature verification
    body = await request.body()

    # Verify webhook signature
    secret = os.getenv("REPLICATE_WEBHOOK_SECRET")
    if secret:
        expected_signature = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()

        if not webhook_signature or not hmac.compare_digest(webhook_signature, expected_signature):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Parse payload
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Extract prediction data
    job_id = payload.get("id")
    status = payload.get("status")
    output = payload.get("output")
    error = payload.get("error")

    if not job_id:
        raise HTTPException(status_code=400, detail="Missing prediction ID")

    # Update database
    supabase = get_supabase()

    # Find generation by job_id
    generation_result = (
        supabase.table("generations")
        .select("id, user_id")
        .eq("job_id", job_id)
        .single()
        .execute()
    )

    if not generation_result.data:
        # Generation not found - this is expected for non-Wan predictions
        return {"status": "ignored", "message": "Generation not found"}

    generation_id = generation_result.data["id"]

    # Prepare update data
    update_data = {}

    # Map Replicate status to our status
    status_map = {
        "starting": "processing",
        "processing": "processing",
        "succeeded": "completed",
        "failed": "failed",
        "canceled": "failed",
    }

    new_status = status_map.get(status, "processing")
    update_data["status"] = new_status

    # Update progress
    if new_status == "processing":
        update_data["progress"] = 50
    elif new_status == "completed":
        update_data["progress"] = 100
        update_data["completed_at"] = datetime.utcnow().isoformat()
    elif new_status == "failed":
        update_data["progress"] = 0

    # Save video URL if completed
    if status == "succeeded" and output:
        video_url = output
        if isinstance(video_url, list):
            video_url = video_url[0]
        update_data["video_url"] = video_url

    # Save error if failed
    if error:
        update_data["error_message"] = str(error)

    # Update database
    supabase.table("generations").update(update_data).eq("id", generation_id).execute()

    # If failed, trigger refund
    if new_status == "failed":
        try:
            supabase.rpc("refund_credits", {"p_gen_id": generation_id}).execute()
        except Exception as refund_error:
            print(f"Failed to refund credits for generation {generation_id}: {refund_error}")

    return {
        "status": "ok",
        "generation_id": generation_id,
        "new_status": new_status
    }


@router.get("/health")
async def webhook_health():
    """Health check endpoint for webhook"""
    return {"status": "ok", "message": "Webhook endpoint is healthy"}
