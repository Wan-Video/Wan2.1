from fastapi import APIRouter, Depends
from routes.generation import get_user_id
from services.credit_service import CreditService
from core.supabase import get_supabase

router = APIRouter()


@router.get("/me")
async def get_current_user(user_id: str = Depends(get_user_id)):
    """Get current user profile"""
    supabase = get_supabase()
    result = supabase.table("users").select("*").eq("id", user_id).single().execute()

    if not result.data:
        return {"error": "User not found"}

    return result.data


@router.get("/credits")
async def get_user_credits(user_id: str = Depends(get_user_id)):
    """Get user's credit balance"""
    credits = await CreditService.get_user_credits(user_id)
    return {"credits": credits}


@router.get("/transactions")
async def get_credit_transactions(user_id: str = Depends(get_user_id)):
    """Get user's credit transaction history"""
    supabase = get_supabase()
    result = (
        supabase.table("credit_transactions")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )

    return {"transactions": result.data}
