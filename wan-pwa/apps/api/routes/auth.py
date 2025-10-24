from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from core.supabase import get_supabase

router = APIRouter()


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup")
async def sign_up(request: SignUpRequest):
    """Sign up a new user"""
    supabase = get_supabase()

    try:
        result = supabase.auth.sign_up({"email": request.email, "password": request.password})

        if result.user:
            # Initialize user with free credits
            supabase.table("users").insert(
                {
                    "id": result.user.id,
                    "email": request.email,
                    "credits": 100,  # Free tier credits
                    "subscription_tier": "free",
                }
            ).execute()

        return {"user": result.user, "session": result.session}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/signin")
async def sign_in(request: SignInRequest):
    """Sign in an existing user"""
    supabase = get_supabase()

    try:
        result = supabase.auth.sign_in_with_password(
            {"email": request.email, "password": request.password}
        )
        return {"user": result.user, "session": result.session}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/signout")
async def sign_out():
    """Sign out the current user"""
    supabase = get_supabase()

    try:
        supabase.auth.sign_out()
        return {"message": "Signed out successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
