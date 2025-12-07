from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class User(BaseModel):
    id: str
    email: EmailStr
    credits: int
    subscription_tier: str = "free"
    created_at: datetime
    updated_at: datetime


class UserCredits(BaseModel):
    user_id: str
    credits: int
    subscription_tier: str


class CreditTransaction(BaseModel):
    id: str
    user_id: str
    amount: int
    type: str
    description: Optional[str] = None
    created_at: datetime
