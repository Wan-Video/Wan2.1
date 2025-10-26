from core.supabase import get_supabase
from typing import Optional


class CreditService:
    """Service for managing user credits"""

    # Credit costs for different operations
    COSTS = {
        "t2v-14B-480p": 10,
        "t2v-14B-720p": 20,
        "t2v-1.3B-480p": 5,
        "i2v-14B-480p": 15,
        "i2v-14B-720p": 25,
    }

    # Free tier credits
    FREE_TIER_CREDITS = 100

    @staticmethod
    async def get_user_credits(user_id: str) -> int:
        """Get current credit balance for user"""
        supabase = get_supabase()
        result = supabase.table("users").select("credits").eq("id", user_id).single().execute()
        return result.data.get("credits", 0) if result.data else 0

    @staticmethod
    async def deduct_credits(user_id: str, amount: int, description: str) -> bool:
        """
        Deduct credits from user account

        Args:
            user_id: User ID
            amount: Amount of credits to deduct
            description: Description of the transaction

        Returns:
            True if successful, False if insufficient credits
        """
        supabase = get_supabase()

        # Get current balance
        current_credits = await CreditService.get_user_credits(user_id)

        if current_credits < amount:
            return False

        # Deduct credits
        new_balance = current_credits - amount
        supabase.table("users").update({"credits": new_balance}).eq("id", user_id).execute()

        # Record transaction
        supabase.table("credit_transactions").insert(
            {
                "user_id": user_id,
                "amount": -amount,
                "type": "deduction",
                "description": description,
            }
        ).execute()

        return True

    @staticmethod
    async def add_credits(user_id: str, amount: int, description: str) -> bool:
        """
        Add credits to user account

        Args:
            user_id: User ID
            amount: Amount of credits to add
            description: Description of the transaction

        Returns:
            True if successful
        """
        supabase = get_supabase()

        # Get current balance
        current_credits = await CreditService.get_user_credits(user_id)

        # Add credits
        new_balance = current_credits + amount
        supabase.table("users").update({"credits": new_balance}).eq("id", user_id).execute()

        # Record transaction
        supabase.table("credit_transactions").insert(
            {
                "user_id": user_id,
                "amount": amount,
                "type": "addition",
                "description": description,
            }
        ).execute()

        return True

    @staticmethod
    def calculate_cost(model: str, resolution: str) -> int:
        """Calculate credit cost for a generation request"""
        key = f"{model}-{resolution}"
        return CreditService.COSTS.get(key, 10)
