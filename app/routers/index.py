# app/routers/index.py
from fastapi import APIRouter, Depends
from ..deps_dual import get_current_user_dual  # <— change import

router = APIRouter(tags=["app"])

@router.get("/index")
async def index(current_user=Depends(get_current_user_dual)):
    return {
        "message": "Welcome to the Bike Suspension Viz backend!",
        "user": {"email": current_user.get("email"), "role": current_user.get("role")},
    }

@router.get("/users/me")
async def whoami(current_user=Depends(get_current_user_dual)):
    return {
        "email": current_user.get("email"),
        "role": current_user.get("role"),
        "is_active": current_user.get("is_active", True),
    }