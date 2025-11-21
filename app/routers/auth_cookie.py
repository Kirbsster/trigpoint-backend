# app/routers/auth_cookie.py
from fastapi import APIRouter, HTTPException, Response, Request, status
from pydantic import BaseModel

from ..db import get_db
from ..security import verify_password, create_access_token, create_refresh_token, decode_token
from ..settings import settings

router = APIRouter(prefix="/auth", tags=["auth"])

COOKIE_ACCESS  = "access_token"
COOKIE_REFRESH = "refresh_token"


class SessionLoginIn(BaseModel):
    email: str
    password: str
    remember_me: bool = False


def _norm_email(e: str) -> str:
    return e.strip().lower()


def _set_auth_cookies(response: Response, access: str, refresh: str, remember: bool = False) -> None:
    """Set HttpOnly cookies for access & refresh JWTs."""
    secure_flag = settings.env.lower() != "dev"

    # default lifetimes
    access_age  = settings.access_token_expire_minutes * 60         # e.g. 30 min
    refresh_age = settings.refresh_token_expire_days * 24 * 3600    # e.g. 7 days

    # extend if remember_me selected
    if remember:
        access_age  = 14 * 24 * 3600    # 14 days access
        refresh_age = 30 * 24 * 3600    # 30 days refresh

    response.set_cookie(
        key=COOKIE_ACCESS,
        value=access,
        httponly=True,
        secure=secure_flag,
        samesite="Lax",
        max_age=access_age,
        path="/",
    )
    response.set_cookie(
        key=COOKIE_REFRESH,
        value=refresh,
        httponly=True,
        secure=secure_flag,
        samesite="Strict",
        max_age=refresh_age,
        path="/auth",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(COOKIE_ACCESS, path="/")
    response.delete_cookie(COOKIE_REFRESH, path="/auth")


@router.post("/session-login")
async def session_login(payload: SessionLoginIn, response: Response):
    """
    Browser login:
    - Accepts JSON { "email", "password", "remember_me" }
    - Sets HttpOnly access+refresh cookies
    """
    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(payload.email)})

    if not user or not user.get("hashed_password") or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")

    # ⬇︎ these two checks are what you were missing
    if settings.require_email_verification and not user.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User inactive",
        )

    role = user.get("role", "user")
    access  = create_access_token(user["email"], role)
    refresh = create_refresh_token(user["email"], role)

    _set_auth_cookies(response, access, refresh, remember=payload.remember_me)

    # ⬇︎ include email/role so AuthState can update immediately
    return {
        "ok": True,
        "email": user["email"],
        "role": role,
        "remember_me": payload.remember_me,
    }


@router.post("/session-refresh")
async def session_refresh(request: Request, response: Response):
    """
    Refresh session based on refresh_token cookie.
    Returns {ok: True} and rotates cookies if valid.
    """
    token = request.cookies.get(COOKIE_REFRESH)
    data = decode_token(token) if token else None

    if not data or data.get("typ") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    email = data.get("sub")
    role  = data.get("role") or "user"

    if not email:
        raise HTTPException(status_code=401, detail="Invalid refresh token payload")

    # ⬇︎ reload user to ensure still active
    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(email)})
    if not user or not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="User inactive or not found")

    access  = create_access_token(email, role)
    refresh = create_refresh_token(email, role)
    _set_auth_cookies(response, access, refresh)

    return {"ok": True}


@router.post("/session-logout")
async def session_logout(response: Response):
    _clear_auth_cookies(response)
    return {"ok": True}