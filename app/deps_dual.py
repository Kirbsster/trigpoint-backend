# app/deps_dual.py
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .security import decode_token
from .db import get_db
from .settings import settings

bearer = HTTPBearer(auto_error=False)

def _norm_email(e: str) -> str:
    return e.strip().lower()

async def get_current_user_dual(request: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    token = credentials.credentials if credentials else request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    data = decode_token(token)
    if not data or data.get("typ") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    email = data["sub"]
    user = await get_db()["users"].find_one({"email_norm": _norm_email(email)})

    if not user or not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="Inactive or unknown user")

    if settings.require_email_verification and not user.get("email_verified", False):
        raise HTTPException(status_code=403, detail="Email not verified")

    # preserve your password change invalidation
    token_iat = int(data.get("iat", 0))
    if token_iat < int(user.get("password_changed_at", 0)):
        raise HTTPException(status_code=401, detail="Token no longer valid; please login again")

    return user