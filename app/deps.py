from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .security import decode_token
from .db import get_db

bearer = HTTPBearer(auto_error=False)

def _norm_email(e: str) -> str:
    return e.strip().lower()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing token")

    data = decode_token(credentials.credentials)
    if not data or data.get("typ") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    email = data["sub"]
    user = await get_db()["users"].find_one({"email_norm": _norm_email(email)})
    
    # deps.py (extra check inside get_current_user)
    token_iat = data.get("iat", 0)
    if token_iat < int(user.get("password_changed_at", 0)):
        raise HTTPException(status_code=401, detail="Token no longer valid; please login again")
    if not user or not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="Inactive or unknown user")
    return user