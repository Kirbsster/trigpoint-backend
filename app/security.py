from datetime import datetime, timedelta, timezone
from time import time
import secrets
from typing import Any, Dict, Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from .settings import settings
from typing import Iterable

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def _encode_jwt(sub: str, role: str, typ: str, expires_delta: timedelta) -> str:
    now = datetime.now(tz=timezone.utc)
    payload: Dict[str, Any] = {
        "sub": sub,
        "role": role,
        "typ": typ,  # "access" | "refresh"
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def create_access_token(sub: str, role: str) -> str:
    return _encode_jwt(sub, role, "access", timedelta(minutes=settings.access_token_expire_minutes))

def create_refresh_token(sub: str, role: str) -> str:
    return _encode_jwt(sub, role, "refresh", timedelta(days=settings.refresh_token_expire_days))

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None

def create_reset_token(sub: str, jti: str, minutes: int = 15) -> str:
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(minutes=minutes)
    payload = {
        "sub": sub,
        "typ": "reset",
        "jti": jti,  # must match user's reset_nonce
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def create_verify_token(sub: str, jti: str, hours: int) -> str:
    now = datetime.now(tz=timezone.utc)
    exp = now + timedelta(hours=hours)
    payload = {
        "sub": sub,
        "typ": "verify",
        "jti": jti,  # must match user's verification_nonce
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def password_reused(candidate: str, hashes: Iterable[str]) -> bool:
    """
    Returns True if 'candidate' matches any hash in 'hashes' (Argon2 verify).
    """
    for h in hashes or []:
        try:
            if verify_password(candidate, h):  # uses Argon2 verify
                return True
        except Exception:
            # ignore malformed/legacy hashes
            continue
    return False
