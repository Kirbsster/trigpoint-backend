from fastapi import APIRouter, HTTPException, status, Depends, Request, Response, BackgroundTasks
from ..rate_limit import SlidingWindowLimiter, RateLimitExceeded
from ..security import (verify_password, hash_password, create_access_token, 
                        create_refresh_token, decode_token, create_reset_token,
                        password_reused, create_verify_token)
from ..email_utils import send_email, verification_email_html, reset_password_email_html
from ..settings import settings
from ..db import get_db
from ..schemas import (LoginIn, RegisterIn, TokenPair, UserOut, ForgotPasswordIn,
                        ResetPasswordIn, ChangePasswordIn)
from ..deps import get_current_user 
from ..schemas import RegisterOut
import secrets
from time import time
from urllib.parse import quote

router = APIRouter(prefix="/auth", tags=["auth"])

# Rate limiters
resend_limiter_ip = SlidingWindowLimiter(limit=5, window=60*60)
forgot_ip_limiter    = SlidingWindowLimiter(limit=10, window=60 * 60)   # 10/hour per IP
forgot_email_limiter = SlidingWindowLimiter(limit=5,  window=60 * 60)   # 5/hour per email
login_limiter_ip    = SlidingWindowLimiter(limit=10, window=60)     # 10/min per IP
login_limiter_email = SlidingWindowLimiter(limit=5,  window=60)     # 5/min per email
register_limiter_ip = SlidingWindowLimiter(limit=5,  window=60*10)  # 5/10min per IP


def _norm_email(e: str) -> str:
    return e.strip().lower()

@router.post("/register", response_model=UserOut, status_code=201)
async def register(payload: RegisterIn, request: Request, background: BackgroundTasks):
    
    ip = request.client.host if request.client else "unknown"
    email_norm = _norm_email(payload.email)
    try:
        await register_limiter_ip.hit(f"login:ip:{ip}")
        await register_limiter_ip.hit(f"login:em:{email_norm}")
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests",
            headers={"Retry-After": str(e.retry_after)},
        )
    
    users = get_db()["users"]

    existing = await users.find_one({"email_norm": email_norm}, {"_id": 1})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    verification_nonce = secrets.token_hex(8)

    doc = {
        "email": payload.email,
        "email_norm": email_norm,
        "hashed_password": hash_password(payload.password),
        "password_history": [],
        "role": "user",
        "is_active": True,
        "email_verified": False,              # NEW
        "verification_nonce": verification_nonce,   # NEW
        "verification_sent_at": int(time()),        # NEW
        "schema_version": 1,
        "password_changed_at": int(time()),
        "reset_nonce": secrets.token_hex(8),
    }
    await users.insert_one(doc)

    # build verify link and send email
    token = create_verify_token(
        sub=doc["email"],
        jti=verification_nonce,
        hours=settings.verification_expire_hours,
    )
    verify_link = f"{settings.public_base_url}/auth/verify-email?token={quote(token, safe='')}"
    html = verification_email_html(verify_link)

    background.add_task(
        send_email,
        doc["email"],
        "Verify your email",
        html,
    )
    dev_echo = {}
    if settings.env.lower() == "dev":
        dev_echo = {
            "verify_token_dev_only": token,
            "verify_link_dev_only": verify_link,
        }
    return RegisterOut(
        email=doc["email"],
        role=doc["role"],
        is_active=doc["is_active"],
        **dev_echo,
    )


@router.post("/login", response_model=TokenPair)
async def login(payload: LoginIn, request: Request, response: Response):
    
    ip = request.client.host if request.client else "unknown"
    email_norm = _norm_email(payload.email)
    try:
        await login_limiter_ip.hit(f"login:ip:{ip}")
        await login_limiter_email.hit(f"login:em:{email_norm}")
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests",
            headers={"Retry-After": str(e.retry_after)},
        )
        
    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(payload.email)})
    if not user or not user.get("hashed_password") or not verify_password(payload.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    access = create_access_token(user["email"], user.get("role", "user"))
    refresh = create_refresh_token(user["email"], user.get("role", "user"))

    # set refresh token as HttpOnly cookie for session persistence
    max_age = settings.refresh_token_expire_days * 24 * 3600
    response.set_cookie(
        key="refresh_token",
        value=refresh,
        httponly=True,
        secure=False,  # True in prod with HTTPS
        samesite="lax",
        max_age=max_age,
    )

    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/guest", response_model=TokenPair, status_code=201)
async def guest_login():
    users = get_db()["users"]
    email = "guest@local"
    user = await users.find_one({"email": email})
    if not user:
        user = {
            "email": email,
            "email_norm": _norm_email(email),
            "hashed_password": None,
            "role": "guest",
            "is_active": True,
            "label": f"guest-{secrets.token_hex(4)}",
        }
        await users.insert_one(user)

    access = create_access_token(user["email"], user.get("role", "guest"))
    refresh = create_refresh_token(user["email"], user.get("role", "guest"))
    return TokenPair(access_token=access, refresh_token=refresh)

@router.post("/refresh", response_model=TokenPair)
async def refresh_token(refresh_token: str):
    data = decode_token(refresh_token)
    if not data or data.get("typ") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    email = data.get("sub")
    role = data.get("role") or "user"
    return TokenPair(
        access_token=create_access_token(email, role),
        refresh_token=create_refresh_token(email, role),
    )

@router.post("/token")
async def oauth2_token(username: str, password: str):
    """
    OAuth2 Password flow helper for Swagger: returns access token (no refresh).
    """
    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(username)})
    if not user or not user.get("hashed_password") or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access = create_access_token(user["email"], user.get("role", "user"))
    return {"access_token": access, "token_type": "bearer"}

@router.get("/me", response_model=UserOut)
async def me(access_token: str):
    data = decode_token(access_token)
    if not data or data.get("typ") != "access":
        raise HTTPException(status_code=401, detail="Invalid token")
    return UserOut(email=data["sub"], role=data.get("role","user"), is_active=True)

def _norm_email(e: str) -> str: return e.strip().lower()

@router.get("/users/me", response_model=UserOut)
async def users_me(current_user=Depends(get_current_user)):
    """Return the current authenticated user based on the Authorization header."""
    return UserOut(
        email=current_user["email"],
        role=current_user.get("role", "user"),
        is_active=current_user.get("is_active", True),
    )


# @router.post("/forgot-password")
# async def forgot_password(payload: ForgotPasswordIn, request: Request, response: Response):
#     users = get_db()["users"]
#     email_norm = _norm_email(payload.email)
#     ip = request.client.host if request.client else "unknown"

#     # --- RATE LIMIT CHECKS ---
#     try:
#         remaining_ip, win_ip = await forgot_ip_limiter.hit(f"fp:ip:{ip}")
#         remaining_em, win_em = await forgot_email_limiter.hit(f"fp:em:{email_norm}")
#         # Optional: expose rate info for clients (remove if you prefer)
#         response.headers["X-RateLimit-Remaining-IP"] = str(remaining_ip)
#         response.headers["X-RateLimit-Window-IP"] = str(win_ip)
#         response.headers["X-RateLimit-Remaining-Email"] = str(remaining_em)
#         response.headers["X-RateLimit-Window-Email"] = str(win_em)
#     except RateLimitExceeded as e:
#         # 429 with Retry-After header
#         raise HTTPException(status_code=429, detail="Too Many Requests", headers={"Retry-After": str(e.retry_after)})

#     # --- NORMAL HANDLER (no user enumeration) ---
#     user = await users.find_one({"email_norm": email_norm})
#     if user:
#         # rotate & persist nonce, then mint reset token that matches it
#         new_nonce = secrets.token_hex(8)
#         await users.update_one({"_id": user["_id"]}, {"$set": {"reset_nonce": new_nonce}})
#         token = create_reset_token(sub=user["email"], jti=new_nonce)
#         # DEV ONLY: return the token so you can test in Swagger
#         return {"ok": True, "reset_token_dev_only": token}

# already imported: create_reset_token, settings, reset_password_email_html
@router.post("/forgot-password")
async def forgot_password(
    payload: ForgotPasswordIn,
    request: Request,
    response: Response,
    background: BackgroundTasks,
):
    users = get_db()["users"]
    email_norm = _norm_email(payload.email)
    ip = request.client.host if request.client else "unknown"

    # --- RATE LIMIT CHECKS (unchanged) ---
    try:
        remaining_ip, win_ip = await forgot_ip_limiter.hit(f"fp:ip:{ip}")
        remaining_em, win_em = await forgot_email_limiter.hit(f"fp:em:{email_norm}")
        response.headers["X-RateLimit-Remaining-IP"] = str(remaining_ip)
        response.headers["X-RateLimit-Window-IP"] = str(win_ip)
        response.headers["X-RateLimit-Remaining-Email"] = str(remaining_em)
        response.headers["X-RateLimit-Window-Email"] = str(win_em)
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests",
            headers={"Retry-After": str(e.retry_after)},
        )

    # --- NORMAL HANDLER (no user enumeration) ---
    user = await users.find_one({"email_norm": email_norm})
    if user:
        # rotate & persist nonce, then mint reset token that matches it
        new_nonce = secrets.token_hex(8)
        await users.update_one(
            {"_id": user["_id"]},
            {"$set": {"reset_nonce": new_nonce}},
        )
        token = create_reset_token(sub=user["email"], jti=new_nonce)

        # Build reset link to the FRONTEND. E.g. /reset?token=...
        reset_link = f"{settings.public_base_url}/reset?token={quote(token, safe='')}"

        html = reset_password_email_html(reset_link)
        background.add_task(
            send_email,
            user["email"],
            "Reset your password",
            html,
        )

        dev_echo = {"reset_token_dev_only": token} if settings.env.lower() == "dev" else {}
        return {"ok": True, **dev_echo}

    # No leak about whether the email exists
    return {"ok": True}


#     return {"ok": True}

@router.post("/reset-password")
async def reset_password(payload: ResetPasswordIn):
    data = decode_token(payload.token)
    if not data or data.get("typ") != "reset":
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    email, jti = data.get("sub"), data.get("jti")
    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(email)})
    if not user or user.get("reset_nonce") != jti:
        raise HTTPException(status_code=400, detail="Invalid reset token")

    # --- reuse check against current + history ---
    history_hashes = [user.get("hashed_password")] + list(user.get("password_history", []))
    if password_reused(payload.new_password, history_hashes):
        raise HTTPException(status_code=400, detail="New password was used recently")

    # --- rotate password & history ---
    new_hash = hash_password(payload.new_password)
    new_nonce = secrets.token_hex(8)
    history = [user.get("hashed_password")] + list(user.get("password_history", []))
    # trim to size-1 because we’re about to store new current hash
    history = history[: max(0, settings.password_history_size - 1)]

    await users.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "hashed_password": new_hash,
            "password_history": history,
            "password_changed_at": int(time()),
            "reset_nonce": new_nonce
        }}
    )
    return {"ok": True}

@router.post("/change-password")
async def change_password(payload: ChangePasswordIn, current_user=Depends(get_current_user)):
    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(current_user["email"])})
    if not user or not user.get("hashed_password"):
        raise HTTPException(status_code=400, detail="Cannot change password for this account")

    if not verify_password(payload.current_password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect current password")

    # --- reuse check against current + history ---
    history_hashes = [user["hashed_password"]] + list(user.get("password_history", []))
    if password_reused(payload.new_password, history_hashes):
        raise HTTPException(status_code=400, detail="New password was used recently")

    # --- rotate password & history ---
    new_hash = hash_password(payload.new_password)
    history = [user["hashed_password"]] + list(user.get("password_history", []))
    history = history[: max(0, settings.password_history_size - 1)]

    await users.update_one(
        {"_id": user["_id"]},
        {"$set": {
            "hashed_password": new_hash,
            "password_history": history,
            "password_changed_at": int(time())
        }}
    )
    return {"ok": True}


@router.get("/session-me")
async def session_me(request: Request):
    """Restore session from refresh_token cookie and return user + new access token."""
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No session cookie")

    data = decode_token(refresh_token)
    if not data or data.get("typ") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    email = data.get("sub")
    role = data.get("role") or "user"

    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(email)})
    if not user or not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="User not active")

    # Optional: also require email_verified here if you like
    # if not user.get("email_verified"):
    #     raise HTTPException(status_code=403, detail="Email not verified")

    # Mint a fresh access token
    new_access = create_access_token(email, role)

    return {
        "access_token": new_access,
        "user": {
            "email": email,
            "role": role,
            "is_active": user.get("is_active", True),
        },
    }

@router.post("/session-logout")
async def session_logout(response: Response):
    # Clear the refresh cookie
    response.delete_cookie("refresh_token")
    return {"ok": True}

@router.get("/verify-email")
async def verify_email(token: str):
    data = decode_token(token)
    if not data or data.get("typ") != "verify":
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")

    email, jti = data.get("sub"), data.get("jti")
    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(email)})
    if not user or user.get("verification_nonce") != jti:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    await users.update_one(
        {"_id": user["_id"]},
        {"$set": {"email_verified": True, "verification_nonce": secrets.token_hex(8)}},  # rotate
    )
    return {"ok": True, "email": email, "verified": True}

@router.post("/resend-verification")
async def resend_verification(request: Request, email: str):
    # rate limit per-IP
    ip = request.client.host if request.client else "unknown"
    try:
        await resend_limiter_ip.hit(f"rv:ip:{ip}")
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail="Too Many Requests", headers={"Retry-After": str(e.retry_after)})

    users = get_db()["users"]
    user = await users.find_one({"email_norm": _norm_email(email)})
    if not user:
        return {"ok": True}  # do not leak existence
    if user.get("email_verified"):
        return {"ok": True, "already_verified": True}

    new_nonce = secrets.token_hex(8)
    await users.update_one({"_id": user["_id"]}, {"$set": {"verification_nonce": new_nonce, "verification_sent_at": int(time())}})
    token = create_verify_token(sub=user["email"], jti=new_nonce, hours=settings.verification_expire_hours)
    verify_link = f"{settings.public_base_url}/auth/verify-email?token={token}"
    html = verification_email_html(verify_link)
    send_email("Verify your email", user["email"], html, f"Verify: {verify_link}")

    dev_echo = {"verify_token_dev_only": token} if settings.env.lower() == "dev" else {}
    return {"ok": True, **dev_echo}