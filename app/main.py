# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .settings import settings
from .db import ping, ensure_indexes
from .routers import auth, auth_cookie, index, gcs_test

def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, debug=(settings.env.lower() == "dev"))

    # CORS mainly matters if you run the backend standalone on :9000.
    # When attached via BACKEND_APP, frontend+backend are same-origin and this is mostly irrelevant.
    if settings.env.lower() == "dev":
        origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    else:
        # TODO: change to your real frontend domain when you deploy
        origins = ["https://your-frontend-domain.com"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(auth.router)
    app.include_router(auth_cookie.router)
    app.include_router(index.router)
    app.include_router(gcs_test.router)

    @app.on_event("startup")
    async def on_startup():
        if not await ping():
            raise RuntimeError("MongoDB unreachable")
        await ensure_indexes()

    @app.get("/")
    def root():
        return {"status": "ok", "app": settings.app_name, "env": settings.env}

    return app

# For uvicorn/standalone use.
app = create_app()