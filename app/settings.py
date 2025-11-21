import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # --- Pydantic v2 settings config ---
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # so unknown env vars won't crash startup
    )

    app_name: str = os.getenv("APP_NAME", "FastAPI Backend")
    env: str = Field(default="dev", alias="ENV")

    # Auth
    jwt_secret: str = Field(..., alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRE_DAYS")
    password_history_size: int = Field(default=5, alias="PASSWORD_HISTORY_SIZE")

    # Mongo
    mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "")
    #mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_uri_local: str = os.getenv("MONGODB_URI_LOCAL", "")
    mongodb_uri_web: str = os.getenv("MONGODB_URI_WEB", "")
    @property
    def mongodb_uri(self) -> str:
        """Choose the local or remote Mongo URI based on environment."""
        if self.env.lower() in {"dev", "local", "test"}:
            return self.mongodb_uri_web
        return self.mongodb_uri_web
    
    # --- NEW: email verification / SMTP ---
    require_email_verification: bool = Field(default=True, alias="REQUIRE_EMAIL_VERIFICATION")
    verification_expire_hours: int = Field(default=24, alias="VERIFICATION_EXPIRE_HOURS")
    public_base_url: str = Field(default="http://127.0.0.1:8000", alias="PUBLIC_BASE_URL")

    smtp_host: str = Field(default="localhost", alias="SMTP_HOST")
    smtp_port: int = Field(default=1025, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_tls: bool = Field(default=False, alias="SMTP_TLS")
    mail_from: str = Field(default="simon.c.kirby@gmail.com", alias="MAIL_FROM")

settings = Settings()

