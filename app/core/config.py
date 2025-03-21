import os
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, PostgresDsn, validator
import secrets

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ZombitX64 Trading Signals"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))  # 7 days
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    
    # Database
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "signal_zombitx64")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    DATABASE_URI: Optional[PostgresDsn] = None

    @validator("DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    # Redis
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    
    # API Keys
    BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET: Optional[str] = os.getenv("BINANCE_API_SECRET")
    
    # Telegram Bot
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_PREMIUM_GROUP_IDS: Optional[str] = os.getenv("TELEGRAM_PREMIUM_GROUP_IDS")
    
    # Discord Bot Settings - Added and improved
    DISCORD_BOT_TOKEN: Optional[str] = os.getenv("DISCORD_BOT_TOKEN")
    DISCORD_GUILD_ID: Optional[str] = os.getenv("DISCORD_GUILD_ID")
    DISCORD_PREMIUM_ROLE_IDS: Optional[str] = os.getenv("DISCORD_PREMIUM_ROLE_IDS")
    DISCORD_FREE_ROLE_ID: Optional[str] = os.getenv("DISCORD_FREE_ROLE_ID")
    DISCORD_SIGNAL_CHANNEL_ID: Optional[str] = os.getenv("DISCORD_SIGNAL_CHANNEL_ID")
    DISCORD_ADMIN_CHANNEL_ID: Optional[str] = os.getenv("DISCORD_ADMIN_CHANNEL_ID")
    DISCORD_COMMAND_PREFIX: str = os.getenv("DISCORD_COMMAND_PREFIX", "/")
    
    # Stripe
    STRIPE_API_KEY: Optional[str] = os.getenv("STRIPE_API_KEY")
    STRIPE_WEBHOOK_SECRET: Optional[str] = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    # Frontend
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # Trading settings
    SIGNAL_CONFIDENCE_THRESHOLD: float = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", 75.0))
    SIGNAL_FREQUENCY: int = int(os.getenv("SIGNAL_FREQUENCY", 3600))  # seconds
    
    # AI Model Type
    AI_MODEL_TYPE: str = os.getenv("AI_MODEL_TYPE", "ensemble")  # lstm, gru, random_forest, ensemble
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
