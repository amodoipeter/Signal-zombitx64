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
    
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    @validator("SUPABASE_URL", "SUPABASE_KEY")
    def validate_supabase_config(cls, v: str) -> str:
        if not v:
            raise ValueError("Supabase URL and Key must be set")
        return v
    
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
