#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application entry point for the AI Signal Provider + Telegram Bot.
"""

import logging
import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Signal-ZombitX64",
    description="AI Trading Signal Provider with Telegram/Discord Integration",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import database initialization
from app.core.database import init_db

# Import routers
from app.api.routes import (
    auth_router,
    signals_router,
    subscriptions_router,
    users_router,
    reports_router,
    redeem_router,  # New import
)

# Import services
from app.services.telegram_bot.bot import telegram_bot_service
from app.services.discord.bot import discord_bot_service
from app.services.scheduler.signal_scheduler import signal_scheduler

# Register routers
app.include_router(auth_router.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(signals_router.router, prefix="/api/signals", tags=["Trading Signals"])
app.include_router(subscriptions_router.router, prefix="/api/subscriptions", tags=["Subscriptions"])
app.include_router(users_router.router, prefix="/api/users", tags=["Users"])
app.include_router(reports_router.router, prefix="/api/reports", tags=["Performance Reports"])
app.include_router(redeem_router.router, prefix="/api/redeem", tags=["Redeem Codes"])  # New router

@app.on_event("startup")
async def startup_event():
    """Initialize database and start services."""
    logger.info("Starting application...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Start Telegram bot
    try:
        await telegram_bot_service.setup()
        logger.info("Telegram bot initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram bot: {str(e)}")
    
    # Start scheduler
    try:
        await signal_scheduler.start()
        logger.info("Signal scheduler started")
    except Exception as e:
        logger.error(f"Failed to start signal scheduler: {str(e)}")
    
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop services."""
    logger.info("Shutting down application...")
    
    # Stop scheduler
    try:
        await signal_scheduler.stop()
        logger.info("Signal scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping signal scheduler: {str(e)}")
    
    logger.info("Application shutdown complete")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "status": "online",
        "message": "ZombitX64 AI Signal Provider API",
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn

    # Start the application with uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true",
    )
