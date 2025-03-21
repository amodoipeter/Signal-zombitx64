#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application entry point for the AI Signal Provider + Telegram Bot.
"""

import logging
import os
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import asyncio
from typing import Optional

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

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# Discord and Telegram bot tasks
discord_task: Optional[asyncio.Task] = None
telegram_task: Optional[asyncio.Task] = None

@app.on_event("startup")
async def startup_event():
    """Initialize database and start services."""
    logger.info("Starting application...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Start Telegram bot only
    global telegram_task
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        telegram_task = asyncio.create_task(telegram_bot_service.start_polling())
        logger.info("Telegram bot started")
    else:
        logger.warning("TELEGRAM_BOT_TOKEN not set, Telegram bot not started")
    
    # Start Discord bot in a separate task if configured
    global discord_task
    if os.getenv("DISCORD_BOT_TOKEN"):
        discord_task = asyncio.create_task(start_discord_bot())
        logger.info("Discord bot task created")
    else:
        logger.warning("DISCORD_BOT_TOKEN not set, Discord bot not started")
    
    # Start scheduler
    try:
        await signal_scheduler.start()
        logger.info("Signal scheduler started")
    except Exception as e:
        logger.error(f"Failed to start signal scheduler: {str(e)}")
    
    logger.info("Application startup complete")

async def start_discord_bot():
    """Start Discord bot in an exception-safe manner."""
    try:
        logger.info("Starting Discord bot")
        await discord_bot_service.start()
    except Exception as e:
        logger.error(f"Error starting Discord bot: {str(e)}")

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
    
    # Stop Telegram bot
    if telegram_task:
        await telegram_bot_service.stop()
        telegram_task.cancel()
        try:
            await telegram_task
        except asyncio.CancelledError:
            pass
    
    # Stop Discord bot
    if discord_task:
        await discord_bot_service.close()
        discord_task.cancel()
        try:
            await discord_task
        except asyncio.CancelledError:
            pass
    
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
