#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application entry point for the AI Signal Provider + Telegram Bot.
"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    title="AI Signal Provider",
    description="AI-powered trading signal provider with Telegram/Discord bot integration",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from app.api.routes import (
    auth_router,
    signals_router,
    subscriptions_router,
    users_router,
)

# Register routers
app.include_router(auth_router, prefix="/api", tags=["Authentication"])
app.include_router(signals_router, prefix="/api", tags=["Signals"])
app.include_router(subscriptions_router, prefix="/api", tags=["Subscriptions"])
app.include_router(users_router, prefix="/api", tags=["Users"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "name": "AI Signal Provider API",
        "version": "0.1.0",
        "status": "running",
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
