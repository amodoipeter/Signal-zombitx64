"""
API Routes module for the AI Signal Provider.
Contains all the API endpoints for the application.
"""

from .auth_routes import router as auth_router
from .signals_routes import router as signals_router
from .subscriptions_routes import router as subscriptions_router
from .users_routes import router as users_router

__all__ = ["auth_router", "signals_router", "subscriptions_router", "users_router"]
