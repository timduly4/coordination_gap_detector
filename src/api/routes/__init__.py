"""
API route modules.

This package contains all API endpoint routers organized by functionality.
"""

from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.gaps import router as gaps_router
from src.api.routes.search import router as search_router

__all__ = ["search_router", "evaluation_router", "gaps_router"]
