"""
Main FastAPI application for coordination gap detection.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.
    """
    settings = get_settings()
    logger.info(f"Starting coordination gap detector in {settings.environment} mode")

    # Startup: Initialize connections, load models, etc.
    logger.info("Initializing database connections...")
    logger.info("Loading detection models...")

    yield

    # Shutdown: Close connections, cleanup resources
    logger.info("Shutting down coordination gap detector")
    logger.info("Closing database connections...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title="Coordination Gap Detector",
        description="AI-powered coordination gap detection system for identifying failures across enterprise communication channels",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Basic health check endpoint."""
        return {"status": "healthy", "environment": settings.environment}

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "message": "Coordination Gap Detector API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    # TODO: Include routers from api.routes
    # app.include_router(gaps.router, prefix="/api/v1/gaps", tags=["gaps"])
    # app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
    # app.include_router(insights.router, prefix="/api/v1/insights", tags=["insights"])
    # app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
    )
