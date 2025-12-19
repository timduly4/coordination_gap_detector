"""
Main FastAPI application for coordination gap detection.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.db.elasticsearch import get_es_client
from src.db.postgres import check_db_connection
from src.db.vector_store import get_vector_store

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

    # Health check endpoints
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Basic health check endpoint."""
        return {"status": "healthy", "environment": settings.environment}

    @app.get("/health/detailed")
    async def detailed_health_check() -> dict[str, Any]:
        """
        Detailed health check with service connectivity status.
        Checks: Postgres, Redis, ChromaDB, Elasticsearch
        """
        health_status = {
            "status": "healthy",
            "environment": settings.environment,
            "services": {}
        }

        # Check Postgres
        try:
            pg_connected = await check_db_connection()
            health_status["services"]["postgres"] = {
                "status": "connected" if pg_connected else "disconnected",
                "url": settings.postgres_url.split("@")[-1] if "@" in settings.postgres_url else "not_set"
            }
        except Exception as e:
            health_status["services"]["postgres"] = {"status": "error", "message": str(e)}

        # Check Redis (TODO: implement Redis connection check in Milestone 1D+)
        try:
            health_status["services"]["redis"] = {
                "status": "not_implemented",
                "url": settings.redis_url
            }
        except Exception as e:
            health_status["services"]["redis"] = {"status": "error", "message": str(e)}

        # Check ChromaDB
        try:
            vector_store = get_vector_store()
            chroma_connected = vector_store.check_connection()
            doc_count = vector_store.get_collection_count()
            health_status["services"]["chromadb"] = {
                "status": "connected" if chroma_connected else "disconnected",
                "collection": vector_store.collection_name,
                "document_count": doc_count,
                "persist_dir": settings.chroma_persist_dir
            }
        except Exception as e:
            health_status["services"]["chromadb"] = {"status": "error", "message": str(e)}

        # Check Elasticsearch
        try:
            es_client = get_es_client()
            es_connected = es_client.check_connection()
            cluster_health = es_client.get_cluster_health()
            health_status["services"]["elasticsearch"] = {
                "status": "connected" if es_connected else "disconnected",
                "url": settings.elasticsearch_url,
                "cluster_status": cluster_health.get("status", "unknown"),
                "cluster_name": cluster_health.get("cluster_name", "unknown"),
            }
        except Exception as e:
            health_status["services"]["elasticsearch"] = {"status": "error", "message": str(e)}

        # Overall status
        service_statuses = [svc.get("status") for svc in health_status["services"].values()]
        if "error" in service_statuses or "disconnected" in service_statuses:
            health_status["status"] = "degraded"

        return health_status

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "message": "Coordination Gap Detector API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    # Include API routers
    from src.api.routes import search, evaluation
    app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
    app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["evaluation"])

    # TODO: Add additional routers as they are implemented
    # app.include_router(gaps.router, prefix="/api/v1/gaps", tags=["gaps"])
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
