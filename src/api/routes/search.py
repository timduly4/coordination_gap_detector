"""
Search API endpoints for semantic search across messages.

This module provides REST endpoints for searching messages using
semantic similarity and filtering by various criteria.
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db, get_vector_db
from src.db.vector_store import VectorStore
from src.models.schemas import ErrorResponse, SearchRequest, SearchResponse
from src.services.search_service import SearchService, get_search_service

logger = logging.getLogger(__name__)

router = APIRouter()


def get_service(vector_store: VectorStore = Depends(get_vector_db)) -> SearchService:
    """
    Dependency injection for SearchService.

    Args:
        vector_store: VectorStore instance from dependency injection

    Returns:
        SearchService instance
    """
    return get_search_service(vector_store)


@router.post(
    "/",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search messages",
    description="Perform semantic search across all messages using vector similarity",
    responses={
        200: {
            "description": "Search completed successfully",
            "model": SearchResponse,
        },
        400: {
            "description": "Invalid request parameters",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)
async def search_messages(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    service: SearchService = Depends(get_service),
) -> SearchResponse:
    """
    Search for messages using semantic similarity.

    This endpoint performs semantic search using vector embeddings to find
    messages similar to the query. Results can be filtered by source type,
    channel, and date range.

    **Parameters:**
    - **query**: Search query text (1-1000 characters)
    - **limit**: Maximum number of results (1-100, default: 10)
    - **threshold**: Minimum similarity score 0.0-1.0 (default: 0.0)
    - **source_types**: Optional list of source types to filter by
    - **channels**: Optional list of channels to filter by
    - **date_from**: Optional start date for filtering
    - **date_to**: Optional end date for filtering

    **Returns:**
    - List of matching messages with similarity scores
    - Total result count
    - Query execution time

    **Example Request:**
    ```json
    {
        "query": "OAuth implementation",
        "limit": 5,
        "threshold": 0.7,
        "source_types": ["slack"],
        "channels": ["#engineering"]
    }
    ```

    **Example Response:**
    ```json
    {
        "results": [
            {
                "content": "Starting OAuth2 integration...",
                "source": "slack",
                "channel": "#engineering",
                "author": "alice@demo.com",
                "timestamp": "2024-12-01T09:00:00Z",
                "score": 0.89,
                "message_id": 123,
                "external_id": "slack_msg_abc",
                "message_metadata": {}
            }
        ],
        "total": 1,
        "query": "OAuth implementation",
        "query_time_ms": 45,
        "threshold": 0.7
    }
    ```
    """
    try:
        logger.info(f"Received search request: query='{request.query}', limit={request.limit}")

        # Execute search using the service
        response = await service.search(request, db)

        logger.info(
            f"Search completed successfully: {response.total} results in {response.query_time_ms}ms"
        )

        return response

    except ValueError as e:
        # Validation errors
        logger.warning(f"Invalid search request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Search request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your search request",
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Search service health check",
    description="Check the health status of the search service and its dependencies",
)
async def search_health_check(
    service: SearchService = Depends(get_service),
) -> Dict[str, Any]:
    """
    Check the health of the search service.

    Returns status information about:
    - Vector store connection
    - Document count in collection
    - Service availability

    **Example Response:**
    ```json
    {
        "status": "healthy",
        "vector_store": {
            "connected": true,
            "collection": "coordination_messages",
            "document_count": 24
        }
    }
    ```
    """
    try:
        health_status = await service.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed",
        )
