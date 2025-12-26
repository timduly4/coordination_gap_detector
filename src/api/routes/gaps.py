"""
Gap detection API endpoints.

This module provides RESTful endpoints for detecting and managing
coordination gaps across teams and communication channels.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db, get_vector_store
from src.db.vector_store import VectorStore
from src.models.schemas import (
    CoordinationGap,
    ErrorResponse,
    GapDetectionRequest,
    GapDetectionResponse,
    GapListRequest,
    GapListResponse,
)
from src.services.detection_service import GapDetectionService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_detection_service(
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
) -> GapDetectionService:
    """
    Dependency for gap detection service.

    Args:
        db: Database session
        vector_store: Vector store instance

    Returns:
        GapDetectionService instance
    """
    return GapDetectionService(vector_store=vector_store, db_session=db)


@router.post(
    "/detect",
    response_model=GapDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect coordination gaps",
    description="Run gap detection across specified timeframe and sources",
    responses={
        200: {"description": "Gaps detected successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable (e.g., LLM rate limit)"},
    },
)
async def detect_gaps(
    request: GapDetectionRequest,
    service: GapDetectionService = Depends(get_detection_service),
) -> GapDetectionResponse:
    """
    Detect coordination gaps across teams and communication channels.

    This endpoint analyzes messages from specified sources over a timeframe
    to identify coordination failures such as duplicate work, missing context,
    or knowledge silos.

    **Example Request:**
    ```json
    {
      "timeframe_days": 30,
      "sources": ["slack", "github"],
      "gap_types": ["duplicate_work"],
      "teams": ["engineering"],
      "min_impact_score": 0.6,
      "include_evidence": true
    }
    ```

    **Detection Process:**
    1. Retrieve messages from specified sources
    2. Generate/retrieve embeddings for semantic analysis
    3. Run gap detection algorithms (clustering, LLM verification)
    4. Filter by impact score threshold
    5. Return detected gaps with evidence and recommendations

    Args:
        request: Gap detection request parameters
        service: Gap detection service (injected)

    Returns:
        GapDetectionResponse with detected gaps and metadata

    Raises:
        HTTPException: 400 for invalid parameters, 500 for server errors
    """
    try:
        logger.info(
            f"Gap detection request: timeframe={request.timeframe_days}d, "
            f"sources={request.sources}, gap_types={request.gap_types}"
        )

        # Run gap detection
        response = await service.detect_gaps(request)

        logger.info(
            f"Gap detection complete: {response.metadata.total_gaps} gaps detected "
            f"({response.metadata.critical_gaps} critical, {response.metadata.high_gaps} high) "
            f"in {response.metadata.detection_time_ms}ms"
        )

        return response

    except ValueError as e:
        # Invalid parameters (e.g., unknown gap type)
        logger.warning(f"Invalid gap detection request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Gap detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gap detection failed. Please try again later.",
        )


@router.get(
    "",
    response_model=GapListResponse,
    status_code=status.HTTP_200_OK,
    summary="List detected gaps",
    description="List all detected gaps with filtering and pagination",
    responses={
        200: {"description": "Gaps retrieved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_gaps(
    gap_type: Optional[str] = Query(None, description="Filter by gap type"),
    min_impact_score: float = Query(
        0.0, ge=0.0, le=1.0, description="Minimum impact score"
    ),
    teams: Optional[List[str]] = Query(None, description="Filter by teams"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Results per page"),
    service: GapDetectionService = Depends(get_detection_service),
) -> GapListResponse:
    """
    List detected coordination gaps with filtering and pagination.

    Supports filtering by:
    - Gap type (duplicate_work, missing_context, etc.)
    - Minimum impact score
    - Teams involved
    - Date range (coming soon)

    **Example Query:**
    ```
    GET /api/v1/gaps?gap_type=duplicate_work&min_impact_score=0.7&limit=10
    ```

    **Pagination:**
    - `page`: Page number (default: 1)
    - `limit`: Results per page (default: 10, max: 100)
    - `has_more`: Boolean indicating if more results exist

    Args:
        gap_type: Filter by specific gap type
        min_impact_score: Minimum impact score (0-1)
        teams: Filter by teams involved
        page: Page number (1-indexed)
        limit: Results per page (max 100)
        service: Gap detection service (injected)

    Returns:
        GapListResponse with filtered gaps and pagination info

    Raises:
        HTTPException: 400 for invalid parameters, 500 for server errors
    """
    try:
        logger.info(
            f"List gaps request: gap_type={gap_type}, min_score={min_impact_score}, "
            f"teams={teams}, page={page}, limit={limit}"
        )

        # Create filter request
        list_request = GapListRequest(
            gap_type=gap_type,
            min_impact_score=min_impact_score,
            teams=teams,
            page=page,
            limit=limit,
        )

        # Retrieve gaps
        response = await service.list_gaps(list_request)

        logger.info(
            f"Retrieved {len(response.gaps)} gaps (total={response.total}, "
            f"page={response.page}, has_more={response.has_more})"
        )

        return response

    except ValueError as e:
        # Invalid filter parameters
        logger.warning(f"Invalid list gaps request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"List gaps failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve gaps. Please try again later.",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Gap detection service health check",
    description="Check the health of gap detection service and dependencies",
)
async def health_check(
    service: GapDetectionService = Depends(get_detection_service),
) -> dict:
    """
    Check health of gap detection service.

    Returns status of:
    - Gap detection service
    - Vector store connection
    - Database connection
    - Available detectors

    Returns:
        Health status dictionary
    """
    try:
        health_status = await service.health_check()
        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get(
    "/{gap_id}",
    response_model=CoordinationGap,
    status_code=status.HTTP_200_OK,
    summary="Get gap by ID",
    description="Retrieve detailed information about a specific coordination gap",
    responses={
        200: {"description": "Gap retrieved successfully"},
        404: {"model": ErrorResponse, "description": "Gap not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_gap(
    gap_id: str,
    service: GapDetectionService = Depends(get_detection_service),
) -> CoordinationGap:
    """
    Get detailed information about a specific coordination gap.

    Retrieves the full gap object including:
    - Evidence items with relevance scores
    - Temporal overlap analysis
    - LLM verification results
    - Impact breakdown
    - Cost estimation
    - Recommendations

    **Example:**
    ```
    GET /api/v1/gaps/gap_abc123
    ```

    Args:
        gap_id: Unique gap identifier
        service: Gap detection service (injected)

    Returns:
        CoordinationGap object with full details

    Raises:
        HTTPException: 404 if gap not found, 500 for server errors
    """
    try:
        logger.info(f"Get gap request: gap_id={gap_id}")

        # Retrieve gap
        gap = await service.get_gap_by_id(gap_id)

        if gap is None:
            logger.warning(f"Gap not found: {gap_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gap with ID '{gap_id}' not found",
            )

        logger.info(f"Retrieved gap: {gap_id} ({gap.type}, impact={gap.impact_score})")
        return gap

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Unexpected errors
        logger.error(f"Get gap failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve gap. Please try again later.",
        )
