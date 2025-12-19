"""
Evaluation API endpoints for ranking quality assessment.

This module provides REST endpoints for evaluating and comparing
ranking strategies using test queries and relevance judgments.
"""
import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db, get_vector_db
from src.db.vector_store import VectorStore
from src.models.schemas import (
    AddJudgmentRequest,
    AddJudgmentResponse,
    ErrorResponse,
    EvaluateStrategyRequest,
    EvaluationComparisonResponse,
    EvaluationStatisticsResponse,
)
from src.services.evaluation_service import EvaluationService
from src.services.search_service import SearchService, get_search_service

logger = logging.getLogger(__name__)

router = APIRouter()


def get_evaluation_service(
    vector_store: VectorStore = Depends(get_vector_db)
) -> EvaluationService:
    """
    Dependency injection for EvaluationService.

    Args:
        vector_store: VectorStore instance from dependency injection

    Returns:
        EvaluationService instance
    """
    search_service = get_search_service(vector_store)
    return EvaluationService(search_service=search_service)


@router.post(
    "/evaluate",
    response_model=EvaluationComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate ranking strategies",
    description="Evaluate and compare multiple ranking strategies using test queries",
    responses={
        200: {
            "description": "Evaluation completed successfully",
            "model": EvaluationComparisonResponse,
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
async def evaluate_strategies(
    request: EvaluateStrategyRequest,
    db: AsyncSession = Depends(get_db),
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationComparisonResponse:
    """
    Evaluate and compare multiple ranking strategies.

    This endpoint runs offline evaluation using test queries and relevance
    judgments to calculate ranking metrics (MRR, NDCG@k, Precision@k, Recall@k)
    for each strategy.

    **Note:** Relevance judgments must be loaded separately via the
    `/api/v1/evaluation/judgments` endpoint before running evaluation.

    **Parameters:**
    - **test_queries**: List of queries with query_id and query_text
    - **strategies**: List of strategy names (semantic, bm25, hybrid_rrf, hybrid_weighted)
    - **k**: Cutoff for @k metrics (default: 10)

    **Returns:**
    - Evaluation results with metrics for each strategy
    - Best performing strategy
    - Percentage improvement over baseline

    **Example Request:**
    ```json
    {
        "test_queries": [
            {"query_id": "q1", "query_text": "OAuth implementation"},
            {"query_id": "q2", "query_text": "authentication flow"}
        ],
        "strategies": ["semantic", "bm25", "hybrid_rrf"],
        "k": 10
    }
    ```

    **Example Response:**
    ```json
    {
        "strategies": ["semantic", "bm25", "hybrid_rrf"],
        "num_queries": 2,
        "k": 10,
        "results": {
            "mrr": {
                "semantic": 0.68,
                "bm25": 0.62,
                "hybrid_rrf": 0.74
            },
            "ndcg": {
                "semantic": 0.72,
                "bm25": 0.65,
                "hybrid_rrf": 0.79
            }
        },
        "best_strategy": "hybrid_rrf",
        "improvement_over_baseline": 8.8
    }
    ```
    """
    try:
        logger.info(
            f"Received evaluation request: {len(request.test_queries)} queries, "
            f"{len(request.strategies)} strategies"
        )

        # Run comparison
        comparison = await service.compare_strategies(
            strategies=request.strategies,
            test_queries=request.test_queries,
            db=db,
            k=request.k
        )

        logger.info(
            f"Evaluation completed: best={comparison['best_strategy']}, "
            f"improvement={comparison['improvement_over_baseline']:.1f}%"
        )

        return EvaluationComparisonResponse(**comparison)

    except ValueError as e:
        logger.warning(f"Invalid evaluation request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        logger.error(f"Evaluation request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your evaluation request",
        )


@router.post(
    "/judgments",
    response_model=AddJudgmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add relevance judgment",
    description="Add a single relevance judgment for a query-document pair",
)
async def add_judgment(
    request: AddJudgmentRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> AddJudgmentResponse:
    """
    Add a relevance judgment.

    Relevance judgments are used to evaluate ranking quality. Each judgment
    specifies how relevant a document is for a given query.

    **Relevance Scale:**
    - 3: Highly relevant (perfect match)
    - 2: Relevant (good match)
    - 1: Partially relevant (somewhat related)
    - 0: Not relevant

    **Example Request:**
    ```json
    {
        "query_id": "q1",
        "document_id": "123",
        "relevance": 3,
        "query_text": "OAuth implementation"
    }
    ```
    """
    try:
        service.add_judgment(
            query_id=request.query_id,
            document_id=request.document_id,
            relevance=request.relevance,
            query_text=request.query_text
        )

        logger.info(
            f"Added judgment: query={request.query_id}, "
            f"doc={request.document_id}, rel={request.relevance}"
        )

        return AddJudgmentResponse(
            status="success",
            message=f"Judgment added for query {request.query_id}"
        )

    except ValueError as e:
        logger.warning(f"Invalid judgment: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        logger.error(f"Failed to add judgment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while adding the judgment",
        )


@router.get(
    "/statistics",
    response_model=EvaluationStatisticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get evaluation statistics",
    description="Get statistics about loaded queries and judgments",
)
async def get_statistics(
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationStatisticsResponse:
    """
    Get statistics about loaded relevance judgments.

    Returns:
    - Number of queries with judgments
    - Total number of judgments

    **Example Response:**
    ```json
    {
        "num_queries": 50,
        "num_judgments": 412
    }
    ```
    """
    try:
        stats = service.get_statistics()
        return EvaluationStatisticsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving statistics",
        )
