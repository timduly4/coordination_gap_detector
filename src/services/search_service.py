"""
Search service implementing business logic for semantic search operations.

This service coordinates between the API layer and the database/vector store,
implementing search logic, filtering, and result enrichment.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Message, Source
from src.db.vector_store import VectorStore
from src.models.schemas import SearchRequest, SearchResponse, SearchResultItem
from src.search.hybrid_search import HybridSearchFusion, deduplicate_results
from src.search.query_parser import QueryParser
from src.search.retrieval import KeywordRetriever

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service class for handling search operations.

    This class implements the business logic for:
    - Semantic search using vector store
    - Keyword search using BM25/Elasticsearch
    - Hybrid search with multiple fusion strategies
    - Result enrichment with database data
    - Filtering by source, channel, date range
    - Result formatting and validation
    """

    def __init__(self, vector_store: VectorStore, keyword_retriever: Optional[KeywordRetriever] = None):
        """
        Initialize the search service.

        Args:
            vector_store: VectorStore instance for semantic search
            keyword_retriever: KeywordRetriever instance for BM25 search (optional)
        """
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever or KeywordRetriever()
        self.query_parser = QueryParser()
        logger.info("SearchService initialized with semantic and keyword search capabilities")

    async def search(
        self,
        request: SearchRequest,
        db: AsyncSession
    ) -> SearchResponse:
        """
        Execute a search query using the specified ranking strategy.

        Supports multiple strategies:
        - auto: Automatically detect query intent and choose best strategy
        - semantic: Pure semantic/vector search
        - bm25: Pure keyword search with BM25
        - hybrid_rrf: Reciprocal Rank Fusion of semantic + BM25
        - hybrid_weighted: Weighted combination of semantic + BM25 scores

        Args:
            request: SearchRequest with query parameters
            db: Database session for data enrichment

        Returns:
            SearchResponse with results and metadata

        Raises:
            Exception: If search fails
        """
        start_time = time.time()

        try:
            # Parse query and determine strategy
            parsed_query = self.query_parser.parse(request.query)
            strategy = request.ranking_strategy or "auto"

            if strategy == "auto":
                strategy = parsed_query["recommended_strategy"]
                logger.info(
                    f"Auto-detected strategy: {strategy} (intent: {parsed_query['intent']})"
                )

            logger.info(
                f"Executing search: '{request.query}' "
                f"(strategy={strategy}, limit={request.limit}, threshold={request.threshold})"
            )

            # Execute search based on strategy
            if strategy == "semantic":
                results_data = await self._semantic_search(request, db)
            elif strategy == "bm25":
                results_data = await self._keyword_search(request, db)
            elif strategy == "hybrid_rrf":
                results_data = await self._hybrid_search_rrf(request, db)
            elif strategy == "hybrid_weighted":
                results_data = await self._hybrid_search_weighted(request, db)
            else:
                raise ValueError(f"Unsupported ranking strategy: {strategy}")

            # Calculate query time
            query_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Search completed: {len(results_data)} results in {query_time_ms}ms "
                f"(strategy={strategy})"
            )

            return SearchResponse(
                results=results_data,
                total=len(results_data),
                query=request.query,
                query_time_ms=query_time_ms,
                threshold=request.threshold,
                ranking_strategy=strategy,
                query_intent=parsed_query.get("intent") if request.ranking_strategy == "auto" else None,
            )

        except Exception as e:
            query_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Search failed after {query_time_ms}ms: {e}", exc_info=True)
            raise

    async def _semantic_search(
        self,
        request: SearchRequest,
        db: AsyncSession
    ) -> List[SearchResultItem]:
        """Execute pure semantic search using vector store."""
        # Perform semantic search on vector store
        vector_results = self.vector_store.search(
            query=request.query,
            limit=request.limit * 2,  # Get more for filtering
            threshold=request.threshold,
        )

        if not vector_results:
            return []

        # Convert to dict format
        semantic_results = []
        for embedding_id, content, score, metadata in vector_results:
            if embedding_id.startswith("msg_"):
                try:
                    message_id = int(embedding_id.split("_")[1])
                    semantic_results.append({
                        "message_id": message_id,
                        "content": content,
                        "score": score,
                        "semantic_score": score,
                    })
                except (IndexError, ValueError) as e:
                    logger.warning(f"Invalid embedding_id: {embedding_id} - {e}")

        # Enrich with database data
        enriched = await self._enrich_results(semantic_results, request, db)
        return enriched[:request.limit]

    async def _keyword_search(
        self,
        request: SearchRequest,
        db: AsyncSession
    ) -> List[SearchResultItem]:
        """Execute pure keyword search using BM25."""
        # Perform BM25 search
        keyword_results = self.keyword_retriever.search(
            query=request.query,
            limit=request.limit * 2,
        )

        if not keyword_results.get("results"):
            return []

        # Convert to standard format
        bm25_results = []
        for result in keyword_results["results"]:
            bm25_results.append({
                **result,
                "keyword_score": result.get("score", 0.0),
            })

        # Enrich with database data
        enriched = await self._enrich_results(bm25_results, request, db)
        return enriched[:request.limit]

    async def _hybrid_search_rrf(
        self,
        request: SearchRequest,
        db: AsyncSession
    ) -> List[SearchResultItem]:
        """Execute hybrid search with Reciprocal Rank Fusion."""
        # Get semantic results
        vector_results = self.vector_store.search(
            query=request.query,
            limit=request.limit * 2,
            threshold=request.threshold,
        )

        semantic_results = []
        for embedding_id, content, score, metadata in vector_results:
            if embedding_id.startswith("msg_"):
                try:
                    message_id = int(embedding_id.split("_")[1])
                    semantic_results.append({
                        "message_id": message_id,
                        "content": content,
                        "score": score,
                        "semantic_score": score,
                    })
                except (IndexError, ValueError):
                    pass

        # Get keyword results
        keyword_results_raw = self.keyword_retriever.search(
            query=request.query,
            limit=request.limit * 2,
        )

        keyword_results = []
        for result in keyword_results_raw.get("results", []):
            keyword_results.append({
                **result,
                "keyword_score": result.get("score", 0.0),
            })

        # Fuse results using RRF
        fusion = HybridSearchFusion(strategy="rrf")
        fused_results = fusion.fuse(semantic_results, keyword_results)

        # Deduplicate
        fused_results = deduplicate_results(fused_results)

        # Enrich with database data
        enriched = await self._enrich_results(fused_results, request, db)
        return enriched[:request.limit]

    async def _hybrid_search_weighted(
        self,
        request: SearchRequest,
        db: AsyncSession
    ) -> List[SearchResultItem]:
        """Execute hybrid search with weighted score fusion."""
        # Get semantic results
        vector_results = self.vector_store.search(
            query=request.query,
            limit=request.limit * 2,
            threshold=request.threshold,
        )

        semantic_results = []
        for embedding_id, content, score, metadata in vector_results:
            if embedding_id.startswith("msg_"):
                try:
                    message_id = int(embedding_id.split("_")[1])
                    semantic_results.append({
                        "message_id": message_id,
                        "content": content,
                        "score": score,
                        "semantic_score": score,
                    })
                except (IndexError, ValueError):
                    pass

        # Get keyword results
        keyword_results_raw = self.keyword_retriever.search(
            query=request.query,
            limit=request.limit * 2,
        )

        keyword_results = []
        for result in keyword_results_raw.get("results", []):
            keyword_results.append({
                **result,
                "keyword_score": result.get("score", 0.0),
            })

        # Fuse results using weighted strategy
        fusion = HybridSearchFusion(
            strategy="weighted",
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
        )
        fused_results = fusion.fuse(semantic_results, keyword_results)

        # Deduplicate
        fused_results = deduplicate_results(fused_results)

        # Enrich with database data
        enriched = await self._enrich_results(fused_results, request, db)
        return enriched[:request.limit]

    async def _enrich_results(
        self,
        results: List[Dict[str, Any]],
        request: SearchRequest,
        db: AsyncSession
    ) -> List[SearchResultItem]:
        """
        Enrich search results with full database data.

        Args:
            results: List of result dictionaries with message_ids
            request: Original search request (for filtering)
            db: Database session

        Returns:
            List of enriched SearchResultItem objects
        """
        if not results:
            return []

        # Extract message IDs
        message_ids = [r["message_id"] for r in results if "message_id" in r]

        if not message_ids:
            return []

        # Fetch from database
        stmt = (
            select(Message, Source)
            .join(Source, Message.source_id == Source.id)
            .where(Message.id.in_(message_ids))
        )

        # Apply filters
        filters = []
        if request.source_types:
            filters.append(Source.type.in_(request.source_types))
        if request.channels:
            filters.append(Message.channel.in_(request.channels))
        if request.date_from:
            filters.append(Message.timestamp >= request.date_from)
        if request.date_to:
            filters.append(Message.timestamp <= request.date_to)

        if filters:
            stmt = stmt.where(and_(*filters))

        result = await db.execute(stmt)
        rows = result.all()

        # Build lookup by message_id
        messages_by_id = {message.id: (message, source) for message, source in rows}

        # Build enriched results maintaining order
        enriched_results = []
        for result_dict in results:
            message_id = result_dict.get("message_id")
            if message_id not in messages_by_id:
                continue

            message, source = messages_by_id[message_id]

            result_item = SearchResultItem(
                content=message.content,
                source=source.type,
                channel=message.channel,
                author=message.author,
                timestamp=message.timestamp,
                score=result_dict.get("score", 0.0),
                message_id=message.id,
                external_id=message.external_id,
                message_metadata=message.message_metadata,
                ranking_details=result_dict.get("ranking_details"),
            )
            enriched_results.append(result_item)

        return enriched_results

    async def health_check(self) -> dict:
        """
        Check the health of the search service.

        Returns:
            Dictionary with health status information
        """
        try:
            vector_store_connected = self.vector_store.check_connection()
            doc_count = self.vector_store.get_collection_count()

            return {
                "status": "healthy" if vector_store_connected else "degraded",
                "vector_store": {
                    "connected": vector_store_connected,
                    "collection": self.vector_store.collection_name,
                    "document_count": doc_count,
                },
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Global service instance (singleton pattern)
_search_service: Optional[SearchService] = None


def get_search_service(vector_store: VectorStore) -> SearchService:
    """
    Get or create the global SearchService instance.

    Args:
        vector_store: VectorStore instance

    Returns:
        SearchService instance

    Usage:
        from src.db.vector_store import get_vector_store

        vector_store = get_vector_store()
        service = get_search_service(vector_store)
    """
    global _search_service
    if _search_service is None:
        _search_service = SearchService(vector_store)
    return _search_service
