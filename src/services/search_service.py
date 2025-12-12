"""
Search service implementing business logic for semantic search operations.

This service coordinates between the API layer and the database/vector store,
implementing search logic, filtering, and result enrichment.
"""
import logging
import time
from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Message, Source
from src.db.vector_store import VectorStore
from src.models.schemas import SearchRequest, SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service class for handling search operations.

    This class implements the business logic for:
    - Semantic search using vector store
    - Result enrichment with database data
    - Filtering by source, channel, date range
    - Result formatting and validation
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the search service.

        Args:
            vector_store: VectorStore instance for semantic search
        """
        self.vector_store = vector_store
        logger.info("SearchService initialized")

    async def search(
        self,
        request: SearchRequest,
        db: AsyncSession
    ) -> SearchResponse:
        """
        Execute a semantic search query.

        This method:
        1. Performs semantic search using vector store
        2. Enriches results with full message data from database
        3. Applies filters (source types, channels, date range)
        4. Formats results according to API schema

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
            logger.info(
                f"Executing search query: '{request.query}' "
                f"(limit={request.limit}, threshold={request.threshold})"
            )

            # Step 1: Perform semantic search on vector store
            vector_results = self.vector_store.search(
                query=request.query,
                limit=request.limit * 2,  # Get more results for filtering
                threshold=request.threshold,
            )

            if not vector_results:
                logger.info("No results found in vector store")
                query_time_ms = int((time.time() - start_time) * 1000)
                return SearchResponse(
                    results=[],
                    total=0,
                    query=request.query,
                    query_time_ms=query_time_ms,
                    threshold=request.threshold,
                )

            # Step 2: Extract message IDs from vector results
            message_ids = []
            scores_by_id = {}

            for embedding_id, content, score, metadata in vector_results:
                # Extract message_id from embedding_id (format: "msg_{id}")
                if embedding_id.startswith("msg_"):
                    try:
                        message_id = int(embedding_id.split("_")[1])
                        message_ids.append(message_id)
                        scores_by_id[message_id] = score
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Invalid embedding_id format: {embedding_id} - {e}")
                        continue

            if not message_ids:
                logger.warning("No valid message IDs extracted from vector results")
                query_time_ms = int((time.time() - start_time) * 1000)
                return SearchResponse(
                    results=[],
                    total=0,
                    query=request.query,
                    query_time_ms=query_time_ms,
                    threshold=request.threshold,
                )

            # Step 3: Fetch full message data from database with source information
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

            # Step 4: Build search result items
            results = []
            for message, source in rows:
                score = scores_by_id.get(message.id, 0.0)

                result_item = SearchResultItem(
                    content=message.content,
                    source=source.type,
                    channel=message.channel,
                    author=message.author,
                    timestamp=message.timestamp,
                    score=score,
                    message_id=message.id,
                    external_id=message.external_id,
                    message_metadata=message.message_metadata,
                )
                results.append((score, result_item))  # Include score for sorting

            # Step 5: Sort by score (descending) and apply limit
            results.sort(key=lambda x: x[0], reverse=True)
            results = [item for score, item in results[:request.limit]]

            # Calculate query time
            query_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Search completed: {len(results)} results in {query_time_ms}ms "
                f"(vector results: {len(vector_results)}, after filtering: {len(results)})"
            )

            return SearchResponse(
                results=results,
                total=len(results),
                query=request.query,
                query_time_ms=query_time_ms,
                threshold=request.threshold,
            )

        except Exception as e:
            query_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Search failed after {query_time_ms}ms: {e}", exc_info=True)
            raise

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
