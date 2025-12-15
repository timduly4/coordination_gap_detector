"""
Retrieval service for keyword-based search using BM25.

Provides high-level interface for searching messages across
the Elasticsearch index with BM25 scoring.
"""

import logging
from typing import Any, Optional

from src.db.elasticsearch import get_es_client
from src.ranking.constants import BM25_B, BM25_K1, DEFAULT_INDEX_NAME
from src.ranking.scoring import BM25Scorer

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """
    Keyword-based retrieval using BM25 scoring.

    This class provides a high-level interface for searching
    messages with BM25 ranking, wrapping Elasticsearch operations.
    """

    def __init__(
        self,
        index_name: str = DEFAULT_INDEX_NAME,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """
        Initialize keyword retriever.

        Args:
            index_name: Elasticsearch index name (default: "messages")
            k1: BM25 term frequency saturation (default: 1.5)
            b: BM25 length normalization (default: 0.75)
        """
        self.index_name = index_name
        self.k1 = k1
        self.b = b

        # Initialize Elasticsearch client
        self.es_client = get_es_client()

        # Initialize BM25 scorer for local calculations if needed
        self.bm25_scorer = BM25Scorer(k1=k1, b=b)

        logger.info(
            f"KeywordRetriever initialized: index={index_name}, "
            f"k1={k1}, b={b}"
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        source_filter: Optional[str] = None,
        channel_filter: Optional[str] = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """
        Search messages using BM25 keyword matching.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            source_filter: Filter by source platform (e.g., "slack")
            channel_filter: Filter by specific channel
            explain: Include detailed score explanation

        Returns:
            dict: Search results with metadata
                - results: List of matching messages with scores
                - total: Total number of matches
                - query: Original query
                - retrieval_method: "bm25"
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to keyword search")
            return {
                "results": [],
                "total": 0,
                "query": query,
                "retrieval_method": "bm25"
            }

        try:
            # Search using Elasticsearch (which uses BM25 by default)
            search_results = self.es_client.search_messages(
                index_name=self.index_name,
                query=query,
                size=limit,
                source_filter=source_filter,
                channel_filter=channel_filter,
                explain=explain,
            )

            # Add metadata
            results = {
                "results": search_results.get("results", []),
                "total": search_results.get("total", 0),
                "query": query,
                "retrieval_method": "bm25",
                "parameters": {
                    "k1": self.k1,
                    "b": self.b
                }
            }

            logger.info(
                f"BM25 search for '{query}': {results['total']} results"
            )

            return results

        except Exception as e:
            logger.error(f"Error during BM25 search: {e}")
            return {
                "results": [],
                "total": 0,
                "query": query,
                "retrieval_method": "bm25",
                "error": str(e)
            }

    def search_with_explanation(
        self,
        query: str,
        limit: int = 10,
        source_filter: Optional[str] = None,
        channel_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Search with detailed BM25 score explanations.

        Args:
            query: Search query text
            limit: Maximum number of results
            source_filter: Filter by source platform
            channel_filter: Filter by channel

        Returns:
            dict: Results with score explanations
        """
        return self.search(
            query=query,
            limit=limit,
            source_filter=source_filter,
            channel_filter=channel_filter,
            explain=True
        )

    def get_index_stats(self) -> dict[str, Any]:
        """
        Get statistics about the search index.

        Returns:
            dict: Index statistics including document count
        """
        try:
            doc_count = self.es_client.get_document_count(self.index_name)
            cluster_health = self.es_client.get_cluster_health()

            return {
                "index_name": self.index_name,
                "document_count": doc_count,
                "cluster_status": cluster_health.get("status", "unknown"),
                "bm25_parameters": {
                    "k1": self.k1,
                    "b": self.b
                }
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {
                "index_name": self.index_name,
                "error": str(e)
            }

    def check_health(self) -> bool:
        """
        Check if retriever is healthy and can perform searches.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Check Elasticsearch connection
            is_connected = self.es_client.check_connection()

            if not is_connected:
                logger.warning("Elasticsearch not connected")
                return False

            # Check if index exists
            doc_count = self.es_client.get_document_count(self.index_name)
            logger.info(
                f"Keyword retriever healthy: {doc_count} documents in index"
            )

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


def create_keyword_retriever(
    index_name: str = DEFAULT_INDEX_NAME,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> KeywordRetriever:
    """
    Factory function to create a keyword retriever.

    Args:
        index_name: Elasticsearch index name
        k1: BM25 term frequency saturation parameter
        b: BM25 length normalization parameter

    Returns:
        KeywordRetriever: Configured retriever instance
    """
    return KeywordRetriever(index_name=index_name, k1=k1, b=b)
