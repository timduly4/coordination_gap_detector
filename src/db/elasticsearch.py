"""
Elasticsearch client for keyword search and BM25 ranking.
"""
import logging
from typing import Any, Optional

from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from elasticsearch.exceptions import NotFoundError

from src.config import get_settings

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """
    Elasticsearch client wrapper for search operations.

    Provides:
    - Client initialization and connection management
    - Index creation with proper mappings
    - Document indexing for messages
    - BM25-based search queries
    - Health checks and error handling
    """

    # Index mapping for messages
    MESSAGES_INDEX_MAPPING = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "source": {"type": "keyword"},
                "channel": {"type": "keyword"},
                "author": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "metadata": {
                    "type": "object",
                    "enabled": True
                },
                "message_id": {
                    "type": "keyword"
                },
                "thread_id": {
                    "type": "keyword"
                }
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "standard": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        }
    }

    def __init__(self) -> None:
        """Initialize Elasticsearch client."""
        settings = get_settings()
        self.url = settings.elasticsearch_url
        self.api_key = settings.elasticsearch_api_key

        # Initialize synchronous client
        if self.api_key:
            self.client = Elasticsearch(
                self.url,
                api_key=self.api_key,
            )
        else:
            self.client = Elasticsearch(self.url)

        logger.info(f"Elasticsearch client initialized for {self.url}")

    def check_connection(self) -> bool:
        """
        Check if Elasticsearch is reachable.

        Returns:
            bool: True if connected, False otherwise
        """
        try:
            info = self.client.info()
            logger.info(f"Elasticsearch cluster: {info.get('cluster_name', 'unknown')}")
            return True
        except ESConnectionError as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking Elasticsearch connection: {e}")
            return False

    def get_cluster_health(self) -> dict[str, Any]:
        """
        Get Elasticsearch cluster health information.

        Returns:
            dict: Cluster health details
        """
        try:
            health = self.client.cluster.health()
            return {
                "status": health.get("status", "unknown"),
                "cluster_name": health.get("cluster_name", "unknown"),
                "number_of_nodes": health.get("number_of_nodes", 0),
                "active_shards": health.get("active_shards", 0),
            }
        except Exception as e:
            logger.error(f"Error getting cluster health: {e}")
            return {"status": "error", "message": str(e)}

    def create_messages_index(self, index_name: str = "messages") -> bool:
        """
        Create messages index with proper mappings.

        Args:
            index_name: Name of the index to create

        Returns:
            bool: True if created successfully, False otherwise
        """
        try:
            if self.client.indices.exists(index=index_name):
                logger.info(f"Index '{index_name}' already exists")
                return True

            self.client.indices.create(
                index=index_name,
                body=self.MESSAGES_INDEX_MAPPING
            )
            logger.info(f"Created index '{index_name}' with BM25 mappings")
            return True
        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")
            return False

    def delete_index(self, index_name: str) -> bool:
        """
        Delete an index.

        Args:
            index_name: Name of the index to delete

        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if not self.client.indices.exists(index=index_name):
                logger.warning(f"Index '{index_name}' does not exist")
                return False

            self.client.indices.delete(index=index_name)
            logger.info(f"Deleted index '{index_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting index '{index_name}': {e}")
            return False

    def index_message(
        self,
        index_name: str,
        message_id: str,
        content: str,
        source: str,
        channel: str,
        author: str,
        timestamp: str,
        metadata: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> bool:
        """
        Index a message document.

        Args:
            index_name: Index to store the message
            message_id: Unique message identifier
            content: Message content
            source: Source platform (slack, github, etc.)
            channel: Channel or repository name
            author: Message author
            timestamp: Message timestamp (ISO format)
            metadata: Additional metadata
            thread_id: Thread identifier if part of a thread

        Returns:
            bool: True if indexed successfully, False otherwise
        """
        try:
            document = {
                "message_id": message_id,
                "content": content,
                "source": source,
                "channel": channel,
                "author": author,
                "timestamp": timestamp,
                "metadata": metadata or {},
            }

            if thread_id:
                document["thread_id"] = thread_id

            self.client.index(
                index=index_name,
                id=message_id,
                document=document
            )
            logger.debug(f"Indexed message {message_id} in '{index_name}'")
            return True
        except Exception as e:
            logger.error(f"Error indexing message {message_id}: {e}")
            return False

    def bulk_index_messages(
        self,
        index_name: str,
        messages: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Bulk index multiple messages.

        Args:
            index_name: Index to store messages
            messages: List of message documents

        Returns:
            tuple: (successful_count, failed_count)
        """
        from elasticsearch.helpers import bulk

        try:
            actions = [
                {
                    "_index": index_name,
                    "_id": msg.get("message_id"),
                    "_source": msg
                }
                for msg in messages
            ]

            success, failed = bulk(self.client, actions, raise_on_error=False)
            logger.info(f"Bulk indexed {success} messages, {len(failed)} failed")
            return success, len(failed)
        except Exception as e:
            logger.error(f"Error bulk indexing messages: {e}")
            return 0, len(messages)

    def search_messages(
        self,
        index_name: str,
        query: str,
        size: int = 10,
        source_filter: Optional[str] = None,
        channel_filter: Optional[str] = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """
        Search messages using BM25 scoring.

        Args:
            index_name: Index to search
            query: Search query
            size: Number of results to return
            source_filter: Filter by source (e.g., 'slack')
            channel_filter: Filter by channel
            explain: Include score explanation in results

        Returns:
            dict: Search results with scores
        """
        try:
            # Build BM25 query
            search_query = self.build_bm25_query(
                query=query,
                size=size,
                source_filter=source_filter,
                channel_filter=channel_filter,
                explain=explain
            )

            response = self.client.search(
                index=index_name,
                body=search_query
            )

            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "message_id": hit["_id"],
                    "score": hit["_score"],
                    "source": hit["_source"]
                }

                # Add explanation if requested
                if explain and "_explanation" in hit:
                    result["explanation"] = hit["_explanation"]

                results.append(result)

            return {
                "total": response["hits"]["total"]["value"],
                "results": results
            }
        except NotFoundError:
            logger.warning(f"Index '{index_name}' not found")
            return {"total": 0, "results": []}
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return {"total": 0, "results": []}

    def build_bm25_query(
        self,
        query: str,
        size: int = 10,
        source_filter: Optional[str] = None,
        channel_filter: Optional[str] = None,
        explain: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict[str, Any]:
        """
        Build Elasticsearch query with BM25 scoring parameters.

        BM25 parameters:
        - k1: Controls term frequency saturation (default: 1.5)
        - b: Controls length normalization (default: 0.75)

        Args:
            query: Search query string
            size: Number of results to return
            source_filter: Filter by source platform
            channel_filter: Filter by channel
            explain: Include score explanation
            k1: BM25 term frequency saturation parameter
            b: BM25 length normalization parameter

        Returns:
            dict: Elasticsearch query DSL
        """
        # Build must clauses with BM25 parameters
        must_clauses = [
            {
                "match": {
                    "content": {
                        "query": query,
                        "operator": "or",
                        # BM25 similarity is default in ES 5.0+
                        # These parameters can be set at index level
                    }
                }
            }
        ]

        # Add filters
        filter_clauses = []
        if source_filter:
            filter_clauses.append({"term": {"source": source_filter}})
        if channel_filter:
            filter_clauses.append({"term": {"channel": channel_filter}})

        # Build complete query
        search_query: dict[str, Any] = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses
                }
            },
            "size": size,
            "explain": explain
        }

        return search_query

    def get_document_count(self, index_name: str) -> int:
        """
        Get total document count in an index.

        Args:
            index_name: Index name

        Returns:
            int: Number of documents
        """
        try:
            count = self.client.count(index=index_name)
            return count.get("count", 0)
        except NotFoundError:
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    def close(self) -> None:
        """Close the Elasticsearch client connection."""
        try:
            self.client.close()
            logger.info("Elasticsearch client closed")
        except Exception as e:
            logger.error(f"Error closing Elasticsearch client: {e}")


# Global client instance
_es_client: Optional[ElasticsearchClient] = None


def get_es_client() -> ElasticsearchClient:
    """
    Get or create the global Elasticsearch client instance.

    Returns:
        ElasticsearchClient: Singleton client instance
    """
    global _es_client
    if _es_client is None:
        _es_client = ElasticsearchClient()
    return _es_client
