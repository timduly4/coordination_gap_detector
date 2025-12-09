"""
ChromaDB vector store client for semantic search.
"""
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB client for storing and searching message embeddings.
    """

    def __init__(self) -> None:
        """Initialize ChromaDB client."""
        settings = get_settings()

        # In production with docker compose, we'll use HTTP client
        # For now, use persistent client for local file storage
        if settings.environment == "production":
            # HTTP client for Docker service
            self.client = chromadb.HttpClient(
                host="chromadb", port=8000, settings=ChromaSettings(anonymized_telemetry=False)
            )
        else:
            # Persistent client for development
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        logger.info(f"ChromaDB client initialized (environment: {settings.environment})")

        # Get or create default collection
        self.collection_name = "coordination_messages"
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Messages from all sources for coordination gap detection"},
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' ready")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise

    def check_connection(self) -> bool:
        """
        Check if ChromaDB connection is working.
        Returns True if connection is successful, False otherwise.
        """
        try:
            # Try to get heartbeat or list collections
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB connection check failed: {e}")
            return False

    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        Returns 0 if there's an error.
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return 0


# Global vector store instance (singleton pattern)
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get or create the global VectorStore instance.

    Usage:
        store = get_vector_store()
        store.check_connection()
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
