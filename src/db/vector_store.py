"""
ChromaDB vector store client for semantic search.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.models.embeddings import get_embedding_generator

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB client for storing and searching message embeddings.

    This class provides a complete interface for:
    - Inserting documents with embeddings
    - Semantic similarity search
    - Batch operations
    - Metadata filtering
    """

    def __init__(self) -> None:
        """Initialize ChromaDB client."""
        import os

        settings = get_settings()

        # Use HTTP client when CHROMA_HOST is set (Docker/production)
        # Otherwise use persistent client for local development
        chroma_host = os.getenv("CHROMA_HOST", "chromadb")
        use_http_client = os.getenv("CHROMA_HTTP_CLIENT", "false").lower() == "true"

        if use_http_client:
            # HTTP client for Docker Compose or remote ChromaDB
            logger.info(f"Using ChromaDB HTTP client: {chroma_host}:8000")
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=8000,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            # Persistent client for local development
            logger.info(f"Using ChromaDB persistent client: {settings.chroma_persist_dir}")
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

        # Initialize embedding generator
        self.embedding_generator = get_embedding_generator()

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

    def insert(
        self,
        message_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert a single document into the vector store.

        Args:
            message_id: Database message ID
            content: Text content to embed and store
            metadata: Optional metadata to store with the document

        Returns:
            Embedding ID (string representation of message_id)

        Raises:
            Exception: If insertion fails
        """
        embedding_id = f"msg_{message_id}"

        try:
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(content)

            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata["message_id"] = message_id

            # Insert into ChromaDB
            self.collection.add(
                ids=[embedding_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[doc_metadata],
            )

            logger.debug(f"Inserted document {embedding_id} into vector store")
            return embedding_id

        except Exception as e:
            logger.error(f"Failed to insert document {embedding_id}: {e}")
            raise

    def insert_batch(
        self,
        message_ids: List[int],
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Insert multiple documents into the vector store in batch.

        Args:
            message_ids: List of database message IDs
            contents: List of text contents
            metadatas: Optional list of metadata dicts

        Returns:
            List of embedding IDs

        Raises:
            ValueError: If input lists have different lengths
            Exception: If batch insertion fails
        """
        if len(message_ids) != len(contents):
            raise ValueError("message_ids and contents must have the same length")

        if metadatas and len(metadatas) != len(message_ids):
            raise ValueError("metadatas must have the same length as message_ids")

        try:
            # Generate embedding IDs
            embedding_ids = [f"msg_{msg_id}" for msg_id in message_ids]

            # Generate embeddings in batch
            embeddings = self.embedding_generator.generate_embeddings(contents, batch_size=32)

            # Prepare metadata
            if metadatas:
                doc_metadatas = []
                for i, metadata in enumerate(metadatas):
                    meta = metadata.copy()
                    meta["message_id"] = message_ids[i]
                    doc_metadatas.append(meta)
            else:
                doc_metadatas = [{"message_id": msg_id} for msg_id in message_ids]

            # Insert into ChromaDB
            self.collection.add(
                ids=embedding_ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=doc_metadatas,
            )

            logger.info(f"Inserted {len(embedding_ids)} documents into vector store")
            return embedding_ids

        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            raise

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: Query text to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)
            filter_metadata: Optional metadata filters

        Returns:
            List of tuples: (embedding_id, content, score, metadata)

        Note:
            ChromaDB returns similarity scores, where higher is better.
            Scores are cosine similarity (range: -1 to 1, normalized embeddings: 0 to 1)
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)

            # Build where clause for metadata filtering
            where = filter_metadata if filter_metadata else None

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Process results
            output = []
            if results and results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                documents = results["documents"][0] if results["documents"] else []
                metadatas = results["metadatas"][0] if results["metadatas"] else []
                distances = results["distances"][0] if results["distances"] else []

                for i in range(len(ids)):
                    # Convert distance to similarity score
                    # ChromaDB returns L2 distance, convert to cosine similarity
                    # For normalized embeddings: similarity = 1 - (distance^2 / 2)
                    distance = distances[i] if i < len(distances) else 0
                    similarity = 1 - (distance * distance / 2)

                    # Apply threshold filter
                    if similarity >= threshold:
                        output.append(
                            (
                                ids[i],
                                documents[i] if i < len(documents) else "",
                                similarity,
                                metadatas[i] if i < len(metadatas) else {},
                            )
                        )

            logger.debug(f"Search returned {len(output)} results (threshold: {threshold})")
            return output

        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            raise

    def get_embeddings_by_message_ids(
        self, message_ids: List[int]
    ) -> Dict[int, List[float]]:
        """
        Retrieve embeddings for multiple messages by their message IDs.

        Args:
            message_ids: List of database message IDs

        Returns:
            Dictionary mapping message_id -> embedding vector
        """
        if not message_ids:
            return {}

        try:
            # Build ChromaDB IDs from message IDs
            embedding_ids = [f"msg_{msg_id}" for msg_id in message_ids]

            # Retrieve from ChromaDB
            results = self.collection.get(
                ids=embedding_ids,
                include=["embeddings"],
            )

            # Build result dictionary
            embeddings_dict = {}
            if results and results.get("embeddings") is not None and len(results.get("embeddings", [])) > 0:
                for i, emb_id in enumerate(results["ids"]):
                    # Extract message_id from embedding_id (format: "msg_{id}")
                    msg_id = int(emb_id.replace("msg_", ""))
                    embeddings_dict[msg_id] = results["embeddings"][i]

            logger.debug(f"Retrieved {len(embeddings_dict)} embeddings for {len(message_ids)} messages")
            return embeddings_dict

        except Exception as e:
            logger.error(f"Failed to retrieve embeddings: {e}")
            return {}

    def delete(self, embedding_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            embedding_id: ID of the document to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.collection.delete(ids=[embedding_id])
            logger.debug(f"Deleted document {embedding_id} from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {embedding_id}: {e}")
            return False

    def delete_batch(self, embedding_ids: List[str]) -> bool:
        """
        Delete multiple documents from the vector store.

        Args:
            embedding_ids: List of embedding IDs to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.collection.delete(ids=embedding_ids)
            logger.info(f"Deleted {len(embedding_ids)} documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete batch: {e}")
            return False

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all document IDs and delete them (preserves collection UUID)
            result = self.collection.get()
            ids = result.get('ids', [])

            if ids:
                self.collection.delete(ids=ids)
                logger.warning(f"Cleared {len(ids)} documents from collection '{self.collection_name}'")
            else:
                logger.info(f"Collection '{self.collection_name}' is already empty")

            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_by_id(self, embedding_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieve a document by its embedding ID.

        Args:
            embedding_id: ID of the document to retrieve

        Returns:
            Tuple of (content, metadata) if found, None otherwise
        """
        try:
            result = self.collection.get(
                ids=[embedding_id],
                include=["documents", "metadatas"],
            )

            if result and result["ids"] and len(result["ids"]) > 0:
                content = result["documents"][0] if result["documents"] else ""
                metadata = result["metadatas"][0] if result["metadatas"] else {}
                return (content, metadata)

            return None
        except Exception as e:
            logger.error(f"Failed to get document {embedding_id}: {e}")
            return None


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
