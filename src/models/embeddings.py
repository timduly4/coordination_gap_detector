"""
Embedding generation for semantic search.

This module provides functionality to generate dense vector embeddings
from text using sentence-transformers models.
"""
import logging
from functools import lru_cache
from typing import List, Union

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.

    Uses the 'all-MiniLM-L6-v2' model by default, which provides:
    - Good performance for semantic similarity
    - Fast inference (suitable for real-time)
    - Compact 384-dimensional embeddings
    - Multilingual support
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' for good performance/speed balance.
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Embedding model loaded successfully. Dimension: {self.embedding_dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model '{model_name}': {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            # Return zero vector for empty text
            return [0.0] * self.embedding_dimension

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show a progress bar

        Returns:
            List of embedding vectors (list of lists of floats)
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding generation")
            return []

        # Filter out empty texts but keep track of indices
        non_empty_texts = []
        non_empty_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        if not non_empty_texts:
            logger.warning("All texts are empty, returning zero vectors")
            return [[0.0] * self.embedding_dimension] * len(texts)

        try:
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Create result with zero vectors for empty texts
            result = [[0.0] * self.embedding_dimension] * len(texts)
            for idx, embedding in zip(non_empty_indices, embeddings):
                result[idx] = embedding.tolist()

            return result
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embeddings.

        Returns:
            Integer representing embedding dimension
        """
        return self.embedding_dimension


# Global embedding generator instance (singleton pattern)
_embedding_generator: Union[EmbeddingGenerator, None] = None


@lru_cache()
def get_embedding_generator(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingGenerator:
    """
    Get or create the global EmbeddingGenerator instance.

    This uses a singleton pattern to avoid loading the model multiple times.

    Args:
        model_name: Name of the sentence-transformers model to use

    Returns:
        EmbeddingGenerator instance

    Usage:
        generator = get_embedding_generator()
        embedding = generator.generate_embedding("Hello world")
    """
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator(model_name=model_name)
    return _embedding_generator
