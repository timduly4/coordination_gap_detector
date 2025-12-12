"""
Tests for embedding generation module.
"""
import pytest

from src.models.embeddings import EmbeddingGenerator, get_embedding_generator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create an embedding generator instance for testing."""
        return EmbeddingGenerator()

    def test_initialization(self, generator):
        """Test that the generator initializes correctly."""
        assert generator is not None
        assert generator.model is not None
        assert generator.embedding_dimension > 0
        assert generator.model_name == "all-MiniLM-L6-v2"

    def test_generate_embedding_single_text(self, generator):
        """Test generating embedding for a single text."""
        text = "This is a test message about OAuth implementation."
        embedding = generator.generate_embedding(text)

        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == generator.embedding_dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_generate_embedding_empty_text(self, generator):
        """Test generating embedding for empty text returns zero vector."""
        embedding = generator.generate_embedding("")

        assert embedding is not None
        assert len(embedding) == generator.embedding_dimension
        assert all(x == 0.0 for x in embedding)

    def test_generate_embedding_whitespace_only(self, generator):
        """Test generating embedding for whitespace-only text."""
        embedding = generator.generate_embedding("   \n\t  ")

        assert embedding is not None
        assert len(embedding) == generator.embedding_dimension
        assert all(x == 0.0 for x in embedding)

    def test_generate_embeddings_batch(self, generator):
        """Test generating embeddings for multiple texts in batch."""
        texts = [
            "OAuth implementation discussion",
            "Database migration planning",
            "API endpoint refactoring",
            "Testing framework setup",
        ]
        embeddings = generator.generate_embeddings(texts)

        assert embeddings is not None
        assert len(embeddings) == len(texts)
        assert all(len(emb) == generator.embedding_dimension for emb in embeddings)

    def test_generate_embeddings_empty_list(self, generator):
        """Test generating embeddings for empty list."""
        embeddings = generator.generate_embeddings([])

        assert embeddings == []

    def test_generate_embeddings_with_empty_texts(self, generator):
        """Test batch generation with some empty texts."""
        texts = [
            "Valid text",
            "",
            "Another valid text",
            "   ",
        ]
        embeddings = generator.generate_embeddings(texts)

        assert len(embeddings) == len(texts)
        # Empty texts should have zero vectors
        assert all(x == 0.0 for x in embeddings[1])
        assert all(x == 0.0 for x in embeddings[3])
        # Valid texts should have non-zero vectors
        assert any(x != 0.0 for x in embeddings[0])
        assert any(x != 0.0 for x in embeddings[2])

    def test_embedding_deterministic(self, generator):
        """Test that same input produces same embedding."""
        text = "Consistent text for testing"

        embedding1 = generator.generate_embedding(text)
        embedding2 = generator.generate_embedding(text)

        assert embedding1 == embedding2

    def test_semantic_similarity(self, generator):
        """Test that similar texts have similar embeddings."""
        text1 = "OAuth authentication implementation"
        text2 = "OAuth authorization setup"
        text3 = "Database schema migration"

        emb1 = generator.generate_embedding(text1)
        emb2 = generator.generate_embedding(text2)
        emb3 = generator.generate_embedding(text3)

        # Calculate cosine similarity
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            return dot_product  # Already normalized

        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)

        # Similar texts (OAuth related) should be more similar than dissimilar texts
        assert sim_1_2 > sim_1_3

    def test_get_dimension(self, generator):
        """Test getting embedding dimension."""
        dimension = generator.get_dimension()

        assert dimension > 0
        assert dimension == generator.embedding_dimension
        assert isinstance(dimension, int)

    def test_singleton_pattern(self):
        """Test that get_embedding_generator returns singleton instance."""
        gen1 = get_embedding_generator()
        gen2 = get_embedding_generator()

        # Should be the same instance
        assert gen1 is gen2

    def test_batch_processing_large_texts(self, generator):
        """Test batch processing with large number of texts."""
        texts = [f"Test message number {i}" for i in range(100)]
        embeddings = generator.generate_embeddings(texts, batch_size=16)

        assert len(embeddings) == 100
        assert all(len(emb) == generator.embedding_dimension for emb in embeddings)

    def test_unicode_text(self, generator):
        """Test generating embeddings for Unicode text."""
        texts = [
            "Hello world",
            "Bonjour le monde",
            "こんにちは世界",
            "Привет мир",
        ]
        embeddings = generator.generate_embeddings(texts)

        assert len(embeddings) == len(texts)
        assert all(len(emb) == generator.embedding_dimension for emb in embeddings)
        assert all(any(x != 0.0 for x in emb) for emb in embeddings)
