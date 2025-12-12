"""
Tests for ChromaDB vector store operations.
"""
import pytest

from src.db.vector_store import VectorStore, get_vector_store


class TestVectorStore:
    """Test suite for VectorStore class."""

    @pytest.fixture
    def vector_store(self):
        """Create a VectorStore instance for testing."""
        store = VectorStore()
        # Clear any existing data
        store.clear_collection()
        return store

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            {
                "id": 1,
                "content": "We need to implement OAuth2 authentication for the API",
                "metadata": {"author": "alice@demo.com", "channel": "#engineering"},
            },
            {
                "id": 2,
                "content": "OAuth implementation should use JWT tokens",
                "metadata": {"author": "bob@demo.com", "channel": "#architecture"},
            },
            {
                "id": 3,
                "content": "Database migration planning for next quarter",
                "metadata": {"author": "charlie@demo.com", "channel": "#data"},
            },
            {
                "id": 4,
                "content": "Let's refactor the API endpoints to follow REST conventions",
                "metadata": {"author": "diana@demo.com", "channel": "#engineering"},
            },
        ]

    def test_initialization(self, vector_store):
        """Test that VectorStore initializes correctly."""
        assert vector_store is not None
        assert vector_store.client is not None
        assert vector_store.collection is not None
        assert vector_store.embedding_generator is not None

    def test_check_connection(self, vector_store):
        """Test checking ChromaDB connection."""
        is_connected = vector_store.check_connection()
        assert is_connected is True

    def test_get_collection_count_empty(self, vector_store):
        """Test getting count from empty collection."""
        count = vector_store.get_collection_count()
        assert count == 0

    def test_insert_single_document(self, vector_store, sample_messages):
        """Test inserting a single document."""
        msg = sample_messages[0]
        embedding_id = vector_store.insert(
            message_id=msg["id"], content=msg["content"], metadata=msg["metadata"]
        )

        assert embedding_id == f"msg_{msg['id']}"
        assert vector_store.get_collection_count() == 1

    def test_insert_batch(self, vector_store, sample_messages):
        """Test inserting multiple documents in batch."""
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        metadatas = [msg["metadata"] for msg in sample_messages]

        embedding_ids = vector_store.insert_batch(message_ids, contents, metadatas)

        assert len(embedding_ids) == len(sample_messages)
        assert vector_store.get_collection_count() == len(sample_messages)
        assert all(eid.startswith("msg_") for eid in embedding_ids)

    def test_insert_batch_mismatched_lengths(self, vector_store):
        """Test that insert_batch raises error for mismatched lengths."""
        with pytest.raises(ValueError):
            vector_store.insert_batch(
                message_ids=[1, 2], contents=["content1"]  # Mismatched lengths
            )

    def test_search_semantic_similarity(self, vector_store, sample_messages):
        """Test semantic similarity search."""
        # Insert sample messages
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        metadatas = [msg["metadata"] for msg in sample_messages]
        vector_store.insert_batch(message_ids, contents, metadatas)

        # Search for OAuth-related content
        results = vector_store.search("OAuth authentication setup", limit=3)

        assert len(results) > 0
        assert len(results) <= 3

        # Results should be tuples of (embedding_id, content, score, metadata)
        for embedding_id, content, score, metadata in results:
            assert isinstance(embedding_id, str)
            assert isinstance(content, str)
            assert isinstance(score, float)
            assert isinstance(metadata, dict)
            assert 0.0 <= score <= 1.0

        # First result should be most relevant (OAuth-related)
        top_result = results[0]
        assert "oauth" in top_result[1].lower()

    def test_search_with_threshold(self, vector_store, sample_messages):
        """Test search with similarity threshold."""
        # Insert sample messages
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        vector_store.insert_batch(message_ids, contents)

        # Search with high threshold
        results = vector_store.search("OAuth JWT tokens", limit=10, threshold=0.5)

        # All results should have score >= threshold
        assert all(score >= 0.5 for _, _, score, _ in results)

    def test_search_empty_query(self, vector_store, sample_messages):
        """Test search with empty query."""
        # Insert sample messages
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        vector_store.insert_batch(message_ids, contents)

        # Empty query should return results with low similarity scores
        results = vector_store.search("", limit=5)
        assert isinstance(results, list)

    def test_search_no_results(self, vector_store):
        """Test search when collection is empty."""
        results = vector_store.search("test query", limit=5)
        assert results == []

    def test_search_with_metadata_filter(self, vector_store, sample_messages):
        """Test search with metadata filtering."""
        # Insert sample messages
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        metadatas = [msg["metadata"] for msg in sample_messages]
        vector_store.insert_batch(message_ids, contents, metadatas)

        # Search with channel filter
        results = vector_store.search(
            "API implementation", limit=10, filter_metadata={"channel": "#engineering"}
        )

        # All results should be from #engineering channel
        for _, _, _, metadata in results:
            assert metadata.get("channel") == "#engineering"

    def test_delete_document(self, vector_store, sample_messages):
        """Test deleting a single document."""
        msg = sample_messages[0]
        embedding_id = vector_store.insert(message_id=msg["id"], content=msg["content"])

        assert vector_store.get_collection_count() == 1

        # Delete the document
        success = vector_store.delete(embedding_id)
        assert success is True
        assert vector_store.get_collection_count() == 0

    def test_delete_batch(self, vector_store, sample_messages):
        """Test deleting multiple documents."""
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        embedding_ids = vector_store.insert_batch(message_ids, contents)

        assert vector_store.get_collection_count() == len(sample_messages)

        # Delete first two documents
        success = vector_store.delete_batch(embedding_ids[:2])
        assert success is True
        assert vector_store.get_collection_count() == len(sample_messages) - 2

    def test_clear_collection(self, vector_store, sample_messages):
        """Test clearing the entire collection."""
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        vector_store.insert_batch(message_ids, contents)

        assert vector_store.get_collection_count() > 0

        # Clear collection
        success = vector_store.clear_collection()
        assert success is True
        assert vector_store.get_collection_count() == 0

    def test_get_by_id(self, vector_store, sample_messages):
        """Test retrieving a document by ID."""
        msg = sample_messages[0]
        embedding_id = vector_store.insert(
            message_id=msg["id"], content=msg["content"], metadata=msg["metadata"]
        )

        # Retrieve by ID
        result = vector_store.get_by_id(embedding_id)

        assert result is not None
        content, metadata = result
        assert content == msg["content"]
        assert "message_id" in metadata
        assert metadata["message_id"] == msg["id"]

    def test_get_by_id_not_found(self, vector_store):
        """Test retrieving non-existent document."""
        result = vector_store.get_by_id("msg_999999")
        assert result is None

    def test_singleton_pattern(self):
        """Test that get_vector_store returns singleton instance."""
        store1 = get_vector_store()
        store2 = get_vector_store()

        # Should be the same instance
        assert store1 is store2

    def test_batch_insert_preserves_order(self, vector_store, sample_messages):
        """Test that batch insert preserves message order."""
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        embedding_ids = vector_store.insert_batch(message_ids, contents)

        # Embedding IDs should correspond to message IDs
        for i, msg_id in enumerate(message_ids):
            assert embedding_ids[i] == f"msg_{msg_id}"

    def test_search_results_ordered_by_similarity(self, vector_store, sample_messages):
        """Test that search results are ordered by similarity score."""
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        vector_store.insert_batch(message_ids, contents)

        results = vector_store.search("OAuth authentication", limit=5)

        # Check that results are ordered by descending similarity
        scores = [score for _, _, score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_insert_with_special_characters(self, vector_store):
        """Test inserting content with special characters."""
        content = "Test with special chars: @user #channel https://example.com 🚀"
        embedding_id = vector_store.insert(message_id=1, content=content)

        assert embedding_id is not None

        # Should be retrievable
        result = vector_store.get_by_id(embedding_id)
        assert result is not None
        assert result[0] == content

    def test_search_multilingual_content(self, vector_store):
        """Test search with multilingual content."""
        multilingual_messages = [
            (1, "Hello world"),
            (2, "Bonjour le monde"),
            (3, "Hola mundo"),
        ]

        message_ids = [msg[0] for msg in multilingual_messages]
        contents = [msg[1] for msg in multilingual_messages]
        vector_store.insert_batch(message_ids, contents)

        # Search should work across languages
        results = vector_store.search("world greeting", limit=3)
        assert len(results) > 0

    def test_insert_very_long_content(self, vector_store):
        """Test inserting very long content."""
        long_content = "This is a sentence. " * 500  # Very long text
        embedding_id = vector_store.insert(message_id=1, content=long_content)

        assert embedding_id is not None
        assert vector_store.get_collection_count() == 1

    def test_concurrent_operations(self, vector_store, sample_messages):
        """Test that multiple operations work correctly in sequence."""
        # Insert
        message_ids = [msg["id"] for msg in sample_messages]
        contents = [msg["content"] for msg in sample_messages]
        vector_store.insert_batch(message_ids, contents)

        # Search
        results1 = vector_store.search("OAuth", limit=5)
        assert len(results1) > 0

        # Insert more
        vector_store.insert(message_id=99, content="Additional message about OAuth")

        # Search again
        results2 = vector_store.search("OAuth", limit=5)
        assert len(results2) >= len(results1)

        # Delete
        vector_store.delete("msg_99")

        # Final search
        results3 = vector_store.search("OAuth", limit=5)
        assert "msg_99" not in [r[0] for r in results3]
