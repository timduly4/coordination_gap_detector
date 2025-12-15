"""
Unit tests for keyword retrieval service.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.search.retrieval import KeywordRetriever, create_keyword_retriever


class TestKeywordRetriever:
    """Test suite for KeywordRetriever class."""

    @pytest.fixture
    def mock_es_client(self):
        """Create a mock Elasticsearch client."""
        mock_client = MagicMock()
        mock_client.check_connection.return_value = True
        mock_client.get_document_count.return_value = 100
        mock_client.get_cluster_health.return_value = {
            "status": "green",
            "cluster_name": "test-cluster"
        }
        return mock_client

    @pytest.fixture
    def retriever(self, mock_es_client):
        """Create a KeywordRetriever with mocked dependencies."""
        with patch("src.search.retrieval.get_es_client", return_value=mock_es_client):
            return KeywordRetriever(index_name="test_messages", k1=1.5, b=0.75)

    def test_initialization(self, retriever):
        """Test KeywordRetriever initializes correctly."""
        assert retriever.index_name == "test_messages"
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.es_client is not None
        assert retriever.bm25_scorer is not None

    def test_initialization_default_params(self, mock_es_client):
        """Test initialization with default parameters."""
        with patch("src.search.retrieval.get_es_client", return_value=mock_es_client):
            retriever = KeywordRetriever()
            assert retriever.index_name == "messages"
            assert retriever.k1 == 1.5
            assert retriever.b == 0.75

    def test_search_basic(self, retriever, mock_es_client):
        """Test basic search functionality."""
        # Mock search results
        mock_es_client.search_messages.return_value = {
            "results": [
                {
                    "message_id": "msg_1",
                    "score": 5.2,
                    "source": {
                        "content": "OAuth implementation",
                        "source": "slack",
                        "channel": "#engineering"
                    }
                }
            ],
            "total": 1
        }

        results = retriever.search(query="OAuth implementation", limit=10)

        # Verify results structure
        assert "results" in results
        assert "total" in results
        assert "query" in results
        assert "retrieval_method" in results

        # Verify search was called correctly
        mock_es_client.search_messages.assert_called_once()
        call_args = mock_es_client.search_messages.call_args

        assert call_args.kwargs["query"] == "OAuth implementation"
        assert call_args.kwargs["size"] == 10

    def test_search_with_filters(self, retriever, mock_es_client):
        """Test search with source and channel filters."""
        mock_es_client.search_messages.return_value = {
            "results": [],
            "total": 0
        }

        retriever.search(
            query="test",
            limit=5,
            source_filter="slack",
            channel_filter="#engineering"
        )

        # Verify filters were passed to ES client
        call_args = mock_es_client.search_messages.call_args
        assert call_args.kwargs["source_filter"] == "slack"
        assert call_args.kwargs["channel_filter"] == "#engineering"

    def test_search_empty_query(self, retriever, mock_es_client):
        """Test search with empty query."""
        results = retriever.search(query="", limit=10)

        # Should return empty results without calling ES
        assert results["total"] == 0
        assert len(results["results"]) == 0
        mock_es_client.search_messages.assert_not_called()

    def test_search_whitespace_query(self, retriever, mock_es_client):
        """Test search with whitespace-only query."""
        results = retriever.search(query="   ", limit=10)

        # Should return empty results
        assert results["total"] == 0
        assert len(results["results"]) == 0
        mock_es_client.search_messages.assert_not_called()

    def test_search_with_explanation(self, retriever, mock_es_client):
        """Test search with score explanations."""
        mock_es_client.search_messages.return_value = {
            "results": [
                {
                    "message_id": "msg_1",
                    "score": 5.2,
                    "source": {"content": "test"},
                    "explanation": {"details": "BM25 explanation"}
                }
            ],
            "total": 1
        }

        results = retriever.search_with_explanation(query="test", limit=5)

        # Verify explanation was requested
        call_args = mock_es_client.search_messages.call_args
        assert call_args.kwargs["explain"] is True

    def test_search_error_handling(self, retriever, mock_es_client):
        """Test search handles errors gracefully."""
        # Mock ES client to raise an error
        mock_es_client.search_messages.side_effect = Exception("ES error")

        results = retriever.search(query="test", limit=10)

        # Should return error response
        assert results["total"] == 0
        assert len(results["results"]) == 0
        assert "error" in results

    def test_search_includes_parameters(self, retriever, mock_es_client):
        """Test that search results include BM25 parameters."""
        mock_es_client.search_messages.return_value = {
            "results": [],
            "total": 0
        }

        results = retriever.search(query="test", limit=10)

        # Verify parameters are included
        assert "parameters" in results
        assert results["parameters"]["k1"] == 1.5
        assert results["parameters"]["b"] == 0.75

    def test_get_index_stats(self, retriever, mock_es_client):
        """Test getting index statistics."""
        mock_es_client.get_document_count.return_value = 100
        mock_es_client.get_cluster_health.return_value = {
            "status": "green",
            "cluster_name": "test-cluster"
        }

        stats = retriever.get_index_stats()

        # Verify stats structure
        assert "index_name" in stats
        assert "document_count" in stats
        assert "cluster_status" in stats
        assert "bm25_parameters" in stats

        # Verify values
        assert stats["index_name"] == "test_messages"
        assert stats["document_count"] == 100
        assert stats["cluster_status"] == "green"

    def test_get_index_stats_error(self, retriever, mock_es_client):
        """Test index stats handles errors."""
        mock_es_client.get_document_count.side_effect = Exception("ES error")

        stats = retriever.get_index_stats()

        # Should return error info
        assert "error" in stats
        assert stats["index_name"] == "test_messages"

    def test_check_health_healthy(self, retriever, mock_es_client):
        """Test health check when system is healthy."""
        mock_es_client.check_connection.return_value = True
        mock_es_client.get_document_count.return_value = 50

        is_healthy = retriever.check_health()

        assert is_healthy is True

    def test_check_health_es_disconnected(self, retriever, mock_es_client):
        """Test health check when ES is disconnected."""
        mock_es_client.check_connection.return_value = False

        is_healthy = retriever.check_health()

        assert is_healthy is False

    def test_check_health_error(self, retriever, mock_es_client):
        """Test health check handles errors."""
        mock_es_client.check_connection.side_effect = Exception("Connection error")

        is_healthy = retriever.check_health()

        assert is_healthy is False


class TestKeywordRetrieverFactory:
    """Test suite for KeywordRetriever factory function."""

    def test_create_keyword_retriever_default(self):
        """Test factory function with default parameters."""
        with patch("src.search.retrieval.get_es_client"):
            retriever = create_keyword_retriever()

            assert isinstance(retriever, KeywordRetriever)
            assert retriever.index_name == "messages"
            assert retriever.k1 == 1.5
            assert retriever.b == 0.75

    def test_create_keyword_retriever_custom(self):
        """Test factory function with custom parameters."""
        with patch("src.search.retrieval.get_es_client"):
            retriever = create_keyword_retriever(
                index_name="custom_index",
                k1=2.0,
                b=0.5
            )

            assert retriever.index_name == "custom_index"
            assert retriever.k1 == 2.0
            assert retriever.b == 0.5


class TestKeywordRetrieverIntegration:
    """Integration-style tests for KeywordRetriever."""

    def test_search_returns_bm25_method(self):
        """Test that retrieval method is correctly identified."""
        mock_es_client = MagicMock()
        mock_es_client.check_connection.return_value = True
        mock_es_client.search_messages.return_value = {
            "results": [],
            "total": 0
        }

        with patch("src.search.retrieval.get_es_client", return_value=mock_es_client):
            retriever = KeywordRetriever()
            results = retriever.search(query="test")

            assert results["retrieval_method"] == "bm25"

    def test_multiple_searches_use_same_client(self):
        """Test that multiple searches reuse the same ES client."""
        mock_es_client = MagicMock()
        mock_es_client.check_connection.return_value = True
        mock_es_client.search_messages.return_value = {
            "results": [],
            "total": 0
        }

        with patch("src.search.retrieval.get_es_client", return_value=mock_es_client) as mock_getter:
            retriever = KeywordRetriever()

            # Perform multiple searches
            retriever.search(query="test1")
            retriever.search(query="test2")
            retriever.search(query="test3")

            # ES client should be retrieved only once during initialization
            assert mock_getter.call_count == 1

            # But search should be called multiple times
            assert mock_es_client.search_messages.call_count == 3
