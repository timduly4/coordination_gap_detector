"""
Tests for Elasticsearch client and operations.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
from elasticsearch.exceptions import ConnectionError as ESConnectionError
from elasticsearch.exceptions import NotFoundError

from src.db.elasticsearch import ElasticsearchClient, get_es_client


class TestElasticsearchClient:
    """Test suite for ElasticsearchClient class."""

    @pytest.fixture
    def mock_es_client(self):
        """Create a mock Elasticsearch client."""
        with patch("src.db.elasticsearch.Elasticsearch") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def es_client(self, mock_es_client):
        """Create an ElasticsearchClient instance with mocked ES."""
        # Reset singleton before test
        import src.db.elasticsearch as es_module
        es_module._es_client = None

        with patch("src.db.elasticsearch.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                elasticsearch_url="http://localhost:9200",
                elasticsearch_api_key=None,
            )
            client = ElasticsearchClient()
            yield client

            # Reset singleton after test to avoid polluting other tests
            es_module._es_client = None

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        now = datetime.utcnow()
        return [
            {
                "message_id": "msg_1",
                "content": "We need to implement OAuth2 authentication",
                "source": "slack",
                "channel": "#engineering",
                "author": "alice@demo.com",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "metadata": {"reactions": 5},
            },
            {
                "message_id": "msg_2",
                "content": "OAuth implementation using JWT tokens",
                "source": "slack",
                "channel": "#architecture",
                "author": "bob@demo.com",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "metadata": {"thread_count": 3},
            },
            {
                "message_id": "msg_3",
                "content": "Database migration planning",
                "source": "github",
                "channel": "data-team/repo",
                "author": "charlie@demo.com",
                "timestamp": now.isoformat(),
                "metadata": {},
            },
        ]

    def test_initialization(self, es_client):
        """Test that ElasticsearchClient initializes correctly."""
        assert es_client is not None
        assert es_client.client is not None
        assert es_client.url == "http://localhost:9200"

    def test_initialization_with_api_key(self, mock_es_client):
        """Test initialization with API key."""
        with patch("src.db.elasticsearch.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                elasticsearch_url="http://localhost:9200",
                elasticsearch_api_key="test-api-key",
            )
            client = ElasticsearchClient()
            assert client.api_key == "test-api-key"

    def test_check_connection_success(self, es_client, mock_es_client):
        """Test successful connection check."""
        mock_es_client.info.return_value = {
            "cluster_name": "test-cluster",
            "version": {"number": "8.12.0"},
        }

        is_connected = es_client.check_connection()

        assert is_connected is True
        mock_es_client.info.assert_called_once()

    def test_check_connection_failure(self, es_client, mock_es_client):
        """Test connection check failure."""
        mock_es_client.info.side_effect = ESConnectionError("Connection refused")

        is_connected = es_client.check_connection()

        assert is_connected is False

    def test_check_connection_exception(self, es_client, mock_es_client):
        """Test connection check with generic exception."""
        mock_es_client.info.side_effect = Exception("Unknown error")

        is_connected = es_client.check_connection()

        assert is_connected is False

    def test_get_cluster_health_success(self, es_client, mock_es_client):
        """Test getting cluster health information."""
        mock_es_client.cluster.health.return_value = {
            "status": "green",
            "cluster_name": "test-cluster",
            "number_of_nodes": 1,
            "active_shards": 5,
        }

        health = es_client.get_cluster_health()

        assert health["status"] == "green"
        assert health["cluster_name"] == "test-cluster"
        assert health["number_of_nodes"] == 1
        assert health["active_shards"] == 5

    def test_get_cluster_health_error(self, es_client, mock_es_client):
        """Test cluster health check with error."""
        mock_es_client.cluster.health.side_effect = Exception("Cluster unavailable")

        health = es_client.get_cluster_health()

        assert health["status"] == "error"
        assert "message" in health

    def test_create_messages_index_success(self, es_client, mock_es_client):
        """Test creating messages index."""
        mock_es_client.indices.exists.return_value = False
        mock_es_client.indices.create.return_value = {"acknowledged": True}

        result = es_client.create_messages_index("test_messages")

        assert result is True
        mock_es_client.indices.exists.assert_called_once_with(index="test_messages")
        mock_es_client.indices.create.assert_called_once()

    def test_create_messages_index_already_exists(self, es_client, mock_es_client):
        """Test creating index that already exists."""
        mock_es_client.indices.exists.return_value = True

        result = es_client.create_messages_index("test_messages")

        assert result is True
        mock_es_client.indices.create.assert_not_called()

    def test_create_messages_index_error(self, es_client, mock_es_client):
        """Test index creation with error."""
        mock_es_client.indices.exists.return_value = False
        mock_es_client.indices.create.side_effect = Exception("Creation failed")

        result = es_client.create_messages_index("test_messages")

        assert result is False

    def test_delete_index_success(self, es_client, mock_es_client):
        """Test deleting an index."""
        mock_es_client.indices.exists.return_value = True
        mock_es_client.indices.delete.return_value = {"acknowledged": True}

        result = es_client.delete_index("test_messages")

        assert result is True
        mock_es_client.indices.delete.assert_called_once_with(index="test_messages")

    def test_delete_index_not_found(self, es_client, mock_es_client):
        """Test deleting non-existent index."""
        mock_es_client.indices.exists.return_value = False

        result = es_client.delete_index("test_messages")

        assert result is False
        mock_es_client.indices.delete.assert_not_called()

    def test_delete_index_error(self, es_client, mock_es_client):
        """Test index deletion with error."""
        mock_es_client.indices.exists.return_value = True
        mock_es_client.indices.delete.side_effect = Exception("Deletion failed")

        result = es_client.delete_index("test_messages")

        assert result is False

    def test_index_message_success(self, es_client, mock_es_client, sample_messages):
        """Test indexing a single message."""
        msg = sample_messages[0]
        mock_es_client.index.return_value = {"_id": msg["message_id"], "result": "created"}

        result = es_client.index_message(
            index_name="messages",
            message_id=msg["message_id"],
            content=msg["content"],
            source=msg["source"],
            channel=msg["channel"],
            author=msg["author"],
            timestamp=msg["timestamp"],
            metadata=msg["metadata"],
        )

        assert result is True
        mock_es_client.index.assert_called_once()
        call_args = mock_es_client.index.call_args
        assert call_args.kwargs["index"] == "messages"
        assert call_args.kwargs["id"] == msg["message_id"]

    def test_index_message_with_thread_id(self, es_client, mock_es_client):
        """Test indexing message with thread ID."""
        mock_es_client.index.return_value = {"_id": "msg_1", "result": "created"}

        result = es_client.index_message(
            index_name="messages",
            message_id="msg_1",
            content="Test message",
            source="slack",
            channel="#test",
            author="test@demo.com",
            timestamp=datetime.utcnow().isoformat(),
            thread_id="thread_123",
        )

        assert result is True
        call_args = mock_es_client.index.call_args
        assert call_args.kwargs["document"]["thread_id"] == "thread_123"

    def test_index_message_error(self, es_client, mock_es_client):
        """Test message indexing with error."""
        mock_es_client.index.side_effect = Exception("Indexing failed")

        result = es_client.index_message(
            index_name="messages",
            message_id="msg_1",
            content="Test",
            source="slack",
            channel="#test",
            author="test@demo.com",
            timestamp=datetime.utcnow().isoformat(),
        )

        assert result is False

    def test_bulk_index_messages_success(self, es_client, mock_es_client, sample_messages):
        """Test bulk indexing messages."""
        with patch("elasticsearch.helpers.bulk") as mock_bulk:
            mock_bulk.return_value = (3, [])  # 3 successful, 0 failed

            success, failed = es_client.bulk_index_messages("messages", sample_messages)

            assert success == 3
            assert failed == 0
            mock_bulk.assert_called_once()

    def test_bulk_index_messages_partial_failure(self, es_client, mock_es_client, sample_messages):
        """Test bulk indexing with some failures."""
        with patch("elasticsearch.helpers.bulk") as mock_bulk:
            mock_bulk.return_value = (2, [{"index": {"error": "error"}}])  # 2 success, 1 failed

            success, failed = es_client.bulk_index_messages("messages", sample_messages)

            assert success == 2
            assert failed == 1

    def test_bulk_index_messages_error(self, es_client, mock_es_client, sample_messages):
        """Test bulk indexing with exception."""
        with patch("elasticsearch.helpers.bulk") as mock_bulk:
            mock_bulk.side_effect = Exception("Bulk indexing failed")

            success, failed = es_client.bulk_index_messages("messages", sample_messages)

            assert success == 0
            assert failed == len(sample_messages)

    def test_search_messages_success(self, es_client, mock_es_client):
        """Test searching messages."""
        mock_es_client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "msg_1",
                        "_score": 0.92,
                        "_source": {
                            "content": "OAuth implementation",
                            "source": "slack",
                            "channel": "#engineering",
                        },
                    },
                    {
                        "_id": "msg_2",
                        "_score": 0.85,
                        "_source": {
                            "content": "OAuth with JWT",
                            "source": "slack",
                            "channel": "#architecture",
                        },
                    },
                ],
            }
        }

        results = es_client.search_messages("messages", "OAuth implementation", size=10)

        assert results["total"] == 2
        assert len(results["results"]) == 2
        assert results["results"][0]["message_id"] == "msg_1"
        assert results["results"][0]["score"] == 0.92

    def test_search_messages_with_filters(self, es_client, mock_es_client):
        """Test searching with source and channel filters."""
        mock_es_client.search.return_value = {
            "hits": {"total": {"value": 1}, "hits": []}
        }

        results = es_client.search_messages(
            "messages",
            "OAuth",
            size=10,
            source_filter="slack",
            channel_filter="#engineering",
        )

        call_args = mock_es_client.search.call_args
        query = call_args.kwargs["body"]["query"]

        # Check that filters are in the query
        assert "bool" in query
        assert "filter" in query["bool"]
        assert len(query["bool"]["filter"]) == 2

    def test_search_messages_index_not_found(self, es_client, mock_es_client):
        """Test searching non-existent index."""
        # NotFoundError requires meta and body arguments in ES 8.x
        mock_meta = MagicMock()
        mock_es_client.search.side_effect = NotFoundError("Index not found", mock_meta, {})

        results = es_client.search_messages("nonexistent", "query")

        assert results["total"] == 0
        assert results["results"] == []

    def test_search_messages_error(self, es_client, mock_es_client):
        """Test search with exception."""
        mock_es_client.search.side_effect = Exception("Search failed")

        results = es_client.search_messages("messages", "query")

        assert results["total"] == 0
        assert results["results"] == []

    def test_get_document_count_success(self, es_client, mock_es_client):
        """Test getting document count."""
        mock_es_client.count.return_value = {"count": 42}

        count = es_client.get_document_count("messages")

        assert count == 42
        mock_es_client.count.assert_called_once_with(index="messages")

    def test_get_document_count_index_not_found(self, es_client, mock_es_client):
        """Test document count for non-existent index."""
        # NotFoundError requires meta and body arguments in ES 8.x
        mock_meta = MagicMock()
        mock_es_client.count.side_effect = NotFoundError("Index not found", mock_meta, {})

        count = es_client.get_document_count("nonexistent")

        assert count == 0

    def test_get_document_count_error(self, es_client, mock_es_client):
        """Test document count with exception."""
        mock_es_client.count.side_effect = Exception("Count failed")

        count = es_client.get_document_count("messages")

        assert count == 0

    def test_close_client(self, es_client, mock_es_client):
        """Test closing the Elasticsearch client."""
        es_client.close()

        mock_es_client.close.assert_called_once()

    def test_close_client_error(self, es_client, mock_es_client):
        """Test closing client with error."""
        mock_es_client.close.side_effect = Exception("Close failed")

        # Should not raise exception
        es_client.close()

    def test_singleton_pattern(self):
        """Test that get_es_client returns singleton instance."""
        # Reset the global client
        import src.db.elasticsearch as es_module
        es_module._es_client = None

        with patch("src.db.elasticsearch.ElasticsearchClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            client1 = get_es_client()
            client2 = get_es_client()

            # Should be the same instance
            assert client1 is client2
            # Constructor should only be called once
            assert mock_class.call_count == 1

    def test_messages_index_mapping(self):
        """Test that messages index mapping is properly defined."""
        mapping = ElasticsearchClient.MESSAGES_INDEX_MAPPING

        assert "mappings" in mapping
        assert "settings" in mapping

        properties = mapping["mappings"]["properties"]
        assert "content" in properties
        assert properties["content"]["type"] == "text"
        assert "source" in properties
        assert properties["source"]["type"] == "keyword"
        assert "channel" in properties
        assert "author" in properties
        assert "timestamp" in properties
        assert properties["timestamp"]["type"] == "date"

    def test_index_message_validates_required_fields(self, es_client, mock_es_client):
        """Test that all required fields are included in indexed document."""
        mock_es_client.index.return_value = {"_id": "msg_1", "result": "created"}

        es_client.index_message(
            index_name="messages",
            message_id="msg_1",
            content="Test content",
            source="slack",
            channel="#test",
            author="test@demo.com",
            timestamp=datetime.utcnow().isoformat(),
        )

        call_args = mock_es_client.index.call_args
        document = call_args.kwargs["document"]

        # Verify all required fields are present
        assert "message_id" in document
        assert "content" in document
        assert "source" in document
        assert "channel" in document
        assert "author" in document
        assert "timestamp" in document
        assert "metadata" in document

    def test_search_query_structure(self, es_client, mock_es_client):
        """Test that search builds correct query structure."""
        mock_es_client.search.return_value = {
            "hits": {"total": {"value": 0}, "hits": []}
        }

        es_client.search_messages("messages", "test query", size=5)

        call_args = mock_es_client.search.call_args
        query = call_args.kwargs["body"]["query"]

        # Verify query structure
        assert "bool" in query
        assert "must" in query["bool"]
        assert len(query["bool"]["must"]) > 0
        assert "match" in query["bool"]["must"][0]

    def test_bulk_actions_structure(self, es_client, sample_messages):
        """Test that bulk actions are properly structured."""
        with patch("elasticsearch.helpers.bulk") as mock_bulk:
            mock_bulk.return_value = (len(sample_messages), [])

            es_client.bulk_index_messages("messages", sample_messages)

            call_args = mock_bulk.call_args
            actions = call_args[0][1]  # Second argument is actions list

            # Verify action structure
            assert len(actions) == len(sample_messages)
            for action in actions:
                assert "_index" in action
                assert "_id" in action
                assert "_source" in action
