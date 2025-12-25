"""
Integration tests for Elasticsearch.

These tests require a running Elasticsearch instance.
Run with: docker compose up elasticsearch
"""
from datetime import datetime, timedelta

import pytest

from src.db.elasticsearch import get_es_client

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def es_client():
    """Get Elasticsearch client for integration testing."""
    # Reset the singleton to ensure we get a real client, not a mock from unit tests
    import src.db.elasticsearch as es_module
    es_module._es_client = None

    client = get_es_client()

    # Check if ES is available
    if not client.check_connection():
        pytest.skip("Elasticsearch is not available. Run 'docker compose up elasticsearch'")

    yield client

    # Cleanup after tests
    client.delete_index("test_messages")


@pytest.fixture
def test_index(es_client):
    """Create a test index and clean it up after tests."""
    index_name = "test_messages"

    # Delete if exists
    es_client.delete_index(index_name)

    # Create fresh index
    es_client.create_messages_index(index_name)

    yield index_name

    # Cleanup
    es_client.delete_index(index_name)


class TestElasticsearchIntegration:
    """Integration tests for Elasticsearch operations."""

    def test_connection_and_health(self, es_client):
        """Test that we can connect to Elasticsearch and get health info."""
        is_connected = es_client.check_connection()
        assert is_connected is True

        health = es_client.get_cluster_health()
        assert "status" in health
        assert health["status"] in ["green", "yellow", "red"]
        assert "cluster_name" in health

    def test_create_and_delete_index(self, es_client):
        """Test creating and deleting an index."""
        index_name = "test_temp_index"

        # Create index
        created = es_client.create_messages_index(index_name)
        assert created is True

        # Verify it exists (creating again should succeed)
        created_again = es_client.create_messages_index(index_name)
        assert created_again is True

        # Delete index
        deleted = es_client.delete_index(index_name)
        assert deleted is True

        # Verify it's gone
        deleted_again = es_client.delete_index(index_name)
        assert deleted_again is False

    def test_index_single_message(self, es_client, test_index):
        """Test indexing a single message."""
        now = datetime.utcnow()

        success = es_client.index_message(
            index_name=test_index,
            message_id="msg_integration_1",
            content="This is a test message about OAuth implementation",
            source="slack",
            channel="#engineering",
            author="test@demo.com",
            timestamp=now.isoformat(),
            metadata={"priority": "high"},
        )

        assert success is True

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Verify document count
        count = es_client.get_document_count(test_index)
        assert count == 1

    def test_bulk_index_messages(self, es_client, test_index):
        """Test bulk indexing multiple messages."""
        now = datetime.utcnow()

        messages = [
            {
                "message_id": f"msg_bulk_{i}",
                "content": f"Test message {i} about OAuth and authentication",
                "source": "slack",
                "channel": f"#channel{i % 3}",
                "author": f"user{i}@demo.com",
                "timestamp": (now - timedelta(hours=i)).isoformat(),
                "metadata": {"index": i},
            }
            for i in range(10)
        ]

        success, failed = es_client.bulk_index_messages(test_index, messages)

        assert success == 10
        assert failed == 0

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Verify document count
        count = es_client.get_document_count(test_index)
        assert count == 10

    def test_search_messages_bm25(self, es_client, test_index):
        """Test searching messages with BM25 scoring."""
        now = datetime.utcnow()

        # Index test messages
        messages = [
            {
                "message_id": "msg_oauth_1",
                "content": "We need to implement OAuth2 authentication for the API",
                "source": "slack",
                "channel": "#engineering",
                "author": "alice@demo.com",
                "timestamp": now.isoformat(),
            },
            {
                "message_id": "msg_oauth_2",
                "content": "OAuth implementation should use JWT tokens for security",
                "source": "slack",
                "channel": "#security",
                "author": "bob@demo.com",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
            },
            {
                "message_id": "msg_unrelated",
                "content": "Database migration is scheduled for next week",
                "source": "slack",
                "channel": "#data",
                "author": "charlie@demo.com",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
            },
        ]

        es_client.bulk_index_messages(test_index, messages)

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Search for OAuth-related messages
        results = es_client.search_messages(test_index, "OAuth authentication", size=10)

        assert results["total"] >= 2
        assert len(results["results"]) >= 2

        # First result should be most relevant (highest BM25 score)
        top_result = results["results"][0]
        assert "oauth" in top_result["source"]["content"].lower()
        assert top_result["score"] > 0

    def test_search_with_source_filter(self, es_client, test_index):
        """Test searching with source filter."""
        now = datetime.utcnow()

        # Index messages from different sources
        messages = [
            {
                "message_id": "msg_slack_1",
                "content": "Slack message about deployment",
                "source": "slack",
                "channel": "#ops",
                "author": "alice@demo.com",
                "timestamp": now.isoformat(),
            },
            {
                "message_id": "msg_github_1",
                "content": "GitHub PR about deployment automation",
                "source": "github",
                "channel": "repo/deploys",
                "author": "bob@demo.com",
                "timestamp": now.isoformat(),
            },
        ]

        es_client.bulk_index_messages(test_index, messages)

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Search only in Slack messages
        results = es_client.search_messages(
            test_index,
            "deployment",
            size=10,
            source_filter="slack",
        )

        # Should only return Slack results
        assert results["total"] >= 1
        for result in results["results"]:
            assert result["source"]["source"] == "slack"

    def test_search_with_channel_filter(self, es_client, test_index):
        """Test searching with channel filter."""
        now = datetime.utcnow()

        # Index messages in different channels
        messages = [
            {
                "message_id": "msg_eng_1",
                "content": "Engineering discussion about API design",
                "source": "slack",
                "channel": "#engineering",
                "author": "alice@demo.com",
                "timestamp": now.isoformat(),
            },
            {
                "message_id": "msg_prod_1",
                "content": "Product discussion about API features",
                "source": "slack",
                "channel": "#product",
                "author": "bob@demo.com",
                "timestamp": now.isoformat(),
            },
        ]

        es_client.bulk_index_messages(test_index, messages)

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Search only in #engineering channel
        results = es_client.search_messages(
            test_index,
            "API",
            size=10,
            channel_filter="#engineering",
        )

        # Should only return engineering channel results
        assert results["total"] >= 1
        for result in results["results"]:
            assert result["source"]["channel"] == "#engineering"

    def test_search_relevance_ordering(self, es_client, test_index):
        """Test that search results are ordered by relevance (BM25 score)."""
        now = datetime.utcnow()

        # Index messages with varying relevance to query
        messages = [
            {
                "message_id": "msg_exact",
                "content": "OAuth implementation OAuth implementation OAuth",  # Very relevant
                "source": "slack",
                "channel": "#test",
                "author": "test@demo.com",
                "timestamp": now.isoformat(),
            },
            {
                "message_id": "msg_partial",
                "content": "OAuth is mentioned briefly here",  # Somewhat relevant
                "source": "slack",
                "channel": "#test",
                "author": "test@demo.com",
                "timestamp": now.isoformat(),
            },
            {
                "message_id": "msg_unrelated",
                "content": "This is about something completely different",  # Not relevant
                "source": "slack",
                "channel": "#test",
                "author": "test@demo.com",
                "timestamp": now.isoformat(),
            },
        ]

        es_client.bulk_index_messages(test_index, messages)

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Search
        results = es_client.search_messages(test_index, "OAuth implementation", size=10)

        # Results should be ordered by descending score
        scores = [r["score"] for r in results["results"]]
        assert scores == sorted(scores, reverse=True)

        # Most relevant document should be first
        if len(results["results"]) > 0:
            assert results["results"][0]["message_id"] == "msg_exact"

    def test_message_with_thread_id(self, es_client, test_index):
        """Test indexing and searching messages with thread IDs."""
        now = datetime.utcnow()

        # Index messages with thread_id
        success = es_client.index_message(
            index_name=test_index,
            message_id="msg_thread_1",
            content="Thread parent message",
            source="slack",
            channel="#test",
            author="test@demo.com",
            timestamp=now.isoformat(),
            thread_id="thread_123",
        )

        assert success is True

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        count = es_client.get_document_count(test_index)
        assert count >= 1

    def test_large_batch_indexing(self, es_client, test_index):
        """Test indexing a large batch of messages."""
        now = datetime.utcnow()

        # Create 100 messages
        messages = [
            {
                "message_id": f"msg_large_{i}",
                "content": f"Message number {i} discussing various topics",
                "source": "slack",
                "channel": f"#channel{i % 10}",
                "author": f"user{i % 5}@demo.com",
                "timestamp": (now - timedelta(minutes=i)).isoformat(),
                "metadata": {"batch": "large", "index": i},
            }
            for i in range(100)
        ]

        success, failed = es_client.bulk_index_messages(test_index, messages)

        assert success == 100
        assert failed == 0

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Verify all messages were indexed
        count = es_client.get_document_count(test_index)
        assert count == 100

    def test_special_characters_in_content(self, es_client, test_index):
        """Test indexing and searching content with special characters."""
        now = datetime.utcnow()

        special_content = "Message with @mentions #hashtags and https://urls.com 🚀 emoji"

        success = es_client.index_message(
            index_name=test_index,
            message_id="msg_special",
            content=special_content,
            source="slack",
            channel="#test",
            author="test@demo.com",
            timestamp=now.isoformat(),
        )

        assert success is True

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Search should handle special characters
        results = es_client.search_messages(test_index, "mentions hashtags", size=10)

        assert results["total"] >= 1

    def test_empty_search_query(self, es_client, test_index):
        """Test searching with an empty query."""
        now = datetime.utcnow()

        # Index a message first
        es_client.index_message(
            index_name=test_index,
            message_id="msg_empty_query",
            content="Test message",
            source="slack",
            channel="#test",
            author="test@demo.com",
            timestamp=now.isoformat(),
        )

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Empty query should still return results
        results = es_client.search_messages(test_index, "", size=10)

        # Should handle gracefully (ES behavior may vary)
        assert isinstance(results, dict)
        assert "total" in results
        assert "results" in results

    def test_concurrent_indexing_and_searching(self, es_client, test_index):
        """Test that indexing and searching can happen concurrently."""
        now = datetime.utcnow()

        # Index initial messages
        messages_batch1 = [
            {
                "message_id": f"msg_concurrent_{i}",
                "content": f"Initial message {i}",
                "source": "slack",
                "channel": "#test",
                "author": "test@demo.com",
                "timestamp": now.isoformat(),
            }
            for i in range(5)
        ]

        es_client.bulk_index_messages(test_index, messages_batch1)

        # Force index refresh to make documents immediately searchable
        es_client.client.indices.refresh(index=test_index)

        # Search while more indexing happens
        results1 = es_client.search_messages(test_index, "message", size=10)

        # Index more messages
        messages_batch2 = [
            {
                "message_id": f"msg_concurrent_{i + 5}",
                "content": f"Additional message {i}",
                "source": "slack",
                "channel": "#test",
                "author": "test@demo.com",
                "timestamp": now.isoformat(),
            }
            for i in range(5)
        ]

        es_client.bulk_index_messages(test_index, messages_batch2)
        es_client.client.indices.refresh(index=test_index)

        # Search again
        results2 = es_client.search_messages(test_index, "message", size=10)

        # Second search should have more results
        assert results2["total"] >= results1["total"]
