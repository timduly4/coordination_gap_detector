"""
Integration tests for end-to-end search flow.

This module tests the complete search pipeline from API request through
business logic to database and vector store operations.

Tests cover:
- Full search request lifecycle
- Multi-component integration (API -> Service -> DB -> Vector Store)
- Data flow through the entire system
- Real-world usage scenarios
- Performance characteristics
"""
from datetime import datetime, timedelta
from typing import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db, get_vector_db
from src.db.models import Message, Source
from src.db.vector_store import VectorStore
from src.models.schemas import SearchRequest
from src.services.search_service import SearchService


class TestSearchFlowIntegration:
    """End-to-end integration tests for search functionality."""

    @pytest.mark.asyncio
    async def test_complete_search_flow(
        self,
        app,
        client: TestClient,
        async_db_session: AsyncSession,
        sample_messages: list[Message],
        mock_vector_store,
    ):
        """
        Test complete search flow from HTTP request to response.

        Flow:
        1. Client makes POST request to /api/v1/search/
        2. API layer validates request
        3. SearchService coordinates search
        4. VectorStore performs semantic search
        5. Database enriches results
        6. Response is formatted and returned
        """
        # Override dependencies
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            # Make search request
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth implementation",
                    "limit": 10,
                    "threshold": 0.5,
                },
            )

            # Verify response
            assert response.status_code == 200
            data = response.json()

            # Validate response structure
            assert "results" in data
            assert "total" in data
            assert "query" in data
            assert "query_time_ms" in data
            assert "threshold" in data

            # Validate response data
            assert data["query"] == "OAuth implementation"
            assert data["threshold"] == 0.5
            assert isinstance(data["results"], list)
            assert isinstance(data["query_time_ms"], int)
            assert data["query_time_ms"] >= 0

            # Validate result items
            for result in data["results"]:
                assert "content" in result
                assert "source" in result
                assert "channel" in result
                assert "author" in result
                assert "timestamp" in result
                assert "score" in result
                assert 0.0 <= result["score"] <= 1.0

        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_search_with_database_filtering(
        self,
        app,
        client: TestClient,
        async_db_session: AsyncSession,
        sample_messages: list[Message],
        mock_vector_store,
    ):
        """
        Test search with database-level filtering.

        This tests the integration between vector store results and
        database filtering capabilities.
        """
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            # Search with source type filter
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth",
                    "limit": 10,
                    "source_types": ["slack"],
                },
            )

            assert response.status_code == 200
            data = response.json()

            # All results should be from Slack
            for result in data["results"]:
                assert result["source"] == "slack"

        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_search_service_with_real_database(
        self,
        async_db_session: AsyncSession,
        sample_messages: list[Message],
        mock_vector_store,
    ):
        """
        Test SearchService integration with real database session.

        This bypasses the API layer to test the service directly
        with a real async database session.
        """
        service = SearchService(mock_vector_store)

        request = SearchRequest(
            query="OAuth implementation",
            limit=5,
            threshold=0.6,
        )

        response = await service.search(request, async_db_session)

        # Verify response
        assert response.query == "OAuth implementation"
        assert response.threshold == 0.6
        assert isinstance(response.results, list)
        assert response.query_time_ms >= 0

    @pytest.mark.asyncio
    async def test_search_with_date_range_filtering(
        self,
        app,
        client: TestClient,
        async_db_session: AsyncSession,
        sample_messages: list[Message],
        mock_vector_store,
    ):
        """
        Test search with temporal filtering.

        Verifies that date range filters are applied correctly
        throughout the search pipeline.
        """
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            now = datetime.utcnow()
            date_from = (now - timedelta(hours=2)).isoformat()
            date_to = now.isoformat()

            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth",
                    "date_from": date_from,
                    "date_to": date_to,
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Verify all results are within date range
            for result in data["results"]:
                timestamp = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
                assert timestamp >= datetime.fromisoformat(date_from)
                assert timestamp <= datetime.fromisoformat(date_to)

        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_multi_filter_search_integration(
        self,
        app,
        client: TestClient,
        async_db_session: AsyncSession,
        sample_messages: list[Message],
        mock_vector_store,
    ):
        """
        Test search with multiple filters applied simultaneously.

        This tests the complex interaction between:
        - Source type filtering
        - Channel filtering
        - Date range filtering
        - Similarity threshold
        - Result limit
        """
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            now = datetime.utcnow()
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth",
                    "limit": 3,
                    "threshold": 0.7,
                    "source_types": ["slack"],
                    "channels": ["#platform"],
                    "date_from": (now - timedelta(days=7)).isoformat(),
                    "date_to": now.isoformat(),
                },
            )

            assert response.status_code == 200
            data = response.json()

            # Verify filters were applied
            assert len(data["results"]) <= 3
            for result in data["results"]:
                assert result["source"] == "slack"
                assert result["channel"] == "#platform"
                assert result["score"] >= 0.7

        finally:
            app.dependency_overrides.clear()


class TestVectorStoreIntegration:
    """Integration tests for vector store operations."""

    @pytest.mark.asyncio
    async def test_message_embedding_and_search_flow(self, async_db_session: AsyncSession):
        """
        Test the complete flow of embedding messages and searching them.

        This is a more realistic test that uses actual embedding
        generation (if available) or mocks it appropriately.
        """
        from datetime import datetime

        # Create a real vector store instance (will use in-memory ChromaDB)
        vector_store = VectorStore()

        # Create test source
        source = Source(type="test", name="Test Source", config={})
        async_db_session.add(source)
        await async_db_session.commit()
        await async_db_session.refresh(source)

        # Create test messages
        messages = [
            Message(
                source_id=source.id,
                content="Implementing OAuth2 authentication for our API",
                author="alice@test.com",
                channel="#engineering",
                timestamp=datetime.utcnow(),
                external_id="test_msg_1",
            ),
            Message(
                source_id=source.id,
                content="Discussing database migration strategy",
                author="bob@test.com",
                channel="#engineering",
                timestamp=datetime.utcnow(),
                external_id="test_msg_2",
            ),
            Message(
                source_id=source.id,
                content="OAuth integration with third-party services",
                author="charlie@test.com",
                channel="#platform",
                timestamp=datetime.utcnow(),
                external_id="test_msg_3",
            ),
        ]

        for msg in messages:
            async_db_session.add(msg)
        await async_db_session.commit()

        # Embed messages in vector store
        for msg in messages:
            await async_db_session.refresh(msg)
            vector_store.add_document(
                doc_id=f"msg_{msg.id}",
                content=msg.content,
                metadata={
                    "message_id": msg.id,
                    "source_id": msg.source_id,
                    "channel": msg.channel,
                },
            )

        # Search for OAuth-related messages
        results = vector_store.search(query="OAuth authentication", limit=5)

        # Verify results
        assert len(results) > 0

        # The OAuth-related messages should rank higher
        oauth_content = [content for _, content, _, _ in results if "OAuth" in content]
        assert len(oauth_content) > 0


class TestSearchPerformance:
    """Integration tests for search performance characteristics."""

    @pytest.mark.asyncio
    async def test_search_performance_with_large_result_set(
        self,
        async_db_session: AsyncSession,
        mock_vector_store,
    ):
        """
        Test search performance with a large number of results.

        Verifies that search completes within reasonable time
        even with many matching documents.
        """
        import time
        from datetime import datetime

        # Create a source
        source = Source(type="test", name="Performance Test", config={})
        async_db_session.add(source)
        await async_db_session.commit()
        await async_db_session.refresh(source)

        # Create many messages
        messages = [
            Message(
                source_id=source.id,
                content=f"Message about OAuth implementation number {i}",
                author=f"user{i}@test.com",
                channel="#test",
                timestamp=datetime.utcnow(),
                external_id=f"perf_msg_{i}",
            )
            for i in range(100)
        ]

        async_db_session.add_all(messages)
        await async_db_session.commit()

        # Mock vector store to return many results
        mock_vector_store.search.return_value = [
            (f"msg_{i}", f"Content {i}", 0.9 - (i * 0.001), {"message_id": i})
            for i in range(1, 51)  # 50 results
        ]

        # Perform search and time it
        service = SearchService(mock_vector_store)
        request = SearchRequest(query="OAuth", limit=20)

        start_time = time.time()
        response = await service.search(request, async_db_session)
        elapsed_time = time.time() - start_time

        # Verify performance
        assert elapsed_time < 2.0  # Should complete in under 2 seconds
        assert response.query_time_ms < 2000
        assert len(response.results) <= 20  # Respects limit

    @pytest.mark.asyncio
    async def test_concurrent_search_requests(
        self,
        async_db_session: AsyncSession,
        sample_messages: list[Message],
        mock_vector_store,
    ):
        """
        Test handling of concurrent search requests.

        Verifies that the system can handle multiple simultaneous
        search requests without errors or data corruption.
        """
        import asyncio

        service = SearchService(mock_vector_store)

        # Create multiple search requests
        requests = [
            SearchRequest(query=f"Query {i}", limit=5)
            for i in range(10)
        ]

        # Execute searches concurrently
        tasks = [
            service.search(request, async_db_session)
            for request in requests
        ]

        responses = await asyncio.gather(*tasks)

        # Verify all searches completed successfully
        assert len(responses) == 10
        for i, response in enumerate(responses):
            assert response.query == f"Query {i}"
            assert isinstance(response.results, list)


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""

    @pytest.mark.asyncio
    async def test_database_error_handling(
        self,
        app,
        client: TestClient,
        async_db_session: AsyncSession,
        mock_vector_store,
    ):
        """
        Test that database errors are handled gracefully.

        Verifies that errors in the database layer are caught
        and returned as appropriate HTTP responses.
        """

        def override_get_db():
            # Simulate database error
            raise Exception("Database connection failed")

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            response = client.post(
                "/api/v1/search/",
                json={"query": "test", "limit": 5},
            )

            # Should return 500 error
            assert response.status_code == 500

        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_empty_database_handling(
        self,
        app,
        client: TestClient,
        async_db_session: AsyncSession,
        mock_vector_store,
    ):
        """
        Test search behavior with empty database.

        Verifies that the system handles the case where there
        are no messages in the database gracefully.
        """
        # Mock vector store to return message IDs that don't exist
        mock_vector_store.search.return_value = [
            ("msg_999", "Non-existent message", 0.9, {"message_id": 999}),
        ]

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            response = client.post(
                "/api/v1/search/",
                json={"query": "test", "limit": 5},
            )

            assert response.status_code == 200
            data = response.json()

            # Should return empty results, not error
            assert data["total"] == 0
            assert data["results"] == []

        finally:
            app.dependency_overrides.clear()
