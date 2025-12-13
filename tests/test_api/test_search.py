"""
Tests for the search API endpoint.

This module contains comprehensive tests for the search API including:
- Successful search queries
- Filtering by various parameters
- Error handling
- Edge cases
- Health checks
"""
from datetime import datetime, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db, get_vector_db
from src.api.routes.search import get_service
from src.models.schemas import SearchRequest, SearchResponse
from src.services.search_service import SearchService


class TestSearchEndpoint:
    """Tests for the /api/v1/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_basic_query(
        self,
        app,
        client,
        async_db_session,
        sample_messages,
        mock_vector_store,
    ):
        """Test basic search query returns results."""
        # Override dependencies using FastAPI's dependency_overrides
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            # Make the search request
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth implementation",
                    "limit": 10,
                    "threshold": 0.7,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Validate response structure
            assert "results" in data
            assert "total" in data
            assert "query" in data
            assert "query_time_ms" in data
            assert "threshold" in data

            # Validate query echo
            assert data["query"] == "OAuth implementation"
            assert data["threshold"] == 0.7
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        app,
        client,
        async_db_session,
        sample_messages,
        mock_vector_store,
    ):
        """Test search with source type and channel filters."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth",
                    "limit": 5,
                    "source_types": ["slack"],
                    "channels": ["#platform"],
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # All results should be from specified source and channel
            for result in data["results"]:
                assert result["source"] == "slack"
                assert result["channel"] == "#platform"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_search_with_date_range(
        self,
        app,
        client,
        async_db_session,
        sample_messages,
        mock_vector_store,
    ):
        """Test search with date range filter."""
        now = datetime.utcnow()
        date_from = (now - timedelta(hours=3)).isoformat()
        date_to = now.isoformat()

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth",
                    "date_from": date_from,
                    "date_to": date_to,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # All results should be within date range
            for result in data["results"]:
                result_time = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
                assert result_time >= datetime.fromisoformat(date_from)
                assert result_time <= datetime.fromisoformat(date_to)
        finally:
            app.dependency_overrides.clear()

    def test_search_empty_query(self, client):
        """Test that empty query returns validation error."""
        response = client.post(
            "/api/v1/search/",
            json={
                "query": "",
                "limit": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_query_too_long(self, client):
        """Test that excessively long query returns validation error."""
        long_query = "a" * 1001  # Max is 1000 characters

        response = client.post(
            "/api/v1/search/",
            json={
                "query": long_query,
                "limit": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_invalid_limit(self, client):
        """Test that invalid limit values return validation error."""
        # Test limit = 0
        response = client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "limit": 0,
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test limit > 100
        response = client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "limit": 101,
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_invalid_threshold(self, client):
        """Test that invalid threshold values return validation error."""
        # Test threshold < 0
        response = client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "threshold": -0.1,
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test threshold > 1
        response = client.post(
            "/api/v1/search/",
            json={
                "query": "test",
                "threshold": 1.1,
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_no_results(
        self,
        app,
        client,
        async_db_session,
        sample_messages,
    ):
        """Test search with no matching results."""
        # Mock vector store with no results
        mock_store = MagicMock()
        mock_store.search.return_value = []

        # Create service with the empty mock store
        empty_service = SearchService(mock_store)

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_service():
            return empty_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_service] = override_get_service

        try:
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "nonexistent query",
                    "limit": 10,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["total"] == 0
            assert len(data["results"]) == 0
            assert data["query"] == "nonexistent query"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self,
        app,
        client,
        async_db_session,
        sample_messages,
        mock_vector_store,
    ):
        """Test that search respects the limit parameter."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth",
                    "limit": 1,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Should return at most the requested limit
            assert len(data["results"]) <= 1
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_score(
        self,
        app,
        client,
        async_db_session,
        sample_messages,
        mock_vector_store,
    ):
        """Test that search results are sorted by score (descending)."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            response = client.post(
                "/api/v1/search/",
                json={
                    "query": "OAuth",
                    "limit": 10,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            if len(data["results"]) > 1:
                # Check that results are sorted by score (descending)
                scores = [result["score"] for result in data["results"]]
                assert scores == sorted(scores, reverse=True)
        finally:
            app.dependency_overrides.clear()

    def test_search_missing_query(self, client):
        """Test that missing query parameter returns validation error."""
        response = client.post(
            "/api/v1/search/",
            json={
                "limit": 10,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_whitespace_query(self, client):
        """Test that whitespace-only query returns validation error."""
        response = client.post(
            "/api/v1/search/",
            json={
                "query": "   ",
                "limit": 10,
            },
        )

        # Should return validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSearchHealthEndpoint:
    """Tests for the /api/v1/search/health endpoint."""

    def test_health_check_success(self, app, client, mock_vector_store):
        """Test health check returns success when all services are healthy."""
        def override_get_vector_db():
            return mock_vector_store

        app.dependency_overrides[get_vector_db] = override_get_vector_db

        try:
            response = client.get("/api/v1/search/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "status" in data
            assert "vector_store" in data
            assert data["status"] == "healthy"
            assert data["vector_store"]["connected"] is True
        finally:
            app.dependency_overrides.clear()

    def test_health_check_degraded(self, app, client):
        """Test health check returns degraded status when vector store is down."""
        mock_store = MagicMock()
        mock_store.check_connection.return_value = False
        mock_store.get_collection_count.return_value = 0
        mock_store.collection_name = "test_collection"

        # Create service with the degraded mock store
        degraded_service = SearchService(mock_store)

        def override_get_service():
            return degraded_service

        app.dependency_overrides[get_service] = override_get_service

        try:
            response = client.get("/api/v1/search/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["status"] == "degraded"
            assert data["vector_store"]["connected"] is False
        finally:
            app.dependency_overrides.clear()


class TestSearchServiceIntegration:
    """Integration tests for SearchService with database."""

    @pytest.mark.asyncio
    async def test_search_service_integration(
        self,
        async_db_session,
        sample_messages,
        mock_vector_store,
    ):
        """Test SearchService with real database session."""
        from src.services.search_service import SearchService

        service = SearchService(mock_vector_store)

        request = SearchRequest(
            query="OAuth implementation",
            limit=10,
            threshold=0.5,
        )

        response = await service.search(request, async_db_session)

        assert isinstance(response, SearchResponse)
        assert response.query == "OAuth implementation"
        assert response.threshold == 0.5
        assert response.query_time_ms >= 0

    @pytest.mark.asyncio
    async def test_search_service_health_check(self, mock_vector_store):
        """Test SearchService health check."""
        from src.services.search_service import SearchService

        service = SearchService(mock_vector_store)
        health = await service.health_check()

        assert "status" in health
        assert "vector_store" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
