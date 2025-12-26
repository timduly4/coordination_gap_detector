"""
Tests for the gaps API endpoints.

This module contains comprehensive tests for the gap detection API including:
- Gap detection endpoint (/api/v1/gaps/detect)
- List gaps endpoint (/api/v1/gaps)
- Get gap by ID endpoint (/api/v1/gaps/{gap_id})
- Health check endpoint (/api/v1/gaps/health)
- Error handling and validation
- Filtering and pagination
"""
from datetime import datetime, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db, get_vector_store
from src.api.routes.gaps import get_detection_service
from src.models.schemas import (
    CoordinationGap,
    EvidenceItem,
    GapDetectionMetadata,
    GapDetectionRequest,
    GapDetectionResponse,
    GapListRequest,
    GapListResponse,
    LLMVerification,
    TemporalOverlap,
)
from src.services.detection_service import GapDetectionService


@pytest.fixture
def sample_coordination_gap() -> CoordinationGap:
    """Create a sample coordination gap for testing."""
    now = datetime.utcnow()

    return CoordinationGap(
        id="gap_test_123",
        type="duplicate_work",
        title="Two teams building OAuth integration",
        topic="OAuth implementation",
        teams_involved=["platform-team", "auth-team"],
        impact_score=0.85,
        impact_tier="HIGH",
        confidence=0.82,
        evidence=[
            EvidenceItem(
                source="slack",
                content="Starting OAuth implementation in platform",
                timestamp=now - timedelta(days=5),
                relevance_score=0.92,
                team="platform-team",
                author="alice@company.com",
                channel="#platform",
            ),
            EvidenceItem(
                source="github",
                content="PR: Add OAuth2 support to auth service",
                timestamp=now - timedelta(days=4),
                relevance_score=0.89,
                team="auth-team",
                author="bob@company.com",
                url="https://github.com/company/auth/pull/123",
            ),
        ],
        temporal_overlap=TemporalOverlap(
            start=now - timedelta(days=5),
            end=now - timedelta(days=2),
            overlap_days=3,
        ),
        verification=LLMVerification(
            is_duplicate=True,
            confidence=0.85,
            reasoning="Both teams are implementing OAuth2 independently without coordination",
            evidence=["Quote from platform", "Quote from auth"],
            recommendation="Connect alice@company.com and bob@company.com to consolidate efforts",
            overlap_ratio=0.8,
        ),
        insight="Platform and Auth teams are independently implementing OAuth2 with significant temporal overlap",
        recommendation="Consolidate efforts by having one team lead the implementation",
        detected_at=now,
    )


@pytest.fixture
def mock_detection_service(sample_coordination_gap):
    """Create a mock GapDetectionService for testing."""
    mock_service = AsyncMock(spec=GapDetectionService)

    # Mock detect_gaps response
    mock_service.detect_gaps.return_value = GapDetectionResponse(
        gaps=[sample_coordination_gap],
        metadata=GapDetectionMetadata(
            total_gaps=1,
            critical_gaps=0,
            high_gaps=1,
            medium_gaps=0,
            low_gaps=0,
            detection_time_ms=3200,
            messages_analyzed=150,
            clusters_found=5,
        ),
    )

    # Mock list_gaps response
    mock_service.list_gaps.return_value = GapListResponse(
        gaps=[sample_coordination_gap],
        total=1,
        page=1,
        limit=10,
        has_more=False,
    )

    # Mock get_gap_by_id response
    mock_service.get_gap_by_id.return_value = sample_coordination_gap

    # Mock health_check response
    mock_service.health_check.return_value = {
        "status": "healthy",
        "vector_store": {"connected": True},
        "database": {"connected": True},
        "detectors": ["duplicate_work", "missing_context"],
    }

    return mock_service


class TestDetectGapsEndpoint:
    """Tests for POST /api/v1/gaps/detect endpoint."""

    @pytest.mark.asyncio
    async def test_detect_gaps_basic(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test basic gap detection request."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.post(
                "/api/v1/gaps/detect",
                json={
                    "timeframe_days": 30,
                    "sources": ["slack", "github"],
                    "gap_types": ["duplicate_work"],
                    "min_impact_score": 0.7,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Validate response structure
            assert "gaps" in data
            assert "metadata" in data

            # Validate metadata
            assert data["metadata"]["total_gaps"] == 1
            assert data["metadata"]["high_gaps"] == 1
            assert data["metadata"]["messages_analyzed"] == 150
            assert data["metadata"]["clusters_found"] == 5

            # Validate gaps
            assert len(data["gaps"]) == 1
            gap = data["gaps"][0]
            assert gap["type"] == "duplicate_work"
            assert gap["impact_score"] == 0.85
            assert "platform-team" in gap["teams_involved"]
            assert "auth-team" in gap["teams_involved"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_detect_gaps_with_teams_filter(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test gap detection with teams filter."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.post(
                "/api/v1/gaps/detect",
                json={
                    "timeframe_days": 30,
                    "sources": ["slack"],
                    "gap_types": ["duplicate_work"],
                    "teams": ["platform-team", "auth-team"],
                    "min_impact_score": 0.5,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify service was called with teams filter
            mock_detection_service.detect_gaps.assert_called_once()
            call_args = mock_detection_service.detect_gaps.call_args[0][0]
            assert call_args.teams == ["platform-team", "auth-team"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_detect_gaps_with_evidence(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test gap detection with include_evidence flag."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.post(
                "/api/v1/gaps/detect",
                json={
                    "timeframe_days": 30,
                    "sources": ["slack", "github"],
                    "gap_types": ["duplicate_work"],
                    "include_evidence": True,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify evidence is included
            assert len(data["gaps"]) > 0
            gap = data["gaps"][0]
            assert "evidence" in gap
            assert len(gap["evidence"]) == 2

            # Validate evidence structure
            evidence = gap["evidence"][0]
            assert "source" in evidence
            assert "content" in evidence
            assert "timestamp" in evidence
            assert "relevance_score" in evidence
        finally:
            app.dependency_overrides.clear()

    def test_detect_gaps_invalid_timeframe(self, client):
        """Test that invalid timeframe returns validation error."""
        response = client.post(
            "/api/v1/gaps/detect",
            json={
                "timeframe_days": 0,  # Must be >= 1
                "sources": ["slack"],
            },
        )

        # Pydantic validation happens before endpoint is called
        # Just check it's not a successful response
        assert response.status_code != status.HTTP_200_OK

    def test_detect_gaps_invalid_impact_score(self, client):
        """Test that invalid impact score returns validation error."""
        # Test min_impact_score < 0
        response = client.post(
            "/api/v1/gaps/detect",
            json={
                "timeframe_days": 30,
                "sources": ["slack"],
                "min_impact_score": -0.1,
            },
        )
        assert response.status_code != status.HTTP_200_OK

        # Test min_impact_score > 1
        response = client.post(
            "/api/v1/gaps/detect",
            json={
                "timeframe_days": 30,
                "sources": ["slack"],
                "min_impact_score": 1.5,
            },
        )
        assert response.status_code != status.HTTP_200_OK

    def test_detect_gaps_empty_sources(self, client):
        """Test that empty sources list returns validation error."""
        response = client.post(
            "/api/v1/gaps/detect",
            json={
                "timeframe_days": 30,
                "sources": [],  # Must have at least one source
            },
        )

        # Just check it's not a successful response
        assert response.status_code != status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_detect_gaps_no_results(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
    ):
        """Test gap detection with no gaps found."""
        # Mock service with no gaps
        empty_service = AsyncMock(spec=GapDetectionService)
        empty_service.detect_gaps.return_value = GapDetectionResponse(
            gaps=[],
            metadata=GapDetectionMetadata(
                total_gaps=0,
                critical_gaps=0,
                high_gaps=0,
                medium_gaps=0,
                low_gaps=0,
                detection_time_ms=1500,
                messages_analyzed=50,
                clusters_found=0,
            ),
        )

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return empty_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.post(
                "/api/v1/gaps/detect",
                json={
                    "timeframe_days": 7,
                    "sources": ["slack"],
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["metadata"]["total_gaps"] == 0
            assert len(data["gaps"]) == 0
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_detect_gaps_service_error(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
    ):
        """Test gap detection handles service errors."""
        # Mock service that raises exception
        error_service = AsyncMock(spec=GapDetectionService)
        error_service.detect_gaps.side_effect = Exception("Internal service error")

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return error_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.post(
                "/api/v1/gaps/detect",
                json={
                    "timeframe_days": 30,
                    "sources": ["slack"],
                },
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_detect_gaps_invalid_gap_type(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
    ):
        """Test gap detection with invalid gap type returns 400."""
        # Mock service that raises ValueError for invalid gap type
        error_service = AsyncMock(spec=GapDetectionService)
        error_service.detect_gaps.side_effect = ValueError("Invalid gap type: unknown_type")

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return error_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.post(
                "/api/v1/gaps/detect",
                json={
                    "timeframe_days": 30,
                    "sources": ["slack"],
                    "gap_types": ["unknown_type"],
                },
            )

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            data = response.json()
            assert "Invalid request" in data["detail"]
        finally:
            app.dependency_overrides.clear()


class TestListGapsEndpoint:
    """Tests for GET /api/v1/gaps endpoint."""

    @pytest.mark.asyncio
    async def test_list_gaps_basic(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test basic list gaps request."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get("/api/v1/gaps")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Validate response structure
            assert "gaps" in data
            assert "total" in data
            assert "page" in data
            assert "limit" in data
            assert "has_more" in data

            # Validate values
            assert data["total"] == 1
            assert data["page"] == 1
            assert data["limit"] == 10
            assert data["has_more"] is False
            assert len(data["gaps"]) == 1
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_gaps_with_gap_type_filter(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test list gaps with gap type filter."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get(
                "/api/v1/gaps",
                params={"gap_type": "duplicate_work"}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify service was called with gap_type filter
            mock_detection_service.list_gaps.assert_called_once()
            call_args = mock_detection_service.list_gaps.call_args[0][0]
            assert call_args.gap_type == "duplicate_work"
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_gaps_with_min_impact_score(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test list gaps with minimum impact score filter."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get(
                "/api/v1/gaps",
                params={"min_impact_score": 0.8}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # All returned gaps should have impact >= 0.8
            for gap in data["gaps"]:
                assert gap["impact_score"] >= 0.8
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_gaps_with_teams_filter(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test list gaps with teams filter."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get(
                "/api/v1/gaps",
                params={"teams": ["platform-team", "auth-team"]}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify service was called with teams filter
            mock_detection_service.list_gaps.assert_called_once()
            call_args = mock_detection_service.list_gaps.call_args[0][0]
            assert call_args.teams == ["platform-team", "auth-team"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_gaps_pagination(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test list gaps with pagination."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get(
                "/api/v1/gaps",
                params={"page": 2, "limit": 5}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify pagination parameters
            assert data["page"] == 1  # Mock returns page 1
            assert data["limit"] == 10  # Mock returns limit 10

            # Verify service was called with correct pagination
            mock_detection_service.list_gaps.assert_called_once()
            call_args = mock_detection_service.list_gaps.call_args[0][0]
            assert call_args.page == 2
            assert call_args.limit == 5
        finally:
            app.dependency_overrides.clear()

    def test_list_gaps_invalid_page(self, client):
        """Test that invalid page number returns validation error."""
        response = client.get(
            "/api/v1/gaps",
            params={"page": 0}  # Must be >= 1
        )

        assert response.status_code != status.HTTP_200_OK

    def test_list_gaps_invalid_limit(self, client):
        """Test that invalid limit returns validation error."""
        # Test limit = 0
        response = client.get(
            "/api/v1/gaps",
            params={"limit": 0}
        )
        assert response.status_code != status.HTTP_200_OK

        # Test limit > 100
        response = client.get(
            "/api/v1/gaps",
            params={"limit": 101}
        )
        assert response.status_code != status.HTTP_200_OK

    def test_list_gaps_invalid_min_impact_score(self, client):
        """Test that invalid min_impact_score returns validation error."""
        # Test < 0
        response = client.get(
            "/api/v1/gaps",
            params={"min_impact_score": -0.5}
        )
        assert response.status_code != status.HTTP_200_OK

        # Test > 1
        response = client.get(
            "/api/v1/gaps",
            params={"min_impact_score": 1.5}
        )
        assert response.status_code != status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_list_gaps_service_error(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
    ):
        """Test list gaps handles service errors."""
        # Mock service that raises exception
        error_service = AsyncMock(spec=GapDetectionService)
        error_service.list_gaps.side_effect = Exception("Internal service error")

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return error_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get("/api/v1/gaps")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data
        finally:
            app.dependency_overrides.clear()


class TestGetGapByIdEndpoint:
    """Tests for GET /api/v1/gaps/{gap_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_gap_by_id_success(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test get gap by ID returns the gap."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get("/api/v1/gaps/gap_test_123")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Validate gap structure
            assert data["id"] == "gap_test_123"
            assert data["type"] == "duplicate_work"
            assert data["impact_score"] == 0.85
            assert "evidence" in data
            assert "verification" in data
            assert "temporal_overlap" in data
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_gap_by_id_not_found(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
    ):
        """Test get gap by ID returns 404 when gap not found."""
        # Mock service that returns None
        not_found_service = AsyncMock(spec=GapDetectionService)
        not_found_service.get_gap_by_id.return_value = None

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return not_found_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get("/api/v1/gaps/nonexistent_gap")

            assert response.status_code == status.HTTP_404_NOT_FOUND
            data = response.json()
            assert "detail" in data
            assert "nonexistent_gap" in data["detail"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_get_gap_by_id_service_error(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
    ):
        """Test get gap by ID handles service errors."""
        # Mock service that raises exception
        error_service = AsyncMock(spec=GapDetectionService)
        error_service.get_gap_by_id.side_effect = Exception("Internal service error")

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return error_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get("/api/v1/gaps/gap_test_123")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data
        finally:
            app.dependency_overrides.clear()


class TestGapsHealthEndpoint:
    """Tests for GET /api/v1/gaps/health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
        mock_detection_service,
    ):
        """Test health check returns healthy status."""
        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return mock_detection_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get("/api/v1/gaps/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert "status" in data
            assert data["status"] == "healthy"
            assert "vector_store" in data
            assert "database" in data
            assert "detectors" in data
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(
        self,
        app,
        client,
        async_db_session,
        mock_vector_store,
    ):
        """Test health check returns unhealthy status on error."""
        # Mock service that raises exception
        error_service = AsyncMock(spec=GapDetectionService)
        error_service.health_check.side_effect = Exception("Service unavailable")

        async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
            yield async_db_session

        def override_get_vector_store():
            return mock_vector_store

        def override_get_detection_service():
            return error_service

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_vector_store] = override_get_vector_store
        app.dependency_overrides[get_detection_service] = override_get_detection_service

        try:
            response = client.get("/api/v1/gaps/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            assert data["status"] == "unhealthy"
            assert "error" in data
        finally:
            app.dependency_overrides.clear()


class TestGapDetectionServiceIntegration:
    """Integration tests for GapDetectionService."""

    @pytest.mark.asyncio
    async def test_detection_service_integration(
        self,
        async_db_session,
        mock_vector_store,
        sample_coordination_gap,
    ):
        """Test GapDetectionService with mocked dependencies."""
        # This is a placeholder for integration tests
        # In a real scenario, you'd test the actual service implementation
        from src.services.detection_service import GapDetectionService

        service = GapDetectionService(
            vector_store=mock_vector_store,
            db_session=async_db_session,
        )

        # Verify service was created successfully
        assert service is not None
        assert service.vector_store == mock_vector_store
        assert service.db_session == async_db_session
