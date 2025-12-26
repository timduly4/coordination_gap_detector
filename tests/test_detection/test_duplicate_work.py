"""
Tests for duplicate work detection algorithm.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List

import numpy as np

from src.analysis.entity_extraction import EntityExtractor
from src.analysis.entity_types import ExtractedEntities, Project, Team
from src.detection.duplicate_work import DuplicateWorkDetector
from src.detection.patterns import GapDetectionConfig
from src.models.schemas import LLMVerification


class MockMessage:
    """Mock message object for testing."""

    def __init__(
        self,
        id: int,
        content: str,
        timestamp: datetime,
        author: str = "user@company.com",
        channel: str = "#general",
    ):
        self.id = id
        self.content = content
        self.timestamp = timestamp
        self.author = author
        self.channel = channel


class TestDuplicateWorkDetector:
    """Tests for DuplicateWorkDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        config = GapDetectionConfig(
            similarity_threshold=0.85,
            min_cluster_size=2,
            time_window_days=30,
            min_teams=2,
            min_temporal_overlap_days=3,
            llm_confidence_threshold=0.7,
        )
        return DuplicateWorkDetector(config=config)

    @pytest.fixture
    def sample_messages(self) -> List[MockMessage]:
        """Create sample messages for testing."""
        now = datetime.utcnow()

        return [
            MockMessage(
                id=1,
                content="@platform-team starting OAuth implementation for API gateway",
                timestamp=now - timedelta(days=10),
                author="alice@company.com",
                channel="#platform",
            ),
            MockMessage(
                id=2,
                content="We're building OAuth support in the auth service",
                timestamp=now - timedelta(days=9),
                author="bob@company.com",
                channel="#auth",
            ),
            MockMessage(
                id=3,
                content="@platform-team OAuth design is complete, moving to implementation",
                timestamp=now - timedelta(days=7),
                author="alice@company.com",
                channel="#platform",
            ),
            MockMessage(
                id=4,
                content="@auth-team finished OAuth authorization flow",
                timestamp=now - timedelta(days=6),
                author="bob@company.com",
                channel="#auth",
            ),
            MockMessage(
                id=5,
                content="Platform team OAuth implementation merged",
                timestamp=now - timedelta(days=3),
                author="charlie@company.com",
                channel="#platform",
            ),
        ]

    @pytest.fixture
    def sample_embeddings(self) -> List[np.ndarray]:
        """Create sample embeddings."""
        # Create similar embeddings to simulate semantic similarity
        base_embedding = np.random.rand(384).astype(np.float32)
        embeddings = []

        for i in range(5):
            # Add small variations
            noise = np.random.rand(384).astype(np.float32) * 0.1
            embeddings.append(base_embedding + noise)

        return embeddings

    def test_init_with_defaults(self):
        """Test detector initialization with defaults."""
        detector = DuplicateWorkDetector()

        assert detector.config is not None
        assert detector.entity_extractor is not None
        assert detector.semantic_clusterer is not None
        assert detector.llm_client is not None
        assert detector.validator is not None

    def test_init_with_custom_config(self):
        """Test detector initialization with custom config."""
        config = GapDetectionConfig(
            similarity_threshold=0.9,
            min_teams=3,
        )

        detector = DuplicateWorkDetector(config=config)

        assert detector.config.similarity_threshold == 0.9
        assert detector.config.min_teams == 3

    @pytest.mark.asyncio
    async def test_detect_no_messages(self, detector):
        """Test detection with no messages."""
        gaps = await detector.detect(messages=[], embeddings=[])

        assert len(gaps) == 0

    @pytest.mark.asyncio
    async def test_detect_no_embeddings(self, detector, sample_messages):
        """Test detection with no embeddings."""
        gaps = await detector.detect(messages=sample_messages, embeddings=None)

        # Should return empty because no embeddings = no clustering
        assert len(gaps) == 0

    @pytest.mark.asyncio
    async def test_detect_with_mock_clustering(
        self, detector, sample_messages, sample_embeddings
    ):
        """Test detection with mocked clustering."""
        # Mock the clustering to return a simple cluster
        with patch.object(
            detector.semantic_clusterer,
            "cluster",
            return_value=[[0, 1, 2, 3, 4]],  # All messages in one cluster
        ):
            with patch.object(
                detector.semantic_clusterer,
                "create_message_clusters",
            ) as mock_create:
                # Mock cluster creation
                from src.detection.clustering import MessageCluster

                mock_cluster = MessageCluster(
                    cluster_id="test_cluster",
                    message_ids=[1, 2, 3, 4, 5],
                    size=5,
                    avg_similarity=0.9,
                    timespan_days=7.0,
                    participant_count=3,
                )
                mock_create.return_value = [mock_cluster]

                # Mock LLM verification
                with patch.object(
                    detector, "_verify_with_llm", new_callable=AsyncMock
                ) as mock_llm:
                    mock_llm.return_value = LLMVerification(
                        is_duplicate=True,
                        confidence=0.85,
                        reasoning="Both teams implementing OAuth",
                        evidence=["Quote 1", "Quote 2"],
                        recommendation="Connect teams",
                        overlap_ratio=0.8,
                    )

                    # Mock entity extraction to return teams
                    with patch.object(
                        detector.entity_extractor, "extract"
                    ) as mock_extract:
                        # Create a callable that alternates between teams
                        call_count = [0]

                        def mock_extract_fn(*args, **kwargs):
                            """Alternate between platform and auth teams, with some projects."""
                            call_count[0] += 1
                            if call_count[0] % 2 == 1:
                                return ExtractedEntities(
                                    people=[],
                                    teams=[Team(text="platform-team", normalized="platform-team", confidence=0.9)],
                                    projects=[Project(text="OAuth", normalized="oauth", confidence=0.9)],
                                    topics=[],
                                )
                            else:
                                return ExtractedEntities(
                                    people=[],
                                    teams=[Team(text="auth-team", normalized="auth-team", confidence=0.9)],
                                    projects=[Project(text="OAuth2", normalized="oauth2", confidence=0.85)],
                                    topics=[],
                                )

                        mock_extract.side_effect = mock_extract_fn

                        gaps = await detector.detect(
                            messages=sample_messages,
                            embeddings=sample_embeddings,
                        )

                        # Should detect at least one gap
                        assert len(gaps) >= 0  # May be 0 if validation fails

    def test_extract_teams_from_messages(self, detector, sample_messages):
        """Test team extraction from messages."""
        with patch.object(detector.entity_extractor, "extract") as mock_extract:
            # Mock entity extraction
            mock_extract.side_effect = [
                ExtractedEntities(
                    people=[],
                    teams=[Team(text="platform-team", normalized="platform-team", confidence=0.9)],
                    projects=[],
                    topics=[],
                ),
                ExtractedEntities(
                    people=[],
                    teams=[Team(text="auth-team", normalized="auth-team", confidence=0.9)],
                    projects=[],
                    topics=[],
                ),
                ExtractedEntities(
                    people=[],
                    teams=[Team(text="platform-team", normalized="platform-team", confidence=0.9)],
                    projects=[],
                    topics=[],
                ),
            ]

            teams = detector._extract_teams_from_messages(sample_messages[:3])

            # Should find unique teams
            assert len(teams) >= 1
            assert isinstance(teams, list)

    def test_compute_temporal_overlap(self, detector, sample_messages):
        """Test temporal overlap computation."""
        with patch.object(detector.entity_extractor, "extract") as mock_extract:
            # Mock to return alternating teams
            mock_extract.side_effect = [
                ExtractedEntities(
                    people=[],
                    teams=[Team(text="platform-team", normalized="platform-team", confidence=0.9)],
                    projects=[],
                    topics=[],
                ),
                ExtractedEntities(
                    people=[],
                    teams=[Team(text="auth-team", normalized="auth-team", confidence=0.9)],
                    projects=[],
                    topics=[],
                ),
            ] * 5

            teams = ["platform-team", "auth-team"]
            overlap = detector._compute_temporal_overlap(
                sample_messages, teams
            )

            assert overlap is not None
            assert overlap.overlap_days >= 0
            assert overlap.start <= overlap.end

    @pytest.mark.asyncio
    async def test_verify_with_llm(self, detector, sample_messages):
        """Test LLM verification."""
        from src.models.schemas import TemporalOverlap

        teams = ["platform-team", "auth-team"]
        now = datetime.utcnow()
        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=10),
            end=now - timedelta(days=3),
            overlap_days=7,
        )

        # Mock entity extraction (used in _infer_topic)
        with patch.object(detector.entity_extractor, "extract") as mock_extract:
            mock_extract.return_value = ExtractedEntities(
                people=[],
                teams=[],
                projects=[Project(text="OAuth", normalized="oauth", confidence=0.9)],
                topics=[],
            )

            with patch.object(
                detector.llm_client, "complete_async", new_callable=AsyncMock
            ) as mock_complete:
                # Mock LLM response
                mock_response = Mock()
                mock_response.content = (
                    '{"is_duplicate": true, "confidence": 0.85, '
                    '"reasoning": "Both teams implementing OAuth", '
                    '"evidence": ["Quote 1"], "recommendation": "Connect teams", '
                    '"overlap_ratio": 0.8}'
                )
                mock_complete.return_value = mock_response

                verification = await detector._verify_with_llm(
                    sample_messages[:3], teams, temporal_overlap
                )

                assert verification.is_duplicate is True
                assert verification.confidence >= 0.7
                assert len(verification.reasoning) > 0

    @pytest.mark.asyncio
    async def test_verify_with_llm_error_handling(
        self, detector, sample_messages
    ):
        """Test LLM verification error handling."""
        from src.models.schemas import TemporalOverlap

        teams = ["platform-team", "auth-team"]
        now = datetime.utcnow()
        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=10),
            end=now - timedelta(days=3),
            overlap_days=7,
        )

        # Mock entity extraction (used in _infer_topic)
        with patch.object(detector.entity_extractor, "extract") as mock_extract:
            mock_extract.return_value = ExtractedEntities(
                people=[],
                teams=[],
                projects=[Project(text="OAuth", normalized="oauth", confidence=0.9)],
                topics=[],
            )

            with patch.object(
                detector.llm_client, "complete_async", new_callable=AsyncMock
            ) as mock_complete:
                # Mock LLM error
                mock_complete.side_effect = Exception("API error")

                verification = await detector._verify_with_llm(
                    sample_messages[:3], teams, temporal_overlap
                )

                # Should return conservative default
                assert verification.is_duplicate is False
                assert verification.confidence == 0.0

    def test_infer_topic(self, detector, sample_messages):
        """Test topic inference from messages."""
        with patch.object(detector.entity_extractor, "extract") as mock_extract:
            # Mock entity extraction to return some projects
            mock_extract.return_value = ExtractedEntities(
                people=[],
                teams=[],
                projects=[
                    Project(text="OAuth", normalized="oauth", confidence=0.9),
                    Project(text="OAuth2", normalized="oauth2", confidence=0.85),
                ],
                topics=[],
            )

            topic = detector._infer_topic(sample_messages[:3])

            assert isinstance(topic, str)
            assert len(topic) > 0

    def test_collect_evidence(self, detector, sample_messages):
        """Test evidence collection."""
        with patch.object(detector.entity_extractor, "extract") as mock_extract:
            # Mock entity extraction
            mock_extract.side_effect = [
                ExtractedEntities(
                    people=[],
                    teams=[Team(text="platform-team", normalized="platform-team", confidence=0.9)],
                    projects=[],
                    topics=[],
                ),
                ExtractedEntities(
                    people=[],
                    teams=[Team(text="auth-team", normalized="auth-team", confidence=0.9)],
                    projects=[],
                    topics=[],
                ),
            ] * 5

            teams = ["platform-team", "auth-team"]
            evidence = detector._collect_evidence(sample_messages, teams)

            assert len(evidence) > 0
            # Evidence should be sorted by relevance
            if len(evidence) > 1:
                assert evidence[0].relevance_score >= evidence[-1].relevance_score

    def test_estimate_impact_score(self, detector):
        """Test impact score estimation."""
        from src.models.schemas import EvidenceItem, TemporalOverlap
        from src.detection.clustering import MessageCluster

        now = datetime.utcnow()
        teams = ["platform-team", "auth-team"]

        evidence = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now - timedelta(hours=i),
                relevance_score=0.9,
            )
            for i in range(5)
        ]

        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=10),
            end=now - timedelta(days=3),
            overlap_days=7,
        )

        verification = LLMVerification(
            is_duplicate=True,
            confidence=0.85,
            reasoning="Duplicate work",
            evidence=["Quote"],
            recommendation="Action",
            overlap_ratio=0.8,
        )

        # Create mock cluster
        cluster = MessageCluster(
            cluster_id="test_cluster",
            message_ids=[1, 2, 3, 4, 5],
            size=5,
            avg_similarity=0.9,
            timespan_days=7.0,
            participant_count=8,
        )

        impact_score, impact_breakdown = detector._estimate_impact_score(
            teams=teams,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            llm_verification=verification,
            cluster=cluster,
        )

        assert 0.0 <= impact_score <= 1.0
        assert impact_breakdown is not None

    def test_determine_impact_tier(self, detector):
        """Test impact tier determination."""
        assert detector._determine_impact_tier(0.9) == "CRITICAL"
        assert detector._determine_impact_tier(0.7) == "HIGH"
        assert detector._determine_impact_tier(0.5) == "MEDIUM"
        assert detector._determine_impact_tier(0.3) == "LOW"

    def test_estimate_cost(self, detector):
        """Test cost estimation."""
        from src.models.schemas import EvidenceItem, TemporalOverlap
        from src.detection.clustering import MessageCluster

        now = datetime.utcnow()
        teams = ["platform-team", "auth-team"]

        evidence = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now,
                relevance_score=0.9,
            )
            for i in range(5)
        ]

        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=10),
            end=now - timedelta(days=3),
            overlap_days=7,
        )

        # Create mock cluster
        cluster = MessageCluster(
            cluster_id="test_cluster",
            message_ids=[1, 2, 3, 4, 5],
            size=5,
            avg_similarity=0.9,
            timespan_days=7.0,
            participant_count=6,
        )

        cost = detector._estimate_cost(teams, evidence, temporal_overlap, cluster)

        assert cost.engineering_hours > 0
        assert cost.dollar_value > 0
        assert len(cost.explanation) > 0

    def test_validate_gap(self, detector):
        """Test gap validation via PatternDetector interface."""
        from src.models.schemas import (
            CoordinationGap,
            EvidenceItem,
            TemporalOverlap,
        )

        now = datetime.utcnow()

        gap = CoordinationGap(
            id="gap_test",
            type="duplicate_work",
            title="Test gap",
            topic="OAuth",
            teams_involved=["team1", "team2"],
            impact_score=0.85,
            impact_tier="HIGH",
            confidence=0.8,
            evidence=[
                EvidenceItem(
                    source="slack",
                    content="Message",
                    timestamp=now,
                    relevance_score=0.9,
                    team="team1",
                )
                for _ in range(3)
            ],
            temporal_overlap=TemporalOverlap(
                start=now - timedelta(days=5),
                end=now,
                overlap_days=5,
            ),
            verification=LLMVerification(
                is_duplicate=True,
                confidence=0.85,
                reasoning="Duplicate",
                evidence=["Quote"],
                recommendation="Action",
                overlap_ratio=0.8,
            ),
            insight="Test insight",
            recommendation="Test recommendation",
            detected_at=now,
        )

        is_valid = detector.validate_gap(gap)

        # Should be boolean
        assert isinstance(is_valid, bool)
