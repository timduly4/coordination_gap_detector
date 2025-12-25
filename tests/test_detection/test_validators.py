"""
Tests for gap validation logic.
"""

import pytest
from datetime import datetime, timedelta

from src.detection.validators import (
    DuplicateWorkValidator,
    GapValidator,
)
from src.models.schemas import (
    CoordinationGap,
    EvidenceItem,
    LLMVerification,
    TemporalOverlap,
)


class TestGapValidator:
    """Tests for GapValidator base class."""

    def test_init_with_defaults(self):
        """Test validator initialization with defaults."""
        validator = GapValidator()

        assert validator.min_confidence == 0.7
        assert validator.min_teams == 2
        assert validator.min_evidence_items == 2
        assert validator.min_temporal_overlap_days == 1
        assert validator.require_llm_verification is True

    def test_init_with_custom_params(self):
        """Test validator initialization with custom parameters."""
        validator = GapValidator(
            min_confidence=0.8,
            min_teams=3,
            min_evidence_items=5,
            min_temporal_overlap_days=7,
            require_llm_verification=False,
        )

        assert validator.min_confidence == 0.8
        assert validator.min_teams == 3
        assert validator.min_evidence_items == 5
        assert validator.min_temporal_overlap_days == 7
        assert validator.require_llm_verification is False

    def test_validate_valid_gap(self):
        """Test validation of a valid gap."""
        validator = GapValidator(
            min_confidence=0.7,
            min_teams=2,
            min_evidence_items=2,
            min_temporal_overlap_days=1,
        )

        gap = self._create_valid_gap()

        is_valid, failures = validator.validate(gap)

        assert is_valid is True
        assert len(failures) == 0

    def test_validate_low_confidence(self):
        """Test validation failure for low confidence."""
        validator = GapValidator(min_confidence=0.9)

        gap = self._create_valid_gap()
        gap.confidence = 0.5  # Below threshold

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("Confidence" in f for f in failures)

    def test_validate_insufficient_teams(self):
        """Test validation failure for insufficient teams."""
        validator = GapValidator(min_teams=3)

        gap = self._create_valid_gap()
        gap.teams_involved = ["team1"]  # Only 1 team

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("teams" in f.lower() for f in failures)

    def test_validate_insufficient_evidence(self):
        """Test validation failure for insufficient evidence."""
        validator = GapValidator(min_evidence_items=5)

        gap = self._create_valid_gap()
        gap.evidence = gap.evidence[:1]  # Only 1 evidence item

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("evidence" in f.lower() for f in failures)

    def test_validate_insufficient_temporal_overlap(self):
        """Test validation failure for insufficient temporal overlap."""
        validator = GapValidator(min_temporal_overlap_days=10)

        gap = self._create_valid_gap()
        gap.temporal_overlap.overlap_days = 3  # Below threshold

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("overlap" in f.lower() for f in failures)

    def test_validate_missing_llm_verification(self):
        """Test validation failure for missing LLM verification."""
        validator = GapValidator(require_llm_verification=True)

        gap = self._create_valid_gap()
        gap.verification = None

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("LLM verification missing" in f for f in failures)

    def test_validate_llm_verification_failed(self):
        """Test validation failure when LLM says not duplicate."""
        validator = GapValidator(require_llm_verification=True)

        gap = self._create_valid_gap()
        gap.verification.is_duplicate = False

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("LLM verification failed" in f for f in failures)

    def test_has_collaboration_indicators(self):
        """Test detection of collaboration indicators."""
        validator = GapValidator()

        gap = self._create_valid_gap()
        gap.evidence[0].content = "We're working with the platform team on this"

        # Should be excluded due to collaboration
        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("Collaboration indicators" in f for f in failures)

    def test_has_cross_team_mentions(self):
        """Test detection of cross-team mentions."""
        validator = GapValidator()

        gap = self._create_valid_gap()
        gap.evidence[0].content = "Coordinating with @auth-team on OAuth"
        gap.evidence[0].team = "platform-team"

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("Collaboration" in f for f in failures)

    def test_is_same_team_work(self):
        """Test detection of same-team work."""
        validator = GapValidator()

        gap = self._create_valid_gap()
        # Set all evidence to same team
        for evidence in gap.evidence:
            evidence.team = "platform-team"

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("same team" in f.lower() for f in failures)

    def test_is_intentional_redundancy(self):
        """Test detection of intentional redundancy."""
        validator = GapValidator()

        gap = self._create_valid_gap()
        gap.insight = "This is an A/B test to compare two approaches"

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("redundancy" in f.lower() for f in failures)

    def test_validate_evidence_quality_valid(self):
        """Test evidence quality validation with valid evidence."""
        validator = GapValidator()

        evidence = [
            EvidenceItem(
                source="slack",
                content="Message 1",
                timestamp=datetime.utcnow() - timedelta(hours=2),
                relevance_score=0.8,
            ),
            EvidenceItem(
                source="slack",
                content="Message 2",
                timestamp=datetime.utcnow() - timedelta(hours=1),
                relevance_score=0.9,
            ),
        ]

        is_valid, failure = validator.validate_evidence_quality(evidence)

        assert is_valid is True
        assert failure is None

    def test_validate_evidence_quality_low_relevance(self):
        """Test evidence quality validation with low relevance."""
        validator = GapValidator()

        evidence = [
            EvidenceItem(
                source="slack",
                content="Message 1",
                timestamp=datetime.utcnow(),
                relevance_score=0.3,
            ),
            EvidenceItem(
                source="slack",
                content="Message 2",
                timestamp=datetime.utcnow(),
                relevance_score=0.4,
            ),
        ]

        is_valid, failure = validator.validate_evidence_quality(
            evidence, min_avg_relevance=0.5
        )

        assert is_valid is False
        assert "relevance" in failure.lower()

    def test_validate_evidence_quality_no_evidence(self):
        """Test evidence quality validation with no evidence."""
        validator = GapValidator()

        is_valid, failure = validator.validate_evidence_quality([])

        assert is_valid is False
        assert "No evidence" in failure

    def test_validate_llm_verification_valid(self):
        """Test LLM verification validation with valid verification."""
        validator = GapValidator()

        verification = LLMVerification(
            is_duplicate=True,
            confidence=0.85,
            reasoning="Both teams are implementing the same OAuth flow",
            evidence=["Quote 1", "Quote 2"],
            recommendation="Connect teams immediately",
            overlap_ratio=0.9,
        )

        is_valid, failure = validator.validate_llm_verification(verification)

        assert is_valid is True
        assert failure is None

    def test_validate_llm_verification_not_duplicate(self):
        """Test LLM verification validation when not duplicate."""
        validator = GapValidator()

        verification = LLMVerification(
            is_duplicate=False,
            confidence=0.85,
            reasoning="Different scopes",
            evidence=[],
            recommendation="No action needed",
            overlap_ratio=0.2,
        )

        is_valid, failure = validator.validate_llm_verification(verification)

        assert is_valid is False
        assert "not duplicate work" in failure

    def test_validate_llm_verification_low_confidence(self):
        """Test LLM verification validation with low confidence."""
        validator = GapValidator()

        verification = LLMVerification(
            is_duplicate=True,
            confidence=0.5,
            reasoning="Might be duplicate",
            evidence=["Quote"],
            recommendation="Investigate",
            overlap_ratio=0.6,
        )

        is_valid, failure = validator.validate_llm_verification(verification, min_confidence=0.7)

        assert is_valid is False
        assert "confidence" in failure.lower()

    # Helper methods

    def _create_valid_gap(self) -> CoordinationGap:
        """Create a valid gap for testing."""
        now = datetime.utcnow()

        evidence = [
            EvidenceItem(
                source="slack",
                message_id=1,
                channel="#platform",
                author="alice@company.com",
                content="Starting OAuth implementation",
                timestamp=now - timedelta(days=5),
                relevance_score=0.9,
                team="platform-team",
            ),
            EvidenceItem(
                source="slack",
                message_id=2,
                channel="#auth",
                author="bob@company.com",
                content="Building OAuth support",
                timestamp=now - timedelta(days=4),
                relevance_score=0.85,
                team="auth-team",
            ),
            EvidenceItem(
                source="slack",
                message_id=3,
                channel="#platform",
                author="alice@company.com",
                content="OAuth design complete",
                timestamp=now - timedelta(days=2),
                relevance_score=0.8,
                team="platform-team",
            ),
        ]

        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=5),
            end=now - timedelta(days=1),
            overlap_days=4,
        )

        verification = LLMVerification(
            is_duplicate=True,
            confidence=0.87,
            reasoning="Both teams implementing OAuth2",
            evidence=["Quote 1", "Quote 2"],
            recommendation="Connect teams",
            overlap_ratio=0.85,
        )

        return CoordinationGap(
            id="gap_test123",
            type="duplicate_work",
            title="Duplicate OAuth work",
            topic="OAuth implementation",
            teams_involved=["platform-team", "auth-team"],
            impact_score=0.85,
            impact_tier="HIGH",
            confidence=0.85,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            verification=verification,
            insight="Two teams working on OAuth independently",
            recommendation="Connect teams immediately",
            detected_at=now,
            cluster_id="cluster_abc",
            people_affected=5,
            timespan_days=5,
            messages_analyzed=10,
        )


class TestDuplicateWorkValidator:
    """Tests for DuplicateWorkValidator specialized validator."""

    def test_init_with_defaults(self):
        """Test duplicate work validator initialization."""
        validator = DuplicateWorkValidator()

        assert validator.min_confidence == 0.7
        assert validator.min_teams == 2
        assert validator.min_evidence_items == 3
        assert validator.min_temporal_overlap_days == 3
        assert validator.min_overlap_ratio == 0.5

    def test_validate_valid_duplicate_work(self):
        """Test validation of valid duplicate work gap."""
        validator = DuplicateWorkValidator()

        gap = self._create_valid_duplicate_work_gap()

        is_valid, failures = validator.validate(gap)

        assert is_valid is True
        assert len(failures) == 0

    def test_validate_low_overlap_ratio(self):
        """Test validation failure for low overlap ratio."""
        validator = DuplicateWorkValidator(min_overlap_ratio=0.8)

        gap = self._create_valid_duplicate_work_gap()
        gap.verification.overlap_ratio = 0.4

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("Overlap ratio" in f for f in failures)

    def test_validate_missing_temporal_overlap(self):
        """Test validation failure for missing temporal overlap."""
        validator = DuplicateWorkValidator()

        gap = self._create_valid_duplicate_work_gap()
        gap.temporal_overlap = None

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("Temporal overlap data missing" in f for f in failures)

    def test_validate_insufficient_teams(self):
        """Test validation failure for insufficient teams."""
        validator = DuplicateWorkValidator()

        gap = self._create_valid_duplicate_work_gap()
        gap.teams_involved = ["single-team"]

        is_valid, failures = validator.validate(gap)

        assert is_valid is False
        assert any("requires 2+ teams" in f for f in failures)

    # Helper methods

    def _create_valid_duplicate_work_gap(self) -> CoordinationGap:
        """Create a valid duplicate work gap for testing."""
        now = datetime.utcnow()

        evidence = [
            EvidenceItem(
                source="slack",
                message_id=i,
                channel=f"#team{i % 2}",
                author=f"user{i}@company.com",
                content=f"OAuth message {i}",
                timestamp=now - timedelta(days=5 - i),
                relevance_score=0.9 - (i * 0.1),
                team=f"team-{i % 2}",
            )
            for i in range(5)
        ]

        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=5),
            end=now - timedelta(days=1),
            overlap_days=4,
        )

        verification = LLMVerification(
            is_duplicate=True,
            confidence=0.87,
            reasoning="Both teams implementing OAuth2",
            evidence=["Quote 1", "Quote 2"],
            recommendation="Connect teams",
            overlap_ratio=0.85,
        )

        return CoordinationGap(
            id="gap_dup123",
            type="duplicate_work",
            title="Duplicate OAuth work",
            topic="OAuth implementation",
            teams_involved=["team-0", "team-1"],
            impact_score=0.85,
            impact_tier="HIGH",
            confidence=0.85,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            verification=verification,
            insight="Two teams working on OAuth independently",
            recommendation="Connect teams immediately",
            detected_at=now,
            cluster_id="cluster_abc",
            people_affected=8,
            timespan_days=5,
            messages_analyzed=15,
        )
