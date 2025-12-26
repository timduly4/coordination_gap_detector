"""
Tests for impact scoring algorithm.
"""

import pytest
from datetime import datetime, timedelta

from src.detection.impact_scoring import ImpactScorer
from src.detection.cost_estimation import CostEstimator
from src.models.schemas import (
    EvidenceItem,
    ImpactBreakdown,
    LLMVerification,
    TemporalOverlap,
    CostEstimate,
)


class TestImpactScorer:
    """Tests for ImpactScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create impact scorer instance."""
        return ImpactScorer()

    @pytest.fixture
    def custom_scorer(self):
        """Create scorer with custom weights."""
        return ImpactScorer(
            team_size_weight=0.3,
            time_investment_weight=0.3,
            project_criticality_weight=0.2,
            velocity_impact_weight=0.1,
            duplicate_effort_weight=0.1,
        )

    def test_init_default_weights(self, scorer):
        """Test initialization with default weights."""
        assert scorer.team_size_weight == 0.25
        assert scorer.time_investment_weight == 0.25
        assert scorer.project_criticality_weight == 0.20
        assert scorer.velocity_impact_weight == 0.15
        assert scorer.duplicate_effort_weight == 0.15

        # Weights should sum to 1.0
        total = (
            scorer.team_size_weight
            + scorer.time_investment_weight
            + scorer.project_criticality_weight
            + scorer.velocity_impact_weight
            + scorer.duplicate_effort_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_init_custom_weights(self, custom_scorer):
        """Test initialization with custom weights."""
        assert custom_scorer.team_size_weight == 0.3
        assert custom_scorer.time_investment_weight == 0.3

    def test_init_invalid_weights(self):
        """Test that invalid weights (not summing to 1.0) raise error."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ImpactScorer(
                team_size_weight=0.5,
                time_investment_weight=0.5,
                project_criticality_weight=0.5,
                velocity_impact_weight=0.1,
                duplicate_effort_weight=0.1,
            )

    def test_calculate_team_size_score_large_team(self, scorer):
        """Test team size score for large teams (10+ people)."""
        score = scorer._calculate_team_size_score(
            teams=["team1", "team2"], participant_count=12
        )
        assert score == 1.0

    def test_calculate_team_size_score_medium_team(self, scorer):
        """Test team size score for medium teams (5-9 people)."""
        score = scorer._calculate_team_size_score(
            teams=["team1", "team2"], participant_count=7
        )
        assert score == 0.7

    def test_calculate_team_size_score_small_team(self, scorer):
        """Test team size score for small teams (2-4 people)."""
        score = scorer._calculate_team_size_score(
            teams=["team1"], participant_count=3
        )
        assert score == 0.4

    def test_calculate_team_size_score_single_person(self, scorer):
        """Test team size score for single person."""
        score = scorer._calculate_team_size_score(
            teams=["team1"], participant_count=1
        )
        assert score == 0.2

    def test_calculate_team_size_score_fallback(self, scorer):
        """Test team size score with no participant count (uses team count estimate)."""
        score = scorer._calculate_team_size_score(
            teams=["team1", "team2"], participant_count=0
        )
        # 2 teams * 5 avg = 10 people => score = 1.0
        assert score == 1.0

    def test_calculate_time_investment_score_critical(self, scorer):
        """Test time investment score for 100+ hours."""
        now = datetime.utcnow()
        evidence = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now,
                relevance_score=0.9,
            )
            for i in range(20)
        ]
        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=10),
            end=now,
            overlap_days=10,
        )

        score = scorer._calculate_time_investment_score(
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            commits_found=50,  # 50 commits * 2 hours = 100 hours
        )

        # Should be 1.0 for 100+ hours
        assert score >= 0.9

    def test_calculate_time_investment_score_medium(self, scorer):
        """Test time investment score for medium hours (20-50)."""
        now = datetime.utcnow()
        evidence = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now,
                relevance_score=0.9,
            )
            for i in range(10)
        ]

        score = scorer._calculate_time_investment_score(
            evidence=evidence,
            temporal_overlap=None,
            commits_found=10,  # 10 commits * 2 hours + 10 messages * 0.5 hours = 25 hours
        )

        # Should be between 0.3 and 0.7
        assert 0.3 <= score <= 0.7

    def test_calculate_time_investment_score_low(self, scorer):
        """Test time investment score for low hours (<10)."""
        now = datetime.utcnow()
        evidence = [
            EvidenceItem(
                source="slack",
                content="Test message",
                timestamp=now,
                relevance_score=0.9,
            )
        ]

        score = scorer._calculate_time_investment_score(
            evidence=evidence,
            temporal_overlap=None,
            commits_found=2,  # 2 * 2 + 1 * 0.5 = 4.5 hours
        )

        # Should be less than 0.3
        assert score < 0.3

    def test_calculate_time_investment_density_boost(self, scorer):
        """Test that high message density boosts time investment score."""
        now = datetime.utcnow()

        # High density: 20 messages over 5 days = 4 messages/day
        evidence_high_density = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now,
                relevance_score=0.9,
            )
            for i in range(20)
        ]
        temporal_high = TemporalOverlap(
            start=now - timedelta(days=5),
            end=now,
            overlap_days=5,
        )

        # Low density: 20 messages over 20 days = 1 message/day
        temporal_low = TemporalOverlap(
            start=now - timedelta(days=20),
            end=now,
            overlap_days=20,
        )

        score_high = scorer._calculate_time_investment_score(
            evidence=evidence_high_density,
            temporal_overlap=temporal_high,
            commits_found=5,
        )

        score_low = scorer._calculate_time_investment_score(
            evidence=evidence_high_density,
            temporal_overlap=temporal_low,
            commits_found=5,
        )

        # High density should score higher
        assert score_high > score_low

    def test_calculate_project_criticality_score_roadmap(self, scorer):
        """Test criticality score for roadmap items."""
        score = scorer._calculate_project_criticality_score(
            project_tags=["roadmap_item", "customer_facing"]
        )
        # Should pick highest: roadmap_item = 0.9
        assert score == 0.9

    def test_calculate_project_criticality_score_security(self, scorer):
        """Test criticality score for security projects."""
        score = scorer._calculate_project_criticality_score(
            project_tags=["security", "production"]
        )
        # Should pick highest: security = 0.8
        assert score == 0.8

    def test_calculate_project_criticality_score_internal(self, scorer):
        """Test criticality score for internal tools."""
        score = scorer._calculate_project_criticality_score(
            project_tags=["internal_tool"]
        )
        assert score == 0.3

    def test_calculate_project_criticality_score_default(self, scorer):
        """Test default criticality score with no tags."""
        score = scorer._calculate_project_criticality_score(
            project_tags=[]
        )
        # Default moderate criticality
        assert score == 0.5

    def test_calculate_project_criticality_score_case_insensitive(self, scorer):
        """Test that project tags are case insensitive."""
        score_upper = scorer._calculate_project_criticality_score(
            project_tags=["ROADMAP"]
        )
        score_lower = scorer._calculate_project_criticality_score(
            project_tags=["roadmap"]
        )
        assert score_upper == score_lower == 0.9

    def test_calculate_velocity_impact_score_critical(self, scorer):
        """Test velocity impact for 5+ blocked items."""
        score = scorer._calculate_velocity_impact_score(blocking_work_count=7)
        assert score == 1.0

    def test_calculate_velocity_impact_score_high(self, scorer):
        """Test velocity impact for 3-4 blocked items."""
        score = scorer._calculate_velocity_impact_score(blocking_work_count=3)
        assert score == 0.7

    def test_calculate_velocity_impact_score_medium(self, scorer):
        """Test velocity impact for 1-2 blocked items."""
        score = scorer._calculate_velocity_impact_score(blocking_work_count=1)
        assert score == 0.4

    def test_calculate_velocity_impact_score_none(self, scorer):
        """Test baseline velocity impact with no blocked items."""
        score = scorer._calculate_velocity_impact_score(blocking_work_count=0)
        # Even with no blocked work, there's opportunity cost
        assert score == 0.2

    def test_calculate_duplicate_effort_score_with_llm(self, scorer):
        """Test duplicate effort score from LLM verification."""
        llm_verification = LLMVerification(
            is_duplicate=True,
            confidence=0.85,
            reasoning="Both teams building same feature",
            evidence=["Quote 1", "Quote 2"],
            recommendation="Consolidate efforts",
            overlap_ratio=0.9,
        )

        score = scorer._calculate_duplicate_effort_score(llm_verification)
        assert score == 0.9

    def test_calculate_duplicate_effort_score_without_llm(self, scorer):
        """Test duplicate effort score without LLM verification."""
        score = scorer._calculate_duplicate_effort_score(None)
        # Conservative default
        assert score == 0.5

    def test_calculate_impact_score_full(self, scorer):
        """Test full impact score calculation with all signals."""
        now = datetime.utcnow()

        teams = ["platform-team", "auth-team"]
        evidence = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now - timedelta(hours=i),
                relevance_score=0.9,
            )
            for i in range(15)
        ]
        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=10),
            end=now,
            overlap_days=10,
        )
        llm_verification = LLMVerification(
            is_duplicate=True,
            confidence=0.85,
            reasoning="Duplicate work",
            evidence=["Quote"],
            recommendation="Connect teams",
            overlap_ratio=0.85,
        )

        impact_score, breakdown = scorer.calculate_impact_score(
            teams=teams,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            llm_verification=llm_verification,
            participant_count=8,
            commits_found=12,
            project_tags=["roadmap"],
            blocking_work_count=3,
        )

        # Verify impact score is in valid range
        assert 0.0 <= impact_score <= 1.0

        # Verify breakdown components
        assert isinstance(breakdown, ImpactBreakdown)
        assert 0.0 <= breakdown.team_size_score <= 1.0
        assert 0.0 <= breakdown.time_investment_score <= 1.0
        assert 0.0 <= breakdown.project_criticality_score <= 1.0
        assert 0.0 <= breakdown.velocity_impact_score <= 1.0
        assert 0.0 <= breakdown.duplicate_effort_score <= 1.0

        # For this scenario, should be high impact
        assert impact_score >= 0.6

    def test_calculate_impact_score_minimal(self, scorer):
        """Test impact score with minimal inputs."""
        now = datetime.utcnow()

        teams = ["team1"]
        evidence = [
            EvidenceItem(
                source="slack",
                content="Single message",
                timestamp=now,
                relevance_score=0.5,
            )
        ]
        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=1),
            end=now,
            overlap_days=1,
        )

        impact_score, breakdown = scorer.calculate_impact_score(
            teams=teams,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            llm_verification=None,
            participant_count=1,
            commits_found=0,
            project_tags=[],
            blocking_work_count=0,
        )

        # Should be low impact
        assert impact_score < 0.5
        assert breakdown.team_size_score == 0.2  # Single person
        assert breakdown.velocity_impact_score == 0.2  # Base opportunity cost
        assert breakdown.duplicate_effort_score == 0.5  # Default without LLM

    def test_determine_impact_tier_critical(self, scorer):
        """Test CRITICAL tier (0.8-1.0)."""
        assert scorer.determine_impact_tier(0.85) == "CRITICAL"
        assert scorer.determine_impact_tier(0.8) == "CRITICAL"
        assert scorer.determine_impact_tier(1.0) == "CRITICAL"

    def test_determine_impact_tier_high(self, scorer):
        """Test HIGH tier (0.6-0.8)."""
        assert scorer.determine_impact_tier(0.7) == "HIGH"
        assert scorer.determine_impact_tier(0.6) == "HIGH"
        assert scorer.determine_impact_tier(0.79) == "HIGH"

    def test_determine_impact_tier_medium(self, scorer):
        """Test MEDIUM tier (0.4-0.6)."""
        assert scorer.determine_impact_tier(0.5) == "MEDIUM"
        assert scorer.determine_impact_tier(0.4) == "MEDIUM"
        assert scorer.determine_impact_tier(0.59) == "MEDIUM"

    def test_determine_impact_tier_low(self, scorer):
        """Test LOW tier (0.0-0.4)."""
        assert scorer.determine_impact_tier(0.3) == "LOW"
        assert scorer.determine_impact_tier(0.1) == "LOW"
        assert scorer.determine_impact_tier(0.0) == "LOW"


class TestCostEstimator:
    """Tests for CostEstimator class."""

    @pytest.fixture
    def estimator(self):
        """Create cost estimator instance."""
        return CostEstimator()

    @pytest.fixture
    def custom_estimator(self):
        """Create estimator with custom parameters."""
        return CostEstimator(
            avg_hourly_rate=150.0,
            message_time_hours=1.0,
            commit_time_hours=3.0,
            coordination_overhead=0.20,
        )

    def test_init_default_params(self, estimator):
        """Test initialization with default parameters."""
        assert estimator.avg_hourly_rate == 100.0
        assert estimator.message_time_hours == 0.5
        assert estimator.commit_time_hours == 2.0
        assert estimator.coordination_overhead == 0.15

    def test_init_custom_params(self, custom_estimator):
        """Test initialization with custom parameters."""
        assert custom_estimator.avg_hourly_rate == 150.0
        assert custom_estimator.message_time_hours == 1.0
        assert custom_estimator.commit_time_hours == 3.0
        assert custom_estimator.coordination_overhead == 0.20

    def test_calculate_base_hours(self, estimator):
        """Test base hours calculation."""
        now = datetime.utcnow()
        evidence = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now,
                relevance_score=0.9,
            )
            for i in range(10)
        ]

        base_hours = estimator._calculate_base_hours(
            evidence=evidence,
            commits_found=5,
        )

        # 10 messages * 0.5 hours + 5 commits * 2 hours = 5 + 10 = 15 hours
        assert base_hours == 15.0

    def test_calculate_base_hours_minimum_floor(self, estimator):
        """Test that base hours has minimum floor."""
        evidence = []  # No messages
        base_hours = estimator._calculate_base_hours(
            evidence=evidence,
            commits_found=0,
        )

        # Should have minimum of 5 hours
        assert base_hours == 5.0

    def test_calculate_team_multiplier_single_team(self, estimator):
        """Test team multiplier for single team."""
        multiplier = estimator._calculate_team_multiplier(
            teams=["team1"],
            participant_count=5,
        )
        # No duplication with single team
        assert multiplier == 1.0

    def test_calculate_team_multiplier_two_teams(self, estimator):
        """Test team multiplier for two teams."""
        multiplier = estimator._calculate_team_multiplier(
            teams=["team1", "team2"],
            participant_count=10,
        )
        # Would be 1.8 but capped at teams * 0.85 = 2 * 0.85 = 1.7
        assert multiplier == 1.7

    def test_calculate_team_multiplier_three_teams(self, estimator):
        """Test team multiplier for three teams."""
        multiplier = estimator._calculate_team_multiplier(
            teams=["team1", "team2", "team3"],
            participant_count=15,
        )
        # Should be 2.4 for 3 teams
        assert multiplier == 2.4

    def test_calculate_team_multiplier_many_teams(self, estimator):
        """Test team multiplier caps reasonably."""
        teams = [f"team{i}" for i in range(10)]
        multiplier = estimator._calculate_team_multiplier(
            teams=teams,
            participant_count=50,
        )
        # Should be capped at teams * 0.85
        assert multiplier <= 10 * 0.85

    def test_estimate_cost_single_team(self, estimator):
        """Test cost estimation for single team."""
        now = datetime.utcnow()
        evidence = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now,
                relevance_score=0.9,
            )
            for i in range(10)
        ]
        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=5),
            end=now,
            overlap_days=5,
        )

        cost = estimator.estimate_cost(
            teams=["team1"],
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            commits_found=5,
            participant_count=3,
        )

        assert isinstance(cost, CostEstimate)
        assert cost.engineering_hours > 0
        assert cost.dollar_value > 0
        assert len(cost.explanation) > 0

        # Verify dollar value approximately = hours * rate (allow for rounding)
        assert abs(cost.dollar_value - cost.engineering_hours * 100) < 10.0

    def test_estimate_cost_multiple_teams(self, estimator):
        """Test cost estimation for multiple teams (duplication)."""
        now = datetime.utcnow()
        evidence = [
            EvidenceItem(
                source="slack",
                content=f"Message {i}",
                timestamp=now,
                relevance_score=0.9,
            )
            for i in range(20)
        ]
        temporal_overlap = TemporalOverlap(
            start=now - timedelta(days=10),
            end=now,
            overlap_days=10,
        )

        cost = estimator.estimate_cost(
            teams=["team1", "team2"],
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            commits_found=10,
            participant_count=8,
        )

        # Should have significant cost due to duplication
        assert cost.engineering_hours >= 30  # Base * multiplier * overhead
        assert cost.dollar_value >= 3000

    def test_estimate_cost_simple(self, estimator):
        """Test simplified cost estimation."""
        cost = estimator.estimate_cost_simple(
            estimated_hours=40,
            team_count=2,
        )

        assert isinstance(cost, CostEstimate)

        # 2 teams * 40 hours * 1.8 multiplier * 1.15 overhead
        expected_hours = 40 * 1.8 * 1.15
        assert abs(cost.engineering_hours - expected_hours) < 5.0

        # Verify explanation
        assert "2 teams" in cost.explanation

    def test_estimate_cost_simple_single_team(self, estimator):
        """Test simplified cost estimation with single team."""
        cost = estimator.estimate_cost_simple(
            estimated_hours=20,
            team_count=1,
        )

        # Single team: 20 hours * 1.15 overhead = 23 hours
        assert abs(cost.engineering_hours - 23.0) < 1.0
        assert "1 team" in cost.explanation

    def test_generate_explanation(self, estimator):
        """Test cost explanation generation."""
        explanation = estimator._generate_explanation(
            teams=["platform", "auth"],
            base_hours=40,
            team_multiplier=1.8,
            duplicated_hours=72,
            total_hours=82.8,
            participant_count=8,
            message_count=20,
            commits_found=10,
        )

        # Should mention key details
        assert "2 teams" in explanation
        assert "platform" in explanation
        assert "auth" in explanation
        assert "20 messages" in explanation
        assert "10 commits" in explanation
        assert "8 people" in explanation

    def test_cost_estimate_note(self, estimator):
        """Test that cost estimate includes clarification note."""
        now = datetime.utcnow()
        cost = estimator.estimate_cost(
            teams=["team1"],
            evidence=[
                EvidenceItem(
                    source="slack",
                    content="Test",
                    timestamp=now,
                    relevance_score=0.9,
                )
            ],
            temporal_overlap=TemporalOverlap(
                start=now - timedelta(days=1),
                end=now,
                overlap_days=1,
            ),
            commits_found=0,
            participant_count=1,
        )

        # Note should clarify this is organizational cost, not API cost
        assert "organizational waste cost" in cost.note.lower()
        assert "claude api" in cost.note.lower()
