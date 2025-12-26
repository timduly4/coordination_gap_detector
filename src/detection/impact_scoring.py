"""
Impact scoring for coordination gaps.

This module implements multi-signal impact scoring to quantify the organizational
cost of coordination failures. Uses signals like team size, time investment,
project criticality, velocity impact, and duplicate effort to calculate impact.
"""

import logging
from typing import Any, Dict, List, Optional

from src.models.schemas import (
    CoordinationGap,
    EvidenceItem,
    ImpactBreakdown,
    LLMVerification,
    TemporalOverlap,
)

logger = logging.getLogger(__name__)


class ImpactScorer:
    """
    Multi-signal impact scoring for coordination gaps.

    Combines multiple signals to estimate the organizational cost:
    - Team size (25%): How many people are affected?
    - Time investment (25%): How much time was wasted?
    - Project criticality (20%): How important is this work?
    - Velocity impact (15%): What's the opportunity cost?
    - Duplicate effort (15%): How much actual duplication?
    """

    def __init__(
        self,
        team_size_weight: float = 0.25,
        time_investment_weight: float = 0.25,
        project_criticality_weight: float = 0.20,
        velocity_impact_weight: float = 0.15,
        duplicate_effort_weight: float = 0.15,
    ):
        """
        Initialize impact scorer with configurable weights.

        Args:
            team_size_weight: Weight for team size score
            time_investment_weight: Weight for time investment score
            project_criticality_weight: Weight for project criticality score
            velocity_impact_weight: Weight for velocity impact score
            duplicate_effort_weight: Weight for duplicate effort score
        """
        # Validate weights sum to 1.0
        total_weight = (
            team_size_weight
            + time_investment_weight
            + project_criticality_weight
            + velocity_impact_weight
            + duplicate_effort_weight
        )

        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        self.team_size_weight = team_size_weight
        self.time_investment_weight = time_investment_weight
        self.project_criticality_weight = project_criticality_weight
        self.velocity_impact_weight = velocity_impact_weight
        self.duplicate_effort_weight = duplicate_effort_weight

        logger.info("Initialized ImpactScorer with weights: team_size=%.2f, time=%.2f, criticality=%.2f, velocity=%.2f, duplicate=%.2f",
                    team_size_weight, time_investment_weight, project_criticality_weight,
                    velocity_impact_weight, duplicate_effort_weight)

    def calculate_impact_score(
        self,
        teams: List[str],
        evidence: List[EvidenceItem],
        temporal_overlap: Optional[TemporalOverlap] = None,
        llm_verification: Optional[LLMVerification] = None,
        participant_count: int = 0,
        commits_found: int = 0,
        project_tags: Optional[List[str]] = None,
        blocking_work_count: int = 0,
    ) -> tuple[float, ImpactBreakdown]:
        """
        Calculate impact score and breakdown.

        Args:
            teams: List of teams involved
            evidence: Evidence items
            temporal_overlap: Temporal overlap analysis
            llm_verification: LLM verification results
            participant_count: Number of unique participants
            commits_found: Number of commits related to this work
            project_tags: Project tags for criticality assessment
            blocking_work_count: Number of items blocked by this work

        Returns:
            Tuple of (impact_score, impact_breakdown)
        """
        logger.debug(f"Calculating impact score for {len(teams)} teams with {len(evidence)} evidence items")

        # Calculate individual component scores
        team_size_score = self._calculate_team_size_score(
            teams, participant_count
        )

        time_investment_score = self._calculate_time_investment_score(
            evidence, temporal_overlap, commits_found
        )

        project_criticality_score = self._calculate_project_criticality_score(
            project_tags or []
        )

        velocity_impact_score = self._calculate_velocity_impact_score(
            blocking_work_count
        )

        duplicate_effort_score = self._calculate_duplicate_effort_score(
            llm_verification
        )

        # Create breakdown
        breakdown = ImpactBreakdown(
            team_size_score=team_size_score,
            time_investment_score=time_investment_score,
            project_criticality_score=project_criticality_score,
            velocity_impact_score=velocity_impact_score,
            duplicate_effort_score=duplicate_effort_score,
        )

        # Calculate weighted impact score
        impact_score = (
            self.team_size_weight * team_size_score
            + self.time_investment_weight * time_investment_score
            + self.project_criticality_weight * project_criticality_score
            + self.velocity_impact_weight * velocity_impact_score
            + self.duplicate_effort_weight * duplicate_effort_score
        )

        # Round to 2 decimal places
        impact_score = round(min(1.0, max(0.0, impact_score)), 2)

        logger.info(f"Calculated impact score: {impact_score} (team={team_size_score:.2f}, time={time_investment_score:.2f}, criticality={project_criticality_score:.2f}, velocity={velocity_impact_score:.2f}, duplicate={duplicate_effort_score:.2f})")

        return impact_score, breakdown

    def _calculate_team_size_score(
        self, teams: List[str], participant_count: int
    ) -> float:
        """
        Calculate team size score based on number of people affected.

        Scoring:
        - 10+ people: 1.0 (critical impact)
        - 5-9 people: 0.7 (high impact)
        - 2-4 people: 0.4 (medium impact)
        - 1 person: 0.2 (low impact)

        Args:
            teams: List of team names
            participant_count: Number of unique participants

        Returns:
            Team size score (0-1)
        """
        # Estimate team size if not provided
        if participant_count > 0:
            team_size = participant_count
        else:
            # Fallback: estimate based on number of teams
            # Assume average team size of 5
            team_size = len(teams) * 5

        # Score based on total people-hours at risk
        if team_size >= 10:
            score = 1.0
        elif team_size >= 5:
            score = 0.7
        elif team_size >= 2:
            score = 0.4
        else:
            score = 0.2

        logger.debug(f"Team size score: {score} (teams={len(teams)}, participants={participant_count}, estimated={team_size})")
        return score

    def _calculate_time_investment_score(
        self,
        evidence: List[EvidenceItem],
        temporal_overlap: Optional[TemporalOverlap],
        commits_found: int,
    ) -> float:
        """
        Calculate time investment score based on estimated hours wasted.

        Heuristic:
        - 1 message ≈ 30 minutes of work
        - 1 commit ≈ 2 hours of work
        - Timespan provides context for effort density

        Scoring:
        - 100+ hours: 1.0 (critical waste)
        - 50-100 hours: 0.7 (high waste)
        - 20-50 hours: 0.5 (medium waste)
        - 10-20 hours: 0.3 (low-medium waste)
        - <10 hours: 0.1 (low waste)

        Args:
            evidence: Evidence items
            temporal_overlap: Temporal overlap analysis
            commits_found: Number of commits

        Returns:
            Time investment score (0-1)
        """
        message_count = len(evidence)

        # Estimate hours from messages and commits
        # Each message represents discussion/planning ~30 min
        message_hours = message_count * 0.5

        # Each commit represents implementation ~2 hours
        commit_hours = commits_found * 2.0

        estimated_hours = message_hours + commit_hours

        # Factor in timespan density (more messages over shorter time = higher intensity)
        if temporal_overlap and temporal_overlap.overlap_days > 0:
            # Boost score if high message density
            density = message_count / temporal_overlap.overlap_days
            if density > 2:  # More than 2 messages per day = high activity
                estimated_hours *= 1.2

        # Normalize to [0, 1] based on thresholds
        if estimated_hours >= 100:
            score = 1.0
        elif estimated_hours >= 50:
            score = 0.7 + (estimated_hours - 50) / 50 * 0.3  # Linear interpolation to 1.0
        elif estimated_hours >= 20:
            score = 0.5 + (estimated_hours - 20) / 30 * 0.2  # Linear to 0.7
        elif estimated_hours >= 10:
            score = 0.3 + (estimated_hours - 10) / 10 * 0.2  # Linear to 0.5
        else:
            score = min(0.3, estimated_hours / 10 * 0.3)  # Linear from 0 to 0.3

        logger.debug(f"Time investment score: {score:.2f} (messages={message_count}, commits={commits_found}, estimated_hours={estimated_hours:.1f})")
        return round(score, 2)

    def _calculate_project_criticality_score(
        self, project_tags: List[str]
    ) -> float:
        """
        Calculate project criticality score based on project attributes.

        Scoring based on tags:
        - roadmap_item: 0.9 (tied to strategic goals)
        - okr_related: 0.85 (tied to objectives)
        - security: 0.8 (security-critical)
        - production: 0.75 (affects production)
        - customer_facing: 0.7 (customer impact)
        - infrastructure: 0.6 (foundational)
        - internal_tool: 0.3 (internal only)

        Args:
            project_tags: List of project tags

        Returns:
            Project criticality score (0-1)
        """
        # Criticality weights for different project types
        criticality_weights = {
            "roadmap_item": 0.9,
            "roadmap": 0.9,
            "okr_related": 0.85,
            "okr": 0.85,
            "security": 0.8,
            "production": 0.75,
            "prod": 0.75,
            "customer_facing": 0.7,
            "customer": 0.7,
            "infrastructure": 0.6,
            "infra": 0.6,
            "internal_tool": 0.3,
            "internal": 0.3,
        }

        # Find highest criticality tag
        matched_tags = []
        matched_scores = []

        for tag in project_tags:
            tag_lower = tag.lower()
            if tag_lower in criticality_weights:
                matched_tags.append(tag_lower)
                matched_scores.append(criticality_weights[tag_lower])

        # Use highest matched score, or default if no matches
        if matched_scores:
            max_criticality = max(matched_scores)
        else:
            max_criticality = 0.5  # Default moderate criticality when no tags match

        logger.debug(f"Project criticality score: {max_criticality} (tags={project_tags}, matched={matched_tags})")
        return max_criticality

    def _calculate_velocity_impact_score(self, blocking_work_count: int) -> float:
        """
        Calculate velocity impact score based on blocked work items.

        Measures opportunity cost - what else could teams be working on?

        Scoring:
        - 5+ blocked items: 1.0 (critical blocking)
        - 3-4 blocked items: 0.7 (high blocking)
        - 1-2 blocked items: 0.4 (medium blocking)
        - 0 blocked items: 0.2 (base opportunity cost)

        Args:
            blocking_work_count: Number of work items blocked

        Returns:
            Velocity impact score (0-1)
        """
        if blocking_work_count >= 5:
            score = 1.0
        elif blocking_work_count >= 3:
            score = 0.7 + (blocking_work_count - 3) / 2 * 0.3  # Linear to 1.0
        elif blocking_work_count >= 1:
            score = 0.4 + (blocking_work_count - 1) / 2 * 0.3  # Linear to 0.7
        else:
            # Even with no blocked work, there's opportunity cost
            score = 0.2

        logger.debug(f"Velocity impact score: {score} (blocked_items={blocking_work_count})")
        return score

    def _calculate_duplicate_effort_score(
        self, llm_verification: Optional[LLMVerification]
    ) -> float:
        """
        Calculate duplicate effort score from LLM assessment.

        Uses LLM's overlap_ratio which estimates percentage of duplicated work.

        Args:
            llm_verification: LLM verification results

        Returns:
            Duplicate effort score (0-1)
        """
        if llm_verification:
            score = llm_verification.overlap_ratio
        else:
            # Conservative default if no LLM verification
            score = 0.5

        logger.debug(f"Duplicate effort score: {score} (from LLM overlap_ratio)")
        return score

    def determine_impact_tier(self, impact_score: float) -> str:
        """
        Determine impact tier from numerical score.

        Tiers:
        - CRITICAL (0.8-1.0): Immediate intervention needed
        - HIGH (0.6-0.8): Address within week
        - MEDIUM (0.4-0.6): Monitor and advise
        - LOW (0.0-0.4): FYI only

        Args:
            impact_score: Impact score (0-1)

        Returns:
            Impact tier string
        """
        if impact_score >= 0.8:
            return "CRITICAL"
        elif impact_score >= 0.6:
            return "HIGH"
        elif impact_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
