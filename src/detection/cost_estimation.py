"""
Cost estimation for coordination gaps.

This module estimates the organizational cost (engineering hours and dollar value)
of coordination failures. This is the cost of wasted duplicate effort, NOT the
cost of running the detection system or Claude API calls.
"""

import logging
from typing import List, Optional

from src.models.schemas import CostEstimate, EvidenceItem, TemporalOverlap

logger = logging.getLogger(__name__)


class CostEstimator:
    """
    Estimate organizational cost of coordination gaps.

    IMPORTANT: This estimates the cost of duplicate work / coordination failures,
    NOT the cost of running the detection system or API calls.

    Example: If two teams each spend 40 hours building the same OAuth feature,
    the cost estimate would be ~80 hours and $8,000+ (not the ~$0.05 to run
    Claude API to detect it).
    """

    def __init__(
        self,
        avg_hourly_rate: float = 100.0,
        message_time_hours: float = 0.5,
        commit_time_hours: float = 2.0,
        coordination_overhead: float = 0.15,
    ):
        """
        Initialize cost estimator with configurable parameters.

        Args:
            avg_hourly_rate: Average loaded cost per engineer ($/hour)
            message_time_hours: Hours per message (discussion/planning)
            commit_time_hours: Hours per commit (implementation)
            coordination_overhead: Multiplier for coordination overhead (e.g., 0.15 = 15% overhead)
        """
        self.avg_hourly_rate = avg_hourly_rate
        self.message_time_hours = message_time_hours
        self.commit_time_hours = commit_time_hours
        self.coordination_overhead = coordination_overhead

        logger.info(
            f"Initialized CostEstimator: rate=${avg_hourly_rate}/hr, "
            f"msg={message_time_hours}hrs, commit={commit_time_hours}hrs, "
            f"overhead={coordination_overhead * 100}%"
        )

    def estimate_cost(
        self,
        teams: List[str],
        evidence: List[EvidenceItem],
        temporal_overlap: Optional[TemporalOverlap] = None,
        commits_found: int = 0,
        participant_count: int = 0,
    ) -> CostEstimate:
        """
        Estimate organizational cost of a coordination gap.

        Args:
            teams: Teams involved
            evidence: Evidence items
            temporal_overlap: Temporal overlap analysis
            commits_found: Number of commits found
            participant_count: Number of unique participants

        Returns:
            CostEstimate with hours, dollar value, and explanation
        """
        logger.debug(
            f"Estimating cost for {len(teams)} teams, {len(evidence)} evidence items, "
            f"{commits_found} commits"
        )

        # Calculate base engineering hours
        base_hours = self._calculate_base_hours(evidence, commits_found)

        # Apply team multiplier (duplicate effort across teams)
        team_multiplier = self._calculate_team_multiplier(teams, participant_count)
        duplicated_hours = base_hours * team_multiplier

        # Add coordination overhead
        total_hours = duplicated_hours * (1 + self.coordination_overhead)

        # Calculate dollar value
        dollar_value = total_hours * self.avg_hourly_rate

        # Generate explanation
        explanation = self._generate_explanation(
            teams=teams,
            base_hours=base_hours,
            team_multiplier=team_multiplier,
            duplicated_hours=duplicated_hours,
            total_hours=total_hours,
            participant_count=participant_count,
            message_count=len(evidence),
            commits_found=commits_found,
        )

        cost_estimate = CostEstimate(
            engineering_hours=round(total_hours, 1),
            dollar_value=round(dollar_value, 2),
            explanation=explanation,
        )

        logger.info(
            f"Cost estimate: {cost_estimate.engineering_hours} hours, "
            f"${cost_estimate.dollar_value:,.2f}"
        )

        return cost_estimate

    def _calculate_base_hours(
        self, evidence: List[EvidenceItem], commits_found: int
    ) -> float:
        """
        Calculate base engineering hours from evidence and commits.

        Args:
            evidence: Evidence items (messages, discussions)
            commits_found: Number of commits

        Returns:
            Base engineering hours (for one team)
        """
        # Messages represent planning, discussion, meetings
        message_hours = len(evidence) * self.message_time_hours

        # Commits represent actual implementation work
        commit_hours = commits_found * self.commit_time_hours

        base_hours = message_hours + commit_hours

        # Minimum floor - even small duplications waste some time
        base_hours = max(base_hours, 5.0)

        logger.debug(
            f"Base hours: {base_hours:.1f} "
            f"(messages={len(evidence)}*{self.message_time_hours}hrs, "
            f"commits={commits_found}*{self.commit_time_hours}hrs)"
        )

        return base_hours

    def _calculate_team_multiplier(
        self, teams: List[str], participant_count: int
    ) -> float:
        """
        Calculate multiplier for duplicate effort across teams.

        If 2 teams each spend 40 hours, that's 80 hours total wasted.
        But there may be some useful work (learning, different approaches),
        so we apply a partial duplication factor.

        Args:
            teams: Teams involved
            participant_count: Number of participants

        Returns:
            Team multiplier (typically between 1.0 and team count)
        """
        num_teams = len(teams)

        if num_teams <= 1:
            # No duplication if only one team
            return 1.0

        # Base multiplier: if 2 teams, assume 80% duplication (1.8x)
        # If 3 teams, assume 70% per additional team (2.4x)
        # This accounts for some variation and learning value

        if num_teams == 2:
            multiplier = 1.8
        elif num_teams == 3:
            multiplier = 2.4
        else:
            # For 4+ teams, diminishing returns (people start noticing)
            multiplier = 2.4 + (num_teams - 3) * 0.4

        # Cap at reasonable limit
        multiplier = min(multiplier, num_teams * 0.85)

        logger.debug(f"Team multiplier: {multiplier:.1f} (teams={num_teams})")

        return multiplier

    def _generate_explanation(
        self,
        teams: List[str],
        base_hours: float,
        team_multiplier: float,
        duplicated_hours: float,
        total_hours: float,
        participant_count: int,
        message_count: int,
        commits_found: int,
    ) -> str:
        """
        Generate human-readable explanation of cost calculation.

        Args:
            teams: Teams involved
            base_hours: Base hours per team
            team_multiplier: Duplication multiplier
            duplicated_hours: Hours after duplication
            total_hours: Total with overhead
            participant_count: Number of participants
            message_count: Number of messages
            commits_found: Number of commits

        Returns:
            Explanation string
        """
        team_names = ", ".join(teams)
        num_teams = len(teams)

        if num_teams > 1:
            explanation = (
                f"{num_teams} teams ({team_names}) working independently. "
                f"Estimated ~{base_hours:.0f} hours per team "
                f"(from {message_count} messages + {commits_found} commits). "
                f"With {int((team_multiplier - 1) * 100)}% duplication across teams: "
                f"~{duplicated_hours:.0f} duplicated hours. "
                f"Including {int(self.coordination_overhead * 100)}% coordination overhead: "
                f"~{total_hours:.0f} total hours wasted."
            )
        else:
            explanation = (
                f"Single team effort: {team_names}. "
                f"Estimated ~{base_hours:.0f} hours "
                f"(from {message_count} messages + {commits_found} commits). "
                f"Including coordination overhead: ~{total_hours:.0f} total hours."
            )

        if participant_count > 0:
            explanation += f" {participant_count} people involved."

        return explanation

    def estimate_cost_simple(
        self,
        estimated_hours: float,
        team_count: int = 1,
    ) -> CostEstimate:
        """
        Simplified cost estimation from pre-calculated hours.

        Args:
            estimated_hours: Pre-calculated engineering hours
            team_count: Number of teams involved

        Returns:
            CostEstimate
        """
        # Apply team multiplier if multiple teams
        if team_count > 1:
            multiplier = self._calculate_team_multiplier(
                teams=[f"team_{i}" for i in range(team_count)],
                participant_count=0
            )
            total_hours = estimated_hours * multiplier
        else:
            total_hours = estimated_hours

        # Add overhead
        total_hours = total_hours * (1 + self.coordination_overhead)

        # Calculate cost
        dollar_value = total_hours * self.avg_hourly_rate

        explanation = (
            f"{team_count} team{'s' if team_count > 1 else ''} × "
            f"~{estimated_hours:.0f} hours each = "
            f"~{total_hours:.0f} total hours wasted"
        )

        return CostEstimate(
            engineering_hours=round(total_hours, 1),
            dollar_value=round(dollar_value, 2),
            explanation=explanation,
        )
