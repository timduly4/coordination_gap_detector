"""
Gap validation logic for coordination gap detection.

This module implements validation rules to ensure detected gaps meet
quality criteria and filter out false positives.
"""

import logging
from typing import Any, Dict, List, Optional

from src.models.schemas import CoordinationGap, EvidenceItem, LLMVerification

logger = logging.getLogger(__name__)


class GapValidator:
    """
    Validator for coordination gaps.

    Applies rules to determine if a detected gap is valid and should be
    reported to users. Helps filter false positives and low-quality detections.
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        min_teams: int = 2,
        min_evidence_items: int = 2,
        min_temporal_overlap_days: int = 1,
        require_llm_verification: bool = True,
    ):
        """
        Initialize gap validator.

        Args:
            min_confidence: Minimum confidence score (0-1)
            min_teams: Minimum number of teams involved
            min_evidence_items: Minimum evidence items required
            min_temporal_overlap_days: Minimum days of temporal overlap
            require_llm_verification: Whether LLM verification is required
        """
        self.min_confidence = min_confidence
        self.min_teams = min_teams
        self.min_evidence_items = min_evidence_items
        self.min_temporal_overlap_days = min_temporal_overlap_days
        self.require_llm_verification = require_llm_verification

        logger.info(
            f"GapValidator initialized with min_confidence={min_confidence}, "
            f"min_teams={min_teams}, min_evidence={min_evidence_items}"
        )

    def validate(self, gap: CoordinationGap) -> tuple[bool, List[str]]:
        """
        Validate a coordination gap.

        Args:
            gap: CoordinationGap to validate

        Returns:
            Tuple of (is_valid, list of validation failures)
        """
        failures = []

        # Check confidence threshold
        if gap.confidence < self.min_confidence:
            failures.append(
                f"Confidence {gap.confidence:.2f} below threshold {self.min_confidence}"
            )

        # Check minimum teams
        if len(gap.teams_involved) < self.min_teams:
            failures.append(
                f"Only {len(gap.teams_involved)} teams involved, need {self.min_teams}"
            )

        # Check minimum evidence
        if len(gap.evidence) < self.min_evidence_items:
            failures.append(
                f"Only {len(gap.evidence)} evidence items, need {self.min_evidence_items}"
            )

        # Check temporal overlap
        if gap.temporal_overlap:
            if gap.temporal_overlap.overlap_days < self.min_temporal_overlap_days:
                failures.append(
                    f"Temporal overlap {gap.temporal_overlap.overlap_days} days "
                    f"below threshold {self.min_temporal_overlap_days}"
                )

        # Check LLM verification if required
        if self.require_llm_verification:
            if not gap.verification:
                failures.append("LLM verification missing")
            elif not gap.verification.is_duplicate:
                failures.append("LLM verification failed: not duplicate work")

        # Check for exclusion patterns
        exclusion_failures = self._check_exclusion_rules(gap)
        failures.extend(exclusion_failures)

        is_valid = len(failures) == 0

        if not is_valid:
            logger.debug(f"Gap {gap.id} validation failed: {', '.join(failures)}")
        else:
            logger.debug(f"Gap {gap.id} validation passed")

        return is_valid, failures

    def _check_exclusion_rules(self, gap: CoordinationGap) -> List[str]:
        """
        Check exclusion rules that filter false positives.

        Args:
            gap: Gap to check

        Returns:
            List of exclusion failures (empty if no exclusions apply)
        """
        failures = []

        # Rule 1: Check for explicit collaboration indicators
        if self._has_collaboration_indicators(gap):
            failures.append("Collaboration indicators present (not duplicate work)")

        # Rule 2: Check for same team (not really duplicate work)
        if self._is_same_team_work(gap):
            failures.append("All evidence from same team (not cross-team duplication)")

        # Rule 3: Check for intentional redundancy patterns
        if self._is_intentional_redundancy(gap):
            failures.append("Intentional redundancy pattern detected")

        return failures

    def _has_collaboration_indicators(self, gap: CoordinationGap) -> bool:
        """
        Check if evidence contains collaboration indicators.

        Indicators include:
        - Cross-references between teams
        - Explicit mentions of working together
        - Joint decision-making

        Args:
            gap: Gap to check

        Returns:
            True if collaboration indicators found
        """
        collaboration_keywords = [
            "working with",
            "collaborating with",
            "partnering with",
            "coordinating with",
            "joint effort",
            "together with",
            "helping",
            "supporting",
            "aligned with",
            "sync'd with",
            "cross-team",
        ]

        # Check evidence content for collaboration keywords
        for evidence in gap.evidence:
            content_lower = evidence.content.lower()
            for keyword in collaboration_keywords:
                if keyword in content_lower:
                    logger.debug(
                        f"Found collaboration keyword '{keyword}' in evidence"
                    )
                    return True

        # Check for cross-team mentions in evidence
        if self._has_cross_team_mentions(gap):
            logger.debug("Found cross-team mentions indicating collaboration")
            return True

        return False

    def _has_cross_team_mentions(self, gap: CoordinationGap) -> bool:
        """
        Check if teams are mentioning each other.

        Args:
            gap: Gap to check

        Returns:
            True if cross-team mentions detected
        """
        if len(gap.teams_involved) < 2:
            return False

        # Check if evidence from one team mentions another team
        for evidence in gap.evidence:
            if not evidence.team:
                continue

            content_lower = evidence.content.lower()
            for team in gap.teams_involved:
                if team != evidence.team and team.lower() in content_lower:
                    return True

            # Also check for @team mentions
            if "@" in content_lower:
                for team in gap.teams_involved:
                    if f"@{team.lower()}" in content_lower:
                        return True

        return False

    def _is_same_team_work(self, gap: CoordinationGap) -> bool:
        """
        Check if all evidence is from the same team.

        Args:
            gap: Gap to check

        Returns:
            True if all evidence from same team
        """
        teams_with_evidence = set()

        for evidence in gap.evidence:
            if evidence.team:
                teams_with_evidence.add(evidence.team)

        # If all evidence is from one team, not duplicate work
        return len(teams_with_evidence) <= 1

    def _is_intentional_redundancy(self, gap: CoordinationGap) -> bool:
        """
        Check for intentional redundancy patterns.

        Examples:
        - A/B testing
        - Backup systems
        - Proof of concept vs production

        Args:
            gap: Gap to check

        Returns:
            True if intentional redundancy detected
        """
        redundancy_keywords = [
            "a/b test",
            "experiment",
            "backup",
            "redundancy",
            "failover",
            "proof of concept",
            "poc",
            "prototype",
            "spike",
            "comparison",
            "alternative approach",
        ]

        # Check evidence and insight for redundancy keywords
        text_to_check = gap.insight.lower()
        if gap.verification:
            text_to_check += " " + gap.verification.reasoning.lower()

        for keyword in redundancy_keywords:
            if keyword in text_to_check:
                logger.debug(f"Found intentional redundancy keyword: {keyword}")
                return True

        return False

    def validate_evidence_quality(
        self, evidence: List[EvidenceItem], min_avg_relevance: float = 0.5
    ) -> tuple[bool, Optional[str]]:
        """
        Validate evidence quality.

        Args:
            evidence: List of evidence items
            min_avg_relevance: Minimum average relevance score

        Returns:
            Tuple of (is_valid, failure_reason)
        """
        if not evidence:
            return False, "No evidence provided"

        # Check average relevance
        avg_relevance = sum(e.relevance_score for e in evidence) / len(evidence)
        if avg_relevance < min_avg_relevance:
            return False, f"Average relevance {avg_relevance:.2f} below {min_avg_relevance}"

        # Check for evidence diversity (different sources/channels)
        sources = set(e.source for e in evidence)
        if len(sources) < 1:
            return False, "Evidence must come from at least one source"

        # Check for temporal spread (not all from same time)
        if len(evidence) > 1:
            timestamps = sorted([e.timestamp for e in evidence])
            time_spread = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_spread < 60:  # Less than 1 minute spread
                return False, "Evidence too temporally concentrated"

        return True, None

    def validate_llm_verification(
        self, verification: LLMVerification, min_confidence: float = 0.7
    ) -> tuple[bool, Optional[str]]:
        """
        Validate LLM verification quality.

        Args:
            verification: LLM verification result
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (is_valid, failure_reason)
        """
        if not verification.is_duplicate:
            return False, "LLM determined this is not duplicate work"

        if verification.confidence < min_confidence:
            return (
                False,
                f"LLM confidence {verification.confidence:.2f} below {min_confidence}",
            )

        if not verification.reasoning or len(verification.reasoning) < 20:
            return False, "LLM reasoning too brief or missing"

        if not verification.evidence or len(verification.evidence) < 1:
            return False, "LLM did not provide supporting evidence"

        return True, None


class DuplicateWorkValidator(GapValidator):
    """
    Specialized validator for duplicate work gaps.

    Adds duplicate-work-specific validation rules on top of base validator.
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        min_teams: int = 2,
        min_evidence_items: int = 3,
        min_temporal_overlap_days: int = 3,
        min_overlap_ratio: float = 0.5,
        **kwargs: Any,
    ):
        """
        Initialize duplicate work validator.

        Args:
            min_confidence: Minimum confidence score
            min_teams: Minimum teams (must be 2+ for duplicate work)
            min_evidence_items: Minimum evidence items
            min_temporal_overlap_days: Minimum temporal overlap
            min_overlap_ratio: Minimum overlap ratio from LLM
            **kwargs: Additional base validator arguments
        """
        super().__init__(
            min_confidence=min_confidence,
            min_teams=min_teams,
            min_evidence_items=min_evidence_items,
            min_temporal_overlap_days=min_temporal_overlap_days,
            require_llm_verification=True,
            **kwargs,
        )
        self.min_overlap_ratio = min_overlap_ratio

    def validate(self, gap: CoordinationGap) -> tuple[bool, List[str]]:
        """
        Validate duplicate work gap with specialized rules.

        Args:
            gap: Gap to validate

        Returns:
            Tuple of (is_valid, list of validation failures)
        """
        # Run base validation
        is_valid, failures = super().validate(gap)

        # Add duplicate-work-specific validation
        if gap.verification and gap.verification.overlap_ratio < self.min_overlap_ratio:
            failures.append(
                f"Overlap ratio {gap.verification.overlap_ratio:.2f} "
                f"below threshold {self.min_overlap_ratio}"
            )
            is_valid = False

        # Ensure temporal overlap exists for duplicate work
        if not gap.temporal_overlap:
            failures.append("Temporal overlap data missing for duplicate work")
            is_valid = False

        # Duplicate work requires at least 2 teams
        if len(gap.teams_involved) < 2:
            failures.append(
                f"Duplicate work requires 2+ teams, found {len(gap.teams_involved)}"
            )
            is_valid = False

        return is_valid, failures
