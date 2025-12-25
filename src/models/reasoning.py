"""
Multi-step reasoning chains for gap detection and analysis.

This module provides reasoning patterns for complex gap detection tasks
that require multiple LLM calls and intermediate reasoning steps.
"""

import logging
from typing import Any, Optional

from pydantic import BaseModel

from src.models.llm import ClaudeClient
from src.models.prompts import (
    GAP_VERIFICATION_PROMPT,
    INSIGHT_GENERATION_PROMPT,
    RECOMMENDATION_PROMPT,
    format_message_list,
    format_evidence,
    format_teams,
    format_timeframe,
)

logger = logging.getLogger(__name__)


class GapVerificationResult(BaseModel):
    """Result from gap verification reasoning."""

    is_duplicate: bool
    confidence: float
    reasoning: str
    evidence: list[str]
    recommendation: str
    overlap_ratio: float = 0.0


class InsightResult(BaseModel):
    """Result from insight generation reasoning."""

    summary: str
    root_cause: str
    impact: str
    immediate_actions: list[str]
    preventive_measures: list[str]


class RecommendationResult(BaseModel):
    """Result from recommendation generation."""

    immediate_actions: list[dict]
    short_term_fixes: list[dict]
    long_term_improvements: list[dict]
    talking_points: list[str]


class GapReasoningChain:
    """
    Multi-step reasoning chain for gap detection and analysis.

    This class orchestrates multiple LLM calls to:
    1. Verify if a gap exists
    2. Generate insights about the gap
    3. Produce actionable recommendations
    """

    def __init__(self, client: Optional[ClaudeClient] = None):
        """
        Initialize reasoning chain.

        Args:
            client: ClaudeClient instance (creates new one if not provided)
        """
        self.client = client or ClaudeClient()
        logger.info("Initialized GapReasoningChain")

    def verify_duplicate_work(
        self,
        messages: list[dict],
        teams: list[str],
        topic: str,
        start_date: str,
        end_date: str,
        overlap_days: int,
    ) -> GapVerificationResult:
        """
        Verify if clustered messages represent duplicate work.

        Args:
            messages: List of message dictionaries
            teams: List of team names involved
            topic: Topic/feature being discussed
            start_date: Start date of timeframe
            end_date: End date of timeframe
            overlap_days: Number of days of temporal overlap

        Returns:
            GapVerificationResult with verification details
        """
        logger.info(f"Verifying duplicate work for topic: {topic}")

        # Format data for prompt
        message_list = format_message_list(messages)
        team_list = format_teams(teams)
        timeframe = format_timeframe(start_date, end_date, overlap_days)

        # Build prompt
        prompt = GAP_VERIFICATION_PROMPT.format(
            topic=topic, message_list=message_list, teams=team_list, timeframe=timeframe
        )

        # Get LLM verification
        try:
            result = self.client.complete_json(
                prompt=prompt, schema=GapVerificationResult, temperature=0.3
            )

            logger.info(
                f"Verification result: is_duplicate={result['is_duplicate']}, "
                f"confidence={result['confidence']}"
            )

            return GapVerificationResult(**result)

        except Exception as e:
            logger.error(f"Error in gap verification: {e}")
            # Return conservative result on error
            return GapVerificationResult(
                is_duplicate=False,
                confidence=0.0,
                reasoning=f"Verification failed: {str(e)}",
                evidence=[],
                recommendation="Manual review required due to verification error",
                overlap_ratio=0.0,
            )

    def generate_insights(
        self, gap_type: str, teams: list[str], topic: str, evidence: list[dict]
    ) -> InsightResult:
        """
        Generate insights about a detected coordination gap.

        Args:
            gap_type: Type of gap (e.g., "DUPLICATE_WORK")
            teams: Teams involved in the gap
            topic: Topic of the gap
            evidence: Evidence supporting the gap

        Returns:
            InsightResult with analysis and insights
        """
        logger.info(f"Generating insights for {gap_type} gap on topic: {topic}")

        # Format data for prompt
        team_list = format_teams(teams)
        evidence_list = format_evidence(evidence)

        # Build prompt
        prompt = INSIGHT_GENERATION_PROMPT.format(
            gap_type=gap_type, teams=team_list, topic=topic, evidence=evidence_list
        )

        # Get LLM insights
        try:
            result = self.client.complete_json(prompt=prompt, schema=InsightResult, temperature=0.4)

            logger.info(f"Generated insights: {result['summary'][:100]}...")

            return InsightResult(**result)

        except Exception as e:
            logger.error(f"Error in insight generation: {e}")
            # Return basic result on error
            return InsightResult(
                summary=f"Unable to generate insights: {str(e)}",
                root_cause="Analysis error",
                impact="Unknown impact",
                immediate_actions=["Manual review recommended"],
                preventive_measures=["Investigate analysis failure"],
            )

    def generate_recommendations(
        self,
        gap_type: str,
        teams: list[str],
        topic: str,
        impact_score: float,
        evidence: list[dict],
    ) -> RecommendationResult:
        """
        Generate actionable recommendations for addressing a gap.

        Args:
            gap_type: Type of gap
            teams: Teams involved
            topic: Topic of the gap
            impact_score: Impact score (0-1)
            evidence: Evidence for the gap

        Returns:
            RecommendationResult with actionable recommendations
        """
        logger.info(f"Generating recommendations for {gap_type} gap")

        # Format data for prompt
        team_list = format_teams(teams)
        evidence_list = format_evidence(evidence)

        # Build prompt
        prompt = RECOMMENDATION_PROMPT.format(
            gap_type=gap_type,
            teams=team_list,
            topic=topic,
            impact_score=impact_score,
            evidence=evidence_list,
        )

        # Get LLM recommendations
        try:
            result = self.client.complete_json(
                prompt=prompt, schema=RecommendationResult, temperature=0.5
            )

            logger.info(
                f"Generated {len(result.get('immediate_actions', []))} immediate actions"
            )

            return RecommendationResult(**result)

        except Exception as e:
            logger.error(f"Error in recommendation generation: {e}")
            # Return basic recommendations on error
            return RecommendationResult(
                immediate_actions=[
                    {
                        "action": "Manual investigation required",
                        "owner": "Team leads",
                        "urgency": "high",
                    }
                ],
                short_term_fixes=[
                    {
                        "action": "Review gap details",
                        "timeline": "This week",
                        "expected_impact": "Better understanding",
                    }
                ],
                long_term_improvements=[
                    {
                        "action": "Improve coordination processes",
                        "rationale": "Prevent similar gaps",
                        "effort": "medium",
                    }
                ],
                talking_points=[
                    "Coordination gap detected",
                    "Recommendation generation encountered an error",
                    "Manual review recommended",
                ],
            )

    def analyze_gap_full(
        self,
        messages: list[dict],
        teams: list[str],
        topic: str,
        start_date: str,
        end_date: str,
        overlap_days: int,
        evidence: Optional[list[dict]] = None,
    ) -> dict:
        """
        Full multi-step gap analysis pipeline.

        This method orchestrates the complete reasoning chain:
        1. Verify the gap exists
        2. Generate insights if verified
        3. Produce recommendations if high confidence

        Args:
            messages: List of message dictionaries
            teams: List of team names
            topic: Topic being discussed
            start_date: Start date
            end_date: End date
            overlap_days: Days of overlap
            evidence: Optional evidence (uses messages if not provided)

        Returns:
            Dictionary with complete analysis results
        """
        logger.info(f"Starting full gap analysis for topic: {topic}")

        # Step 1: Verify gap
        verification = self.verify_duplicate_work(
            messages=messages,
            teams=teams,
            topic=topic,
            start_date=start_date,
            end_date=end_date,
            overlap_days=overlap_days,
        )

        result = {"verification": verification.model_dump(), "insights": None, "recommendations": None}

        # Step 2: Generate insights if gap is verified with reasonable confidence
        if verification.is_duplicate and verification.confidence >= 0.6:
            logger.info("Gap verified, generating insights...")

            insights = self.generate_insights(
                gap_type="DUPLICATE_WORK",
                teams=teams,
                topic=topic,
                evidence=evidence or messages,
            )

            result["insights"] = insights.model_dump()

            # Step 3: Generate recommendations for high-confidence gaps
            if verification.confidence >= 0.7:
                logger.info("High confidence gap, generating recommendations...")

                recommendations = self.generate_recommendations(
                    gap_type="DUPLICATE_WORK",
                    teams=teams,
                    topic=topic,
                    impact_score=verification.confidence,  # Use confidence as proxy for impact
                    evidence=evidence or messages,
                )

                result["recommendations"] = recommendations.model_dump()

        logger.info(
            f"Completed full gap analysis: is_duplicate={verification.is_duplicate}, "
            f"confidence={verification.confidence}"
        )

        return result

    def batch_verify_gaps(self, potential_gaps: list[dict]) -> list[GapVerificationResult]:
        """
        Batch verify multiple potential gaps.

        Args:
            potential_gaps: List of potential gap dictionaries with required fields

        Returns:
            List of GapVerificationResult
        """
        logger.info(f"Batch verifying {len(potential_gaps)} potential gaps")

        results = []
        for i, gap in enumerate(potential_gaps):
            logger.debug(f"Verifying gap {i+1}/{len(potential_gaps)}")

            try:
                result = self.verify_duplicate_work(
                    messages=gap["messages"],
                    teams=gap["teams"],
                    topic=gap["topic"],
                    start_date=gap["start_date"],
                    end_date=gap["end_date"],
                    overlap_days=gap["overlap_days"],
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error verifying gap {i+1}: {e}")
                # Add failed result
                results.append(
                    GapVerificationResult(
                        is_duplicate=False,
                        confidence=0.0,
                        reasoning=f"Verification error: {str(e)}",
                        evidence=[],
                        recommendation="Manual review required",
                        overlap_ratio=0.0,
                    )
                )

        logger.info(
            f"Batch verification complete: "
            f"{sum(1 for r in results if r.is_duplicate)}/{len(results)} verified as duplicates"
        )

        return results
