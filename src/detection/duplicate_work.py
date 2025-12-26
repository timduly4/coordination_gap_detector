"""
Duplicate work detection algorithm.

This module implements the core algorithm for detecting when multiple teams
are independently working on the same problem without coordination.

Detection Pipeline:
1. Semantic Clustering - Group similar discussions
2. Team Detection - Identify teams in each cluster
3. Temporal Overlap - Check if teams working simultaneously
4. LLM Verification - Use Claude to verify actual duplication
5. Evidence Collection - Gather and rank supporting evidence
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from src.analysis.entity_extraction import EntityExtractor
from src.detection.clustering import MessageCluster, SemanticClusterer
from src.detection.cost_estimation import CostEstimator
from src.detection.impact_scoring import ImpactScorer
from src.detection.patterns import (
    GapDetectionConfig,
    GapType,
    PatternDetector,
    extract_temporal_overlap,
)
from src.detection.validators import DuplicateWorkValidator
from src.models.llm import ClaudeClient
from src.models.prompts import (
    GAP_VERIFICATION_PROMPT,
    format_message_list,
    format_teams,
    format_timeframe,
)
from src.models.schemas import (
    CoordinationGap,
    CostEstimate,
    EvidenceItem,
    ImpactBreakdown,
    LLMVerification,
    TemporalOverlap,
)

logger = logging.getLogger(__name__)


class DuplicateWorkDetector(PatternDetector):
    """
    Detector for duplicate work coordination gaps.

    Identifies when multiple teams are independently solving the same problem,
    wasting effort through lack of coordination.
    """

    def __init__(
        self,
        config: Optional[GapDetectionConfig] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        semantic_clusterer: Optional[SemanticClusterer] = None,
        llm_client: Optional[ClaudeClient] = None,
        validator: Optional[DuplicateWorkValidator] = None,
        impact_scorer: Optional[ImpactScorer] = None,
        cost_estimator: Optional[CostEstimator] = None,
    ):
        """
        Initialize duplicate work detector.

        Args:
            config: Detection configuration
            entity_extractor: Entity extractor instance
            semantic_clusterer: Semantic clusterer instance
            llm_client: Claude API client
            validator: Gap validator instance
            impact_scorer: Impact scorer instance
            cost_estimator: Cost estimator instance
        """
        super().__init__(config)

        self.entity_extractor = entity_extractor or EntityExtractor()
        self.semantic_clusterer = semantic_clusterer or SemanticClusterer(
            similarity_threshold=self.config.similarity_threshold,
            min_cluster_size=self.config.min_cluster_size,
            time_window_days=self.config.time_window_days,
        )
        self.llm_client = llm_client or ClaudeClient()
        self.validator = validator or DuplicateWorkValidator(
            min_confidence=self.config.llm_confidence_threshold,
            min_teams=self.config.min_teams,
            min_temporal_overlap_days=self.config.min_temporal_overlap_days,
        )
        self.impact_scorer = impact_scorer or ImpactScorer()
        self.cost_estimator = cost_estimator or CostEstimator()

        logger.info(
            f"DuplicateWorkDetector initialized with similarity_threshold="
            f"{self.config.similarity_threshold}, min_teams={self.config.min_teams}"
        )

    async def detect(
        self, messages: List[Any], embeddings: Optional[List[np.ndarray]] = None, **kwargs: Any
    ) -> List[CoordinationGap]:
        """
        Detect duplicate work gaps in messages.

        Args:
            messages: List of message objects (with id, content, timestamp, author, channel)
            embeddings: Optional pre-computed embeddings (will compute if not provided)
            **kwargs: Additional parameters

        Returns:
            List of detected and validated CoordinationGap objects
        """
        logger.info(f"Starting duplicate work detection on {len(messages)} messages")

        if not messages:
            logger.warning("No messages provided for detection")
            return []

        # Stage 1: Semantic Clustering
        clusters = await self._cluster_messages(messages, embeddings)
        logger.info(f"Found {len(clusters)} semantic clusters")

        if not clusters:
            logger.info("No clusters found, no gaps to detect")
            return []

        # Stage 2-5: Detect gaps from clusters
        gaps = []
        for cluster in clusters:
            cluster_gaps = await self._detect_gaps_in_cluster(cluster, messages)
            gaps.extend(cluster_gaps)

        logger.info(f"Detected {len(gaps)} potential gaps before validation")

        # Validate gaps
        validated_gaps = self._validate_gaps(gaps)
        logger.info(f"Validated {len(validated_gaps)} gaps")

        return validated_gaps

    async def _cluster_messages(
        self, messages: List[Any], embeddings: Optional[List[np.ndarray]] = None
    ) -> List[MessageCluster]:
        """
        Cluster messages by semantic similarity.

        Args:
            messages: List of messages
            embeddings: Optional pre-computed embeddings

        Returns:
            List of MessageCluster objects
        """
        if embeddings is None:
            # In a real implementation, would get embeddings from vector store
            # For now, assume they're provided or return empty
            logger.warning("No embeddings provided, skipping clustering")
            return []

        # Get timestamps for temporal filtering
        timestamps = [msg.timestamp for msg in messages]

        # Cluster embeddings
        cluster_indices = self.semantic_clusterer.cluster(
            embeddings=embeddings,
            timestamps=timestamps,
        )

        # Create MessageCluster objects
        clusters = self.semantic_clusterer.create_message_clusters(
            messages=messages,
            embeddings=embeddings,
            cluster_indices=cluster_indices,
        )

        return clusters

    async def _detect_gaps_in_cluster(
        self, cluster: MessageCluster, all_messages: List[Any]
    ) -> List[CoordinationGap]:
        """
        Detect gaps within a single cluster.

        Args:
            cluster: Message cluster to analyze
            all_messages: All messages (for reference)

        Returns:
            List of detected gaps (may be empty)
        """
        # Get messages in this cluster
        cluster_messages = [msg for msg in all_messages if msg.id in cluster.message_ids]

        if len(cluster_messages) < self.config.min_cluster_size:
            return []

        # Stage 2: Extract teams from cluster
        teams = self._extract_teams_from_messages(cluster_messages)

        if len(teams) < self.config.min_teams:
            logger.debug(
                f"Cluster {cluster.cluster_id} has {len(teams)} teams, "
                f"need {self.config.min_teams}"
            )
            return []

        logger.debug(
            f"Cluster {cluster.cluster_id} has {len(teams)} teams: {teams}"
        )

        # Stage 3: Check temporal overlap
        temporal_overlap = self._compute_temporal_overlap(cluster_messages, teams)

        if temporal_overlap.overlap_days < self.config.min_temporal_overlap_days:
            logger.debug(
                f"Cluster {cluster.cluster_id} has {temporal_overlap.overlap_days} days overlap, "
                f"need {self.config.min_temporal_overlap_days}"
            )
            return []

        # Stage 4: LLM Verification
        llm_verification = await self._verify_with_llm(cluster_messages, teams, temporal_overlap)

        if not llm_verification.is_duplicate:
            logger.debug(
                f"Cluster {cluster.cluster_id} LLM verification failed: not duplicate work"
            )
            return []

        # Stage 5: Collect and rank evidence
        evidence = self._collect_evidence(cluster_messages, teams)

        # Create gap object
        gap = self._create_gap(
            cluster=cluster,
            teams=teams,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            llm_verification=llm_verification,
        )

        return [gap]

    def _extract_teams_from_messages(self, messages: List[Any]) -> List[str]:
        """
        Extract unique teams from messages.

        Args:
            messages: List of messages

        Returns:
            List of unique team names
        """
        teams = set()

        for msg in messages:
            # Extract entities from message
            msg_dict = {
                "content": msg.content,
                "channel": getattr(msg, "channel", ""),
                "author": getattr(msg, "author", ""),
            }

            entities = self.entity_extractor.extract(msg_dict, extract_teams=True)

            # Add extracted teams
            for team in entities.teams:
                teams.add(team.normalized)

        return sorted(list(teams))

    def _compute_temporal_overlap(
        self, messages: List[Any], teams: List[str]
    ) -> TemporalOverlap:
        """
        Compute temporal overlap between teams.

        Args:
            messages: List of messages
            teams: List of team names

        Returns:
            TemporalOverlap object with analysis
        """
        # Build timelines for each team
        team_timelines: Dict[str, List[datetime]] = {team: [] for team in teams}

        for msg in messages:
            # Determine which team this message belongs to
            msg_dict = {
                "content": msg.content,
                "channel": getattr(msg, "channel", ""),
                "author": getattr(msg, "author", ""),
            }
            entities = self.entity_extractor.extract(msg_dict, extract_teams=True)

            for team_entity in entities.teams:
                if team_entity.normalized in team_timelines:
                    team_timelines[team_entity.normalized].append(msg.timestamp)

        # Compute overall overlap
        if len(team_timelines) < 2:
            # Need at least 2 teams for overlap
            return TemporalOverlap(
                start=min(msg.timestamp for msg in messages),
                end=max(msg.timestamp for msg in messages),
                overlap_days=0,
                team_timelines={},
            )

        # Find maximum overlap across all team pairs
        max_overlap_days = 0
        overlap_start = None
        overlap_end = None
        formatted_timelines = {}

        team_list = list(team_timelines.keys())
        for i, team1 in enumerate(team_list):
            for team2 in team_list[i + 1 :]:
                if team_timelines[team1] and team_timelines[team2]:
                    overlap_days = extract_temporal_overlap(
                        team_timelines[team1], team_timelines[team2]
                    )

                    if overlap_days > max_overlap_days:
                        max_overlap_days = overlap_days
                        # Compute overlap period
                        start1 = min(team_timelines[team1])
                        end1 = max(team_timelines[team1])
                        start2 = min(team_timelines[team2])
                        end2 = max(team_timelines[team2])
                        overlap_start = max(start1, start2)
                        overlap_end = min(end1, end2)

        # Format team timelines for output
        for team, timestamps in team_timelines.items():
            if timestamps:
                formatted_timelines[team] = {
                    "start": min(timestamps).isoformat(),
                    "end": max(timestamps).isoformat(),
                }

        # Use first/last message if no overlap found
        if overlap_start is None:
            overlap_start = min(msg.timestamp for msg in messages)
            overlap_end = max(msg.timestamp for msg in messages)

        return TemporalOverlap(
            start=overlap_start,
            end=overlap_end,
            overlap_days=max_overlap_days,
            team_timelines=formatted_timelines,
        )

    async def _verify_with_llm(
        self,
        messages: List[Any],
        teams: List[str],
        temporal_overlap: TemporalOverlap,
    ) -> LLMVerification:
        """
        Verify duplicate work using Claude LLM.

        Args:
            messages: Cluster messages
            teams: Teams involved
            temporal_overlap: Temporal overlap analysis

        Returns:
            LLMVerification object
        """
        # Format messages for prompt
        message_dicts = [
            {
                "content": msg.content,
                "author": getattr(msg, "author", "unknown"),
                "channel": getattr(msg, "channel", "unknown"),
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in messages[:10]  # Limit to 10 messages for prompt
        ]

        # Infer topic from messages
        topic = self._infer_topic(messages)

        # Format prompt
        prompt = GAP_VERIFICATION_PROMPT.format(
            topic=topic,
            message_list=format_message_list(message_dicts),
            teams=format_teams(teams),
            timeframe=format_timeframe(
                temporal_overlap.start.strftime("%Y-%m-%d"),
                temporal_overlap.end.strftime("%Y-%m-%d"),
                temporal_overlap.overlap_days,
            ),
        )

        # Call LLM
        try:
            response = await self.llm_client.complete_async(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
            )

            # Parse JSON response
            result = self.llm_client.parse_json(response)

            return LLMVerification(
                is_duplicate=result.get("is_duplicate", False),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", ""),
                evidence=result.get("evidence", []),
                recommendation=result.get("recommendation", ""),
                overlap_ratio=result.get("overlap_ratio", 0.0),
            )

        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            # Return conservative default
            return LLMVerification(
                is_duplicate=False,
                confidence=0.0,
                reasoning=f"LLM verification error: {str(e)}",
                evidence=[],
                recommendation="Manual review required",
                overlap_ratio=0.0,
            )

    def _infer_topic(self, messages: List[Any]) -> str:
        """
        Infer main topic from messages.

        Args:
            messages: List of messages

        Returns:
            Topic string
        """
        # Simple approach: extract most common technical terms
        all_terms = []

        for msg in messages:
            msg_dict = {
                "content": msg.content,
                "channel": getattr(msg, "channel", ""),
                "author": getattr(msg, "author", ""),
            }
            entities = self.entity_extractor.extract(msg_dict, extract_projects=True)

            for project in entities.projects:
                all_terms.append(project.normalized)

        if all_terms:
            # Return most common term
            from collections import Counter

            counter = Counter(all_terms)
            return counter.most_common(1)[0][0]

        # Fallback: use first few words from first message
        if messages:
            words = messages[0].content.split()[:5]
            return " ".join(words)

        return "Unknown topic"

    def _collect_evidence(
        self, messages: List[Any], teams: List[str]
    ) -> List[EvidenceItem]:
        """
        Collect and rank evidence for the gap.

        Args:
            messages: Cluster messages
            teams: Teams involved

        Returns:
            List of EvidenceItem objects, ranked by relevance
        """
        evidence_items = []

        # Assign each message to a team
        team_messages: Dict[str, List[Any]] = {team: [] for team in teams}

        for msg in messages:
            msg_dict = {
                "content": msg.content,
                "channel": getattr(msg, "channel", ""),
                "author": getattr(msg, "author", ""),
            }
            entities = self.entity_extractor.extract(msg_dict, extract_teams=True)

            # Assign to first matching team
            assigned = False
            for team_entity in entities.teams:
                if team_entity.normalized in team_messages:
                    team_messages[team_entity.normalized].append(msg)
                    assigned = True
                    break

            # If no team matched, try to infer from channel
            if not assigned and hasattr(msg, "channel"):
                for team in teams:
                    if team.lower() in msg.channel.lower():
                        team_messages[team].append(msg)
                        break

        # Select top messages from each team
        for team, team_msgs in team_messages.items():
            # Sort by timestamp (earliest first)
            sorted_msgs = sorted(team_msgs, key=lambda m: m.timestamp)

            # Take first 3 messages from each team
            for msg in sorted_msgs[:3]:
                # Compute relevance score (earlier messages more relevant)
                time_rank = sorted_msgs.index(msg) + 1
                relevance_score = 1.0 / time_rank

                evidence = EvidenceItem(
                    source="slack",  # TODO: Get from message metadata
                    message_id=msg.id,
                    channel=getattr(msg, "channel", None),
                    author=getattr(msg, "author", None),
                    content=msg.content[:500],  # Truncate long messages
                    timestamp=msg.timestamp,
                    relevance_score=relevance_score,
                    team=team,
                )
                evidence_items.append(evidence)

        # Sort by relevance score
        evidence_items.sort(key=lambda e: e.relevance_score, reverse=True)

        return evidence_items

    def _create_gap(
        self,
        cluster: MessageCluster,
        teams: List[str],
        evidence: List[EvidenceItem],
        temporal_overlap: TemporalOverlap,
        llm_verification: LLMVerification,
    ) -> CoordinationGap:
        """
        Create CoordinationGap object from detection results.

        Args:
            cluster: Source cluster
            teams: Teams involved
            evidence: Evidence items
            temporal_overlap: Temporal analysis
            llm_verification: LLM verification

        Returns:
            CoordinationGap object
        """
        # Generate gap ID
        gap_id = f"gap_{uuid4().hex[:12]}"

        # Infer topic
        topic = self._infer_topic_from_evidence(evidence)

        # Generate title
        title = f"Multiple teams working on {topic}"

        # Compute confidence (combination of LLM and cluster quality)
        confidence = (
            0.7 * llm_verification.confidence + 0.3 * cluster.avg_similarity
        )

        # Estimate impact score with multi-signal approach (Milestone 3E)
        impact_score, impact_breakdown = self._estimate_impact_score(
            teams=teams,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            llm_verification=llm_verification,
            cluster=cluster,
            project_tags=None,  # Could be enhanced with project tag extraction
        )

        # Determine impact tier
        impact_tier = self._determine_impact_tier(impact_score)

        # Generate insight
        insight = (
            f"{len(teams)} teams ({', '.join(teams)}) are independently working on {topic}. "
            f"Detected {len(evidence)} pieces of evidence across {temporal_overlap.overlap_days} days. "
            f"{llm_verification.reasoning}"
        )

        # Use LLM recommendation
        recommendation = llm_verification.recommendation

        # Estimate organizational cost (not Claude API cost)
        estimated_cost = self._estimate_cost(
            teams=teams,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            cluster=cluster,
        )

        return CoordinationGap(
            id=gap_id,
            type=GapType.DUPLICATE_WORK.value,
            title=title,
            topic=topic,
            teams_involved=teams,
            impact_score=impact_score,
            impact_tier=impact_tier,
            confidence=confidence,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            verification=llm_verification,
            impact_breakdown=impact_breakdown,  # Milestone 3E enhancement
            estimated_cost=estimated_cost,
            insight=insight,
            recommendation=recommendation,
            detected_at=datetime.utcnow(),
            cluster_id=cluster.cluster_id,
            people_affected=cluster.participant_count,
            timespan_days=int(cluster.timespan_days or 0),
            messages_analyzed=cluster.size,
        )

    def _infer_topic_from_evidence(self, evidence: List[EvidenceItem]) -> str:
        """Infer topic from evidence."""
        if not evidence:
            return "unknown topic"

        # Use content from highest relevance evidence
        top_evidence = evidence[0]

        # Extract key words
        words = top_evidence.content.split()[:5]
        return " ".join(words)

    def _estimate_impact_score(
        self,
        teams: List[str],
        evidence: List[EvidenceItem],
        temporal_overlap: TemporalOverlap,
        llm_verification: LLMVerification,
        cluster: MessageCluster,
        project_tags: Optional[List[str]] = None,
    ) -> Tuple[float, ImpactBreakdown]:
        """
        Estimate impact score using multi-signal approach.

        Uses ImpactScorer to combine:
        - Team size (25%): How many people affected?
        - Time investment (25%): How much time wasted?
        - Project criticality (20%): How important is this?
        - Velocity impact (15%): What's the opportunity cost?
        - Duplicate effort (15%): How much actual duplication?

        Args:
            teams: Teams involved
            evidence: Evidence items
            temporal_overlap: Temporal overlap
            llm_verification: LLM verification
            cluster: Message cluster
            project_tags: Optional project tags for criticality

        Returns:
            Tuple of (impact_score, impact_breakdown)
        """
        # Extract commits count from evidence metadata (if available)
        commits_found = sum(
            1 for e in evidence
            if e.metadata and e.metadata.get("type") == "commit"
        )

        # Calculate impact score with breakdown
        impact_score, breakdown = self.impact_scorer.calculate_impact_score(
            teams=teams,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            llm_verification=llm_verification,
            participant_count=cluster.participant_count,
            commits_found=commits_found,
            project_tags=project_tags,
            blocking_work_count=0,  # Could be enhanced with issue tracking integration
        )

        return impact_score, breakdown

    def _determine_impact_tier(self, impact_score: float) -> str:
        """
        Determine impact tier from score.

        Delegates to ImpactScorer for consistent tier assignment.
        """
        return self.impact_scorer.determine_impact_tier(impact_score)

    def _estimate_cost(
        self,
        teams: List[str],
        evidence: List[EvidenceItem],
        temporal_overlap: TemporalOverlap,
        cluster: MessageCluster,
    ) -> CostEstimate:
        """
        Estimate organizational cost using CostEstimator.

        IMPORTANT: This estimates the cost of duplicate work (organizational waste),
        NOT the cost of running the detection system or Claude API.

        Args:
            teams: Teams involved
            evidence: Evidence items
            temporal_overlap: Temporal overlap
            cluster: Message cluster

        Returns:
            CostEstimate object
        """
        # Extract commits count from evidence metadata (if available)
        commits_found = sum(
            1 for e in evidence
            if e.metadata and e.metadata.get("type") == "commit"
        )

        # Use CostEstimator to calculate organizational waste cost
        cost_estimate = self.cost_estimator.estimate_cost(
            teams=teams,
            evidence=evidence,
            temporal_overlap=temporal_overlap,
            commits_found=commits_found,
            participant_count=cluster.participant_count,
        )

        return cost_estimate

    def _validate_gaps(self, gaps: List[CoordinationGap]) -> List[CoordinationGap]:
        """
        Validate detected gaps.

        Args:
            gaps: List of detected gaps

        Returns:
            List of validated gaps
        """
        validated = []

        for gap in gaps:
            is_valid, failures = self.validator.validate(gap)

            if is_valid:
                validated.append(gap)
            else:
                logger.debug(
                    f"Gap {gap.id} failed validation: {', '.join(failures)}"
                )

        return validated

    def validate_gap(self, gap_data: Dict[str, Any]) -> bool:
        """
        Validate gap data (PatternDetector interface).

        Args:
            gap_data: Gap data dictionary

        Returns:
            True if valid
        """
        # Convert to CoordinationGap if needed
        if isinstance(gap_data, dict):
            try:
                gap = CoordinationGap(**gap_data)
            except Exception as e:
                logger.error(f"Failed to parse gap data: {e}")
                return False
        else:
            gap = gap_data

        is_valid, _ = self.validator.validate(gap)
        return is_valid
