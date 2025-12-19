"""
Feature extraction for ranking.

Extracts 24 ranking features across query-document similarity,
temporal signals, engagement metrics, authority, and content.
"""

import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from src.ranking.feature_config import (
    FeatureConfig,
    FeatureDefinition,
    NormalizationMethod,
    get_default_config,
)
from src.utils.time_utils import (
    calculate_edit_freshness,
    calculate_recency_score,
    calculate_response_velocity,
    calculate_temporal_relevance,
    detect_activity_burst,
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts ranking features from query-document pairs.

    This class computes 24 features across 5 categories:
    - Query-document similarity (6 features)
    - Temporal signals (5 features)
    - Engagement metrics (6 features)
    - Source authority (4 features)
    - Content features (3 features)
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        track_stats: bool = False
    ):
        """
        Initialize feature extractor.

        Args:
            config: Feature configuration (default: all enabled features)
            track_stats: Whether to track feature statistics for normalization
        """
        self.config = config or get_default_config()
        self.track_stats = track_stats

        # Statistics for normalization (if tracking)
        self.feature_stats: Dict[str, Dict[str, List[float]]] = {}

        logger.info(
            f"FeatureExtractor initialized with {len(self.config.enabled_features)} features"
        )

    def extract(
        self,
        query: str,
        document: Dict[str, Any],
        query_context: Optional[Dict[str, Any]] = None,
        semantic_score: Optional[float] = None,
        bm25_score: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Extract all enabled features for a query-document pair.

        Args:
            query: Search query text
            document: Document data (message with metadata)
            query_context: Optional query context (temporal hints, etc.)
            semantic_score: Pre-computed semantic similarity score
            bm25_score: Pre-computed BM25 score

        Returns:
            Dictionary mapping feature names to normalized scores

        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract(
            ...     query="OAuth implementation",
            ...     document=message_dict,
            ...     semantic_score=0.92,
            ...     bm25_score=0.85
            ... )
            >>> features
            {
                'semantic_score': 0.92,
                'bm25_score': 0.85,
                'exact_match': 1.0,
                'recency': 0.95,
                'thread_depth': 0.60,
                ...
            }
        """
        features = {}

        # Extract each enabled feature
        for feature_name in self.config.enabled_features:
            try:
                value = self._extract_feature(
                    feature_name=feature_name,
                    query=query,
                    document=document,
                    query_context=query_context or {},
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                )

                # Normalize feature value
                normalized_value = self._normalize_feature(feature_name, value)

                features[feature_name] = normalized_value

                # Track statistics if enabled
                if self.track_stats:
                    self._update_stats(feature_name, value)

            except Exception as e:
                logger.warning(
                    f"Failed to extract feature '{feature_name}': {e}",
                    exc_info=True
                )
                # Use default value on error
                fd = self.config.get_feature_definition(feature_name)
                features[feature_name] = fd.default_value if fd else 0.0

        return features

    def _extract_feature(
        self,
        feature_name: str,
        query: str,
        document: Dict[str, Any],
        query_context: Dict[str, Any],
        semantic_score: Optional[float],
        bm25_score: Optional[float],
    ) -> float:
        """Extract a single feature value (before normalization)."""
        # Query-document similarity features
        if feature_name == "semantic_score":
            return semantic_score if semantic_score is not None else 0.0

        elif feature_name == "bm25_score":
            return bm25_score if bm25_score is not None else 0.0

        elif feature_name == "exact_match":
            return self._compute_exact_match(query, document)

        elif feature_name == "term_coverage":
            return self._compute_term_coverage(query, document)

        elif feature_name == "title_match":
            return self._compute_title_match(query, document)

        elif feature_name == "entity_overlap":
            return self._compute_entity_overlap(query, document)

        # Temporal features
        elif feature_name == "recency":
            return self._compute_recency(document, query_context)

        elif feature_name == "activity_burst":
            return self._compute_activity_burst(document, query_context)

        elif feature_name == "temporal_relevance":
            return self._compute_temporal_relevance(document, query_context)

        elif feature_name == "edit_freshness":
            return self._compute_edit_freshness(document, query_context)

        elif feature_name == "response_velocity":
            return self._compute_response_velocity(document)

        # Engagement features
        elif feature_name == "thread_depth":
            return self._compute_thread_depth(document)

        elif feature_name == "participant_count":
            return self._compute_participant_count(document)

        elif feature_name == "reaction_count":
            return self._compute_reaction_count(document)

        elif feature_name == "reaction_diversity":
            return self._compute_reaction_diversity(document)

        elif feature_name == "cross_team_engagement":
            return self._compute_cross_team_engagement(document)

        elif feature_name == "view_count":
            return self._compute_view_count(document)

        # Authority features
        elif feature_name == "author_seniority":
            return self._compute_author_seniority(document)

        elif feature_name == "channel_importance":
            return self._compute_channel_importance(document)

        elif feature_name == "team_influence":
            return self._compute_team_influence(document)

        elif feature_name == "domain_expertise":
            return self._compute_domain_expertise(document, query)

        # Content features
        elif feature_name == "message_length":
            return self._compute_message_length(document)

        elif feature_name == "code_snippet_present":
            return self._compute_code_snippet_present(document)

        elif feature_name == "link_count":
            return self._compute_link_count(document)

        else:
            logger.warning(f"Unknown feature: {feature_name}")
            return 0.0

    # =========================================================================
    # QUERY-DOCUMENT SIMILARITY FEATURES
    # =========================================================================

    def _compute_exact_match(self, query: str, document: Dict[str, Any]) -> float:
        """Check if query appears exactly in document (binary)."""
        content = document.get("content", "").lower()
        query_lower = query.lower().strip()

        # Empty query should not match
        if not query_lower:
            return 0.0

        return 1.0 if query_lower in content else 0.0

    def _compute_term_coverage(self, query: str, document: Dict[str, Any]) -> float:
        """Percentage of query terms found in document."""
        content = document.get("content", "").lower()

        # Tokenize query
        query_terms = set(re.findall(r'\w+', query.lower()))

        if not query_terms:
            return 0.0

        # Check which terms appear in document
        matches = sum(1 for term in query_terms if term in content)

        return matches / len(query_terms)

    def _compute_title_match(self, query: str, document: Dict[str, Any]) -> float:
        """Query terms appear in channel/title."""
        channel = document.get("channel", "").lower()

        if not channel:
            return 0.0

        query_terms = set(re.findall(r'\w+', query.lower()))

        if not query_terms:
            return 0.0

        # Check matches in channel name
        matches = sum(1 for term in query_terms if term in channel)

        return matches / len(query_terms)

    def _compute_entity_overlap(self, query: str, document: Dict[str, Any]) -> float:
        """Shared named entities (simplified: capitalized words)."""
        # Extract capitalized words as entity proxies
        query_entities = set(re.findall(r'\b[A-Z][a-z]+\b', query))
        content = document.get("content", "")
        doc_entities = set(re.findall(r'\b[A-Z][a-z]+\b', content))

        if not query_entities:
            return 0.0

        overlap = query_entities.intersection(doc_entities)
        return len(overlap) / len(query_entities)

    # =========================================================================
    # TEMPORAL FEATURES
    # =========================================================================

    def _compute_recency(
        self,
        document: Dict[str, Any],
        query_context: Dict[str, Any]
    ) -> float:
        """Time since message (exponential decay)."""
        timestamp = document.get("timestamp")

        if not timestamp:
            return 0.0

        # Convert to datetime if string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        reference_time = query_context.get("reference_time")
        half_life_days = query_context.get("recency_half_life", 30.0)

        return calculate_recency_score(
            message_time=timestamp,
            reference_time=reference_time,
            half_life_days=half_life_days
        )

    def _compute_activity_burst(
        self,
        document: Dict[str, Any],
        query_context: Dict[str, Any]
    ) -> float:
        """Recent activity spike detection."""
        metadata = document.get("message_metadata") or {}
        thread_timestamps = metadata.get("thread_timestamps", [])

        if not thread_timestamps:
            return 0.0

        # Convert string timestamps to datetime
        timestamps = []
        for ts in thread_timestamps:
            if isinstance(ts, str):
                timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            elif isinstance(ts, datetime):
                timestamps.append(ts)

        reference_time = query_context.get("reference_time")

        return detect_activity_burst(
            timestamps=timestamps,
            reference_time=reference_time,
            window_hours=24.0
        )

    def _compute_temporal_relevance(
        self,
        document: Dict[str, Any],
        query_context: Dict[str, Any]
    ) -> float:
        """Time alignment with query context."""
        timestamp = document.get("timestamp")

        if not timestamp:
            return 0.5  # Neutral

        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        context_time = query_context.get("temporal_context") or query_context.get("reference_time")
        relevance_window_days = query_context.get("relevance_window_days", 90.0)

        return calculate_temporal_relevance(
            message_time=timestamp,
            query_context_time=context_time,
            relevance_window_days=relevance_window_days
        )

    def _compute_edit_freshness(
        self,
        document: Dict[str, Any],
        query_context: Dict[str, Any]
    ) -> float:
        """Time since last edit."""
        created = document.get("timestamp") or document.get("created_at")
        metadata = document.get("message_metadata") or {}
        edited = metadata.get("edited_at")

        if not created:
            return 0.0

        # Convert to datetime
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace('Z', '+00:00'))
        if edited and isinstance(edited, str):
            edited = datetime.fromisoformat(edited.replace('Z', '+00:00'))

        reference_time = query_context.get("reference_time")

        return calculate_edit_freshness(
            created_time=created,
            edited_time=edited,
            reference_time=reference_time,
            half_life_days=7.0
        )

    def _compute_response_velocity(self, document: Dict[str, Any]) -> float:
        """Reply rate in conversation thread."""
        metadata = document.get("message_metadata") or {}
        thread_timestamps = metadata.get("thread_timestamps", [])

        if len(thread_timestamps) < 2:
            return 0.0

        # Convert to datetime
        timestamps = []
        for ts in thread_timestamps:
            if isinstance(ts, str):
                timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            elif isinstance(ts, datetime):
                timestamps.append(ts)

        return calculate_response_velocity(timestamps, window_hours=24.0)

    # =========================================================================
    # ENGAGEMENT FEATURES
    # =========================================================================

    def _compute_thread_depth(self, document: Dict[str, Any]) -> float:
        """Number of replies in thread."""
        metadata = document.get("message_metadata") or {}
        reply_count = metadata.get("reply_count", 0)

        return float(reply_count)

    def _compute_participant_count(self, document: Dict[str, Any]) -> float:
        """Unique participants in discussion."""
        metadata = document.get("message_metadata") or {}
        participants = metadata.get("participants", [])

        return float(len(set(participants)))

    def _compute_reaction_count(self, document: Dict[str, Any]) -> float:
        """Total reactions received."""
        metadata = document.get("message_metadata") or {}
        reactions = metadata.get("reactions", {})

        if isinstance(reactions, dict):
            return float(sum(reactions.values()))
        elif isinstance(reactions, list):
            return float(len(reactions))
        else:
            return 0.0

    def _compute_reaction_diversity(self, document: Dict[str, Any]) -> float:
        """Unique reaction types."""
        metadata = document.get("message_metadata") or {}
        reactions = metadata.get("reactions", {})

        if isinstance(reactions, dict):
            return float(len(reactions))
        else:
            return 0.0

    def _compute_cross_team_engagement(self, document: Dict[str, Any]) -> float:
        """Multiple teams involved in discussion."""
        metadata = document.get("message_metadata") or {}
        teams = metadata.get("teams_involved", [])

        return float(len(set(teams)))

    def _compute_view_count(self, document: Dict[str, Any]) -> float:
        """Message views (if available)."""
        metadata = document.get("message_metadata") or {}
        views = metadata.get("view_count", 0)

        return float(views)

    # =========================================================================
    # AUTHORITY FEATURES
    # =========================================================================

    def _compute_author_seniority(self, document: Dict[str, Any]) -> float:
        """Author's organizational seniority/tenure."""
        metadata = document.get("message_metadata") or {}
        seniority = metadata.get("author_seniority", 0.5)

        return float(seniority)

    def _compute_channel_importance(self, document: Dict[str, Any]) -> float:
        """Channel's activity level and membership."""
        metadata = document.get("message_metadata") or {}
        importance = metadata.get("channel_importance", 0.5)

        return float(importance)

    def _compute_team_influence(self, document: Dict[str, Any]) -> float:
        """Team's organizational importance."""
        metadata = document.get("message_metadata") or {}
        influence = metadata.get("team_influence", 0.5)

        return float(influence)

    def _compute_domain_expertise(
        self,
        document: Dict[str, Any],
        query: str
    ) -> float:
        """Author's expertise in topic area."""
        metadata = document.get("message_metadata") or {}
        expertise = metadata.get("domain_expertise", {})

        # Extract topic from query (simplified)
        query_terms = set(re.findall(r'\w+', query.lower()))

        if isinstance(expertise, dict):
            # Check if author has expertise in query topics
            matching_expertise = [
                score for topic, score in expertise.items()
                if any(term in topic.lower() for term in query_terms)
            ]
            return max(matching_expertise) if matching_expertise else 0.5
        else:
            return float(expertise) if expertise else 0.5

    # =========================================================================
    # CONTENT FEATURES
    # =========================================================================

    def _compute_message_length(self, document: Dict[str, Any]) -> float:
        """Character count."""
        content = document.get("content", "")
        return float(len(content))

    def _compute_code_snippet_present(self, document: Dict[str, Any]) -> float:
        """Contains code blocks (binary)."""
        content = document.get("content", "")

        # Check for code block markers
        has_code = (
            "```" in content or
            "`" in content or
            "    " in content  # Indented code blocks
        )

        return 1.0 if has_code else 0.0

    def _compute_link_count(self, document: Dict[str, Any]) -> float:
        """Number of external references."""
        content = document.get("content", "")

        # Count URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, content)

        return float(len(urls))

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize feature value based on configuration."""
        fd = self.config.get_feature_definition(feature_name)

        if not fd:
            return value

        method = fd.normalization

        if method == NormalizationMethod.NONE:
            return value

        elif method == NormalizationMethod.MINMAX:
            return self._minmax_normalize(feature_name, value)

        elif method == NormalizationMethod.LOG:
            return self._log_normalize(value)

        elif method == NormalizationMethod.ZSCORE:
            return self._zscore_normalize(feature_name, value)

        elif method == NormalizationMethod.SIGMOID:
            return self._sigmoid_normalize(value)

        else:
            return value

    def _minmax_normalize(self, feature_name: str, value: float) -> float:
        """Min-max normalization to [0, 1]."""
        stats = self.config.normalization_stats.get(feature_name, {})
        min_val = stats.get("min", 0.0)
        max_val = stats.get("max", 1.0)

        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def _log_normalize(self, value: float) -> float:
        """Log transformation for count features."""
        if value <= 0:
            return 0.0

        # log(1 + x) to handle zero gracefully
        return math.log(1 + value) / math.log(1 + 1000)  # Normalize by log(1001)

    def _zscore_normalize(self, feature_name: str, value: float) -> float:
        """Z-score standardization."""
        stats = self.config.normalization_stats.get(feature_name, {})
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        if std == 0:
            return 0.0

        return (value - mean) / std

    def _sigmoid_normalize(self, value: float) -> float:
        """Sigmoid normalization to [0, 1]."""
        return 1.0 / (1.0 + math.exp(-value))

    def _update_stats(self, feature_name: str, value: float) -> None:
        """Update feature statistics for normalization."""
        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {"values": []}

        self.feature_stats[feature_name]["values"].append(value)

    def get_normalization_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate normalization statistics from tracked values.

        Returns:
            Dictionary mapping feature names to {min, max, mean, std}
        """
        stats = {}

        for feature_name, data in self.feature_stats.items():
            values = data["values"]

            if not values:
                continue

            stats[feature_name] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "std": self._calculate_std(values),
                "count": len(values),
            }

        return stats

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
