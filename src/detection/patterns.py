"""
Base patterns and interfaces for gap detection.

This module provides abstract base classes and common utilities for
implementing different types of coordination gap detection algorithms.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GapType(str, Enum):
    """Types of coordination gaps that can be detected."""

    DUPLICATE_WORK = "duplicate_work"
    MISSING_CONTEXT = "missing_context"
    STALE_DOCS = "stale_docs"
    KNOWLEDGE_SILO = "knowledge_silo"


class DetectionStatus(str, Enum):
    """Status of gap detection process."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ClusterBase(BaseModel):
    """Base schema for message clusters."""

    cluster_id: str = Field(..., description="Unique cluster identifier")
    message_ids: List[int] = Field(..., description="Database message IDs in cluster")
    size: int = Field(..., description="Number of messages in cluster", ge=0)
    avg_similarity: float = Field(
        ..., description="Average intra-cluster similarity", ge=0.0, le=1.0
    )
    timespan_days: Optional[float] = Field(
        None, description="Timespan from first to last message in days"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When cluster was created"
    )


class GapDetectionConfig(BaseModel):
    """Configuration for gap detection algorithms."""

    # Clustering parameters
    similarity_threshold: float = Field(
        default=0.85,
        description="Minimum similarity for clustering (0-1)",
        ge=0.0,
        le=1.0,
    )
    min_cluster_size: int = Field(
        default=2, description="Minimum messages per cluster", ge=1
    )
    time_window_days: int = Field(
        default=30, description="Time window for clustering", ge=1, le=365
    )

    # Detection parameters
    min_teams: int = Field(
        default=2, description="Minimum teams for gap detection", ge=1
    )
    min_temporal_overlap_days: int = Field(
        default=3, description="Minimum days of temporal overlap", ge=0
    )
    llm_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum LLM confidence for gap detection",
        ge=0.0,
        le=1.0,
    )

    # Impact scoring parameters
    min_impact_score: float = Field(
        default=0.0, description="Minimum impact score to report", ge=0.0, le=1.0
    )

    # Performance parameters
    max_messages: Optional[int] = Field(
        None, description="Maximum messages to process (None = unlimited)"
    )


class PatternDetector(ABC):
    """
    Abstract base class for coordination gap detection algorithms.

    Subclasses should implement specific detection logic for different
    types of coordination gaps (duplicate work, missing context, etc.).
    """

    def __init__(self, config: Optional[GapDetectionConfig] = None):
        """
        Initialize pattern detector.

        Args:
            config: Detection configuration (uses defaults if not provided)
        """
        self.config = config or GapDetectionConfig()

    @abstractmethod
    async def detect(
        self, messages: List[Any], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Detect coordination gaps in a set of messages.

        Args:
            messages: List of messages to analyze
            **kwargs: Additional detection parameters

        Returns:
            List of detected gaps with evidence and metadata
        """
        pass

    @abstractmethod
    def validate_gap(self, gap_data: Dict[str, Any]) -> bool:
        """
        Validate that a detected gap meets quality criteria.

        Args:
            gap_data: Gap detection result to validate

        Returns:
            True if gap is valid, False otherwise
        """
        pass

    def get_config(self) -> GapDetectionConfig:
        """Get current detection configuration."""
        return self.config

    def update_config(self, **kwargs: Any) -> None:
        """
        Update detection configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


class ClusteringStrategy(ABC):
    """
    Abstract base class for clustering algorithms.

    Different strategies can be implemented (DBSCAN, hierarchical, etc.)
    to group similar messages together.
    """

    @abstractmethod
    def cluster(
        self, embeddings: List[Any], **kwargs: Any
    ) -> List[List[int]]:
        """
        Cluster embeddings into groups.

        Args:
            embeddings: List of embedding vectors
            **kwargs: Algorithm-specific parameters

        Returns:
            List of clusters, where each cluster is a list of indices
        """
        pass

    @abstractmethod
    def get_cluster_quality_metrics(
        self, embeddings: List[Any], clusters: List[List[int]]
    ) -> Dict[str, float]:
        """
        Compute quality metrics for clustering result.

        Args:
            embeddings: Original embedding vectors
            clusters: Clustering result (list of index lists)

        Returns:
            Dictionary of quality metrics (silhouette score, etc.)
        """
        pass


def filter_by_timeframe(
    messages: List[Any], start_time: datetime, end_time: datetime
) -> List[Any]:
    """
    Filter messages by time window.

    Args:
        messages: List of messages with 'timestamp' attribute
        start_time: Start of time window
        end_time: End of time window

    Returns:
        Filtered list of messages within timeframe
    """
    return [
        msg for msg in messages if start_time <= msg.timestamp <= end_time
    ]


def extract_temporal_overlap(
    timeline1: List[datetime], timeline2: List[datetime]
) -> int:
    """
    Calculate temporal overlap between two timelines in days.

    Args:
        timeline1: List of timestamps for first entity
        timeline2: List of timestamps for second entity

    Returns:
        Number of overlapping days
    """
    if not timeline1 or not timeline2:
        return 0

    # Get date ranges
    start1, end1 = min(timeline1).date(), max(timeline1).date()
    start2, end2 = min(timeline2).date(), max(timeline2).date()

    # Calculate overlap
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start > overlap_end:
        return 0

    return (overlap_end - overlap_start).days
