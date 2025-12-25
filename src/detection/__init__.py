"""Detection module for coordination gap patterns."""

from src.detection.clustering import MessageCluster, SemanticClusterer
from src.detection.duplicate_work import DuplicateWorkDetector
from src.detection.patterns import (
    ClusterBase,
    ClusteringStrategy,
    DetectionStatus,
    GapDetectionConfig,
    GapType,
    PatternDetector,
)
from src.detection.validators import DuplicateWorkValidator, GapValidator

__all__ = [
    "MessageCluster",
    "SemanticClusterer",
    "DuplicateWorkDetector",
    "ClusterBase",
    "ClusteringStrategy",
    "GapType",
    "DetectionStatus",
    "GapDetectionConfig",
    "PatternDetector",
    "GapValidator",
    "DuplicateWorkValidator",
]
