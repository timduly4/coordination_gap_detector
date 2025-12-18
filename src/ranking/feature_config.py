"""
Feature configuration for ranking.

Defines all available features, their types, normalization strategies,
and importance weights.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FeatureType(Enum):
    """Feature type categories."""
    QUERY_DOC_SIMILARITY = "query_doc_similarity"
    TEMPORAL = "temporal"
    ENGAGEMENT = "engagement"
    AUTHORITY = "authority"
    CONTENT = "content"


class NormalizationMethod(Enum):
    """Feature normalization methods."""
    NONE = "none"  # No normalization
    MINMAX = "minmax"  # Min-max scaling to [0, 1]
    ZSCORE = "zscore"  # Z-score standardization
    LOG = "log"  # Log transformation
    SIGMOID = "sigmoid"  # Sigmoid scaling


@dataclass
class FeatureDefinition:
    """
    Definition of a single ranking feature.

    Attributes:
        name: Feature identifier
        feature_type: Category of feature
        description: Human-readable description
        normalization: Normalization method to apply
        default_value: Default when feature unavailable
        importance: Relative importance weight (0-1)
        enabled: Whether feature is active
    """
    name: str
    feature_type: FeatureType
    description: str
    normalization: NormalizationMethod = NormalizationMethod.MINMAX
    default_value: float = 0.0
    importance: float = 0.5
    enabled: bool = True


# Define all available features
FEATURE_DEFINITIONS: List[FeatureDefinition] = [
    # =========================================================================
    # QUERY-DOCUMENT SIMILARITY FEATURES (6)
    # =========================================================================
    FeatureDefinition(
        name="semantic_score",
        feature_type=FeatureType.QUERY_DOC_SIMILARITY,
        description="Cosine similarity between query and document embeddings",
        normalization=NormalizationMethod.NONE,  # Already in [0, 1]
        default_value=0.0,
        importance=0.9,
        enabled=True,
    ),
    FeatureDefinition(
        name="bm25_score",
        feature_type=FeatureType.QUERY_DOC_SIMILARITY,
        description="BM25 keyword relevance score",
        normalization=NormalizationMethod.MINMAX,  # BM25 scores vary
        default_value=0.0,
        importance=0.85,
        enabled=True,
    ),
    FeatureDefinition(
        name="exact_match",
        feature_type=FeatureType.QUERY_DOC_SIMILARITY,
        description="Exact phrase match bonus (binary)",
        normalization=NormalizationMethod.NONE,  # Binary {0, 1}
        default_value=0.0,
        importance=0.7,
        enabled=True,
    ),
    FeatureDefinition(
        name="term_coverage",
        feature_type=FeatureType.QUERY_DOC_SIMILARITY,
        description="Percentage of query terms found in document",
        normalization=NormalizationMethod.NONE,  # Already percentage
        default_value=0.0,
        importance=0.75,
        enabled=True,
    ),
    FeatureDefinition(
        name="title_match",
        feature_type=FeatureType.QUERY_DOC_SIMILARITY,
        description="Query terms appear in title/channel name",
        normalization=NormalizationMethod.NONE,  # Binary or percentage
        default_value=0.0,
        importance=0.65,
        enabled=True,
    ),
    FeatureDefinition(
        name="entity_overlap",
        feature_type=FeatureType.QUERY_DOC_SIMILARITY,
        description="Shared named entities (people, teams, projects)",
        normalization=NormalizationMethod.MINMAX,
        default_value=0.0,
        importance=0.6,
        enabled=True,
    ),

    # =========================================================================
    # TEMPORAL FEATURES (5)
    # =========================================================================
    FeatureDefinition(
        name="recency",
        feature_type=FeatureType.TEMPORAL,
        description="Time since message (exponential decay)",
        normalization=NormalizationMethod.NONE,  # Already normalized
        default_value=0.0,
        importance=0.7,
        enabled=True,
    ),
    FeatureDefinition(
        name="activity_burst",
        feature_type=FeatureType.TEMPORAL,
        description="Recent activity spike detection",
        normalization=NormalizationMethod.NONE,  # Already normalized
        default_value=0.0,
        importance=0.55,
        enabled=True,
    ),
    FeatureDefinition(
        name="temporal_relevance",
        feature_type=FeatureType.TEMPORAL,
        description="Time alignment with query context",
        normalization=NormalizationMethod.NONE,  # Already normalized
        default_value=0.5,  # Neutral if no context
        importance=0.5,
        enabled=True,
    ),
    FeatureDefinition(
        name="edit_freshness",
        feature_type=FeatureType.TEMPORAL,
        description="Time since last edit",
        normalization=NormalizationMethod.NONE,  # Already normalized
        default_value=0.0,
        importance=0.45,
        enabled=True,
    ),
    FeatureDefinition(
        name="response_velocity",
        feature_type=FeatureType.TEMPORAL,
        description="Reply rate in conversation thread",
        normalization=NormalizationMethod.NONE,  # Already normalized
        default_value=0.0,
        importance=0.5,
        enabled=True,
    ),

    # =========================================================================
    # ENGAGEMENT FEATURES (6)
    # =========================================================================
    FeatureDefinition(
        name="thread_depth",
        feature_type=FeatureType.ENGAGEMENT,
        description="Number of replies in thread",
        normalization=NormalizationMethod.LOG,  # Log scale for counts
        default_value=0.0,
        importance=0.65,
        enabled=True,
    ),
    FeatureDefinition(
        name="participant_count",
        feature_type=FeatureType.ENGAGEMENT,
        description="Unique participants in discussion",
        normalization=NormalizationMethod.LOG,
        default_value=0.0,
        importance=0.6,
        enabled=True,
    ),
    FeatureDefinition(
        name="reaction_count",
        feature_type=FeatureType.ENGAGEMENT,
        description="Total reactions received",
        normalization=NormalizationMethod.LOG,
        default_value=0.0,
        importance=0.55,
        enabled=True,
    ),
    FeatureDefinition(
        name="reaction_diversity",
        feature_type=FeatureType.ENGAGEMENT,
        description="Unique reaction types (indicates diverse response)",
        normalization=NormalizationMethod.MINMAX,
        default_value=0.0,
        importance=0.45,
        enabled=True,
    ),
    FeatureDefinition(
        name="cross_team_engagement",
        feature_type=FeatureType.ENGAGEMENT,
        description="Multiple teams involved in discussion",
        normalization=NormalizationMethod.MINMAX,
        default_value=0.0,
        importance=0.5,
        enabled=True,
    ),
    FeatureDefinition(
        name="view_count",
        feature_type=FeatureType.ENGAGEMENT,
        description="Message views (if available)",
        normalization=NormalizationMethod.LOG,
        default_value=0.0,
        importance=0.4,
        enabled=False,  # Not always available
    ),

    # =========================================================================
    # SOURCE AUTHORITY FEATURES (4)
    # =========================================================================
    FeatureDefinition(
        name="author_seniority",
        feature_type=FeatureType.AUTHORITY,
        description="Author's organizational seniority/tenure",
        normalization=NormalizationMethod.MINMAX,
        default_value=0.5,  # Neutral for unknown authors
        importance=0.5,
        enabled=True,
    ),
    FeatureDefinition(
        name="channel_importance",
        feature_type=FeatureType.AUTHORITY,
        description="Channel's activity level and membership",
        normalization=NormalizationMethod.MINMAX,
        default_value=0.5,
        importance=0.55,
        enabled=True,
    ),
    FeatureDefinition(
        name="team_influence",
        feature_type=FeatureType.AUTHORITY,
        description="Team's organizational importance",
        normalization=NormalizationMethod.MINMAX,
        default_value=0.5,
        importance=0.45,
        enabled=True,
    ),
    FeatureDefinition(
        name="domain_expertise",
        feature_type=FeatureType.AUTHORITY,
        description="Author's expertise in topic area",
        normalization=NormalizationMethod.MINMAX,
        default_value=0.5,
        importance=0.6,
        enabled=True,
    ),

    # =========================================================================
    # CONTENT FEATURES (3)
    # =========================================================================
    FeatureDefinition(
        name="message_length",
        feature_type=FeatureType.CONTENT,
        description="Character count (normalized)",
        normalization=NormalizationMethod.LOG,
        default_value=0.0,
        importance=0.3,
        enabled=True,
    ),
    FeatureDefinition(
        name="code_snippet_present",
        feature_type=FeatureType.CONTENT,
        description="Contains code blocks (binary)",
        normalization=NormalizationMethod.NONE,  # Binary
        default_value=0.0,
        importance=0.5,
        enabled=True,
    ),
    FeatureDefinition(
        name="link_count",
        feature_type=FeatureType.CONTENT,
        description="Number of external references",
        normalization=NormalizationMethod.LOG,
        default_value=0.0,
        importance=0.4,
        enabled=True,
    ),
]


@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction.

    Attributes:
        enabled_features: List of feature names to extract
        feature_definitions: Mapping of feature names to definitions
        normalization_stats: Optional pre-computed normalization statistics
    """
    enabled_features: List[str] = field(default_factory=list)
    feature_definitions: Dict[str, FeatureDefinition] = field(default_factory=dict)
    normalization_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with default features if not specified."""
        if not self.feature_definitions:
            self.feature_definitions = {
                fd.name: fd for fd in FEATURE_DEFINITIONS
            }

        if not self.enabled_features:
            self.enabled_features = [
                name for name, fd in self.feature_definitions.items()
                if fd.enabled
            ]

    def get_feature_definition(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        return self.feature_definitions.get(name)

    def is_feature_enabled(self, name: str) -> bool:
        """Check if a feature is enabled."""
        return name in self.enabled_features

    def get_features_by_type(self, feature_type: FeatureType) -> List[str]:
        """Get all enabled features of a specific type."""
        return [
            name for name in self.enabled_features
            if self.feature_definitions[name].feature_type == feature_type
        ]

    def get_feature_importance(self, name: str) -> float:
        """Get feature importance weight."""
        fd = self.get_feature_definition(name)
        return fd.importance if fd else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "enabled_features": self.enabled_features,
            "feature_count": len(self.enabled_features),
            "features_by_type": {
                ft.value: self.get_features_by_type(ft)
                for ft in FeatureType
            }
        }


def get_default_config() -> FeatureConfig:
    """Get default feature configuration with all enabled features."""
    return FeatureConfig()


def get_minimal_config() -> FeatureConfig:
    """Get minimal configuration with only high-importance features."""
    high_importance_features = [
        fd.name for fd in FEATURE_DEFINITIONS
        if fd.enabled and fd.importance >= 0.7
    ]

    return FeatureConfig(enabled_features=high_importance_features)


def get_features_by_category() -> Dict[str, List[str]]:
    """Get features organized by category."""
    result = {}
    for ft in FeatureType:
        result[ft.value] = [
            fd.name for fd in FEATURE_DEFINITIONS
            if fd.feature_type == ft and fd.enabled
        ]
    return result
