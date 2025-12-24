"""
Analysis module for entity extraction, topic modeling, and pattern detection.
"""

from src.analysis.entity_extraction import EntityExtractor
from src.analysis.entity_types import (
    Entity,
    EntityType,
    ExtractedEntities,
    Person,
    Project,
    Team,
    Topic,
)
from src.analysis.similarity import (
    average_similarity,
    compute_cluster_cohesion,
    compute_cluster_separation,
    cosine_distance,
    cosine_similarity,
    max_similarity,
    pairwise_distance,
    pairwise_similarity,
)

__all__ = [
    "EntityExtractor",
    "Entity",
    "EntityType",
    "ExtractedEntities",
    "Person",
    "Team",
    "Project",
    "Topic",
    "cosine_similarity",
    "cosine_distance",
    "pairwise_similarity",
    "pairwise_distance",
    "average_similarity",
    "max_similarity",
    "compute_cluster_cohesion",
    "compute_cluster_separation",
]
