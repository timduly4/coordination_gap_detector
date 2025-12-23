"""
Entity type definitions for extraction from messages.

Defines entity types (people, teams, projects, topics) and their
data structures for use in gap detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EntityType(str, Enum):
    """Types of entities that can be extracted from messages."""

    PERSON = "person"
    TEAM = "team"
    PROJECT = "project"
    TOPIC = "topic"


@dataclass
class Entity:
    """
    Base entity class with common fields.

    Attributes:
        text: Original text of the entity
        normalized: Normalized/canonical form
        type: Type of entity
        confidence: Confidence score [0, 1]
        metadata: Additional metadata about extraction
    """

    text: str
    normalized: str
    type: EntityType
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class Person(Entity):
    """
    Person/author entity.

    Extracted from:
    - @mentions: @alice, @bob
    - Email addresses: alice@company.com
    - Full names: Alice Johnson

    Normalized to canonical email format.
    """

    type: EntityType = field(default=EntityType.PERSON, init=False)

    @property
    def email(self) -> Optional[str]:
        """Get email address if available."""
        if "@" in self.normalized:
            return self.normalized
        return self.metadata.get("email")

    @property
    def username(self) -> Optional[str]:
        """Get username without @ prefix."""
        if self.normalized.startswith("@"):
            return self.normalized[1:]
        if "@" in self.normalized:
            return self.normalized.split("@")[0]
        return self.normalized


@dataclass
class Team(Entity):
    """
    Team entity.

    Extracted from:
    - Team mentions: @platform-team, @auth-team
    - Channel names: #platform → platform team
    - Department names: engineering, product

    Normalized to lowercase with hyphens.
    """

    type: EntityType = field(default=EntityType.TEAM, init=False)

    @property
    def team_name(self) -> str:
        """Get clean team name."""
        name = self.normalized
        # Remove @ prefix and -team suffix if present
        if name.startswith("@"):
            name = name[1:]
        if name.endswith("-team"):
            return name[:-5]
        return name


@dataclass
class Project(Entity):
    """
    Project/feature entity.

    Extracted from:
    - Feature names: OAuth, authentication, API gateway
    - Project codes: PROJ-123, EPIC-456
    - Technical terms: microservices, Kubernetes
    - Acronyms: SSO, RBAC, JWT

    Normalized to lowercase.
    """

    type: EntityType = field(default=EntityType.PROJECT, init=False)

    @property
    def is_acronym(self) -> bool:
        """Check if this is an acronym (all caps)."""
        return self.text.isupper() and len(self.text) <= 6


@dataclass
class Topic(Entity):
    """
    Discussion topic entity.

    Extracted from:
    - Keyword extraction from content
    - Technical stack mentions
    - Problem/solution identification

    Normalized to lowercase.
    """

    type: EntityType = field(default=EntityType.TOPIC, init=False)


@dataclass
class ExtractedEntities:
    """
    Collection of all entities extracted from a message.

    Attributes:
        people: List of person entities
        teams: List of team entities
        projects: List of project entities
        topics: List of topic entities
        message_id: Optional message ID for tracking
    """

    people: list[Person] = field(default_factory=list)
    teams: list[Team] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    topics: list[Topic] = field(default_factory=list)
    message_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "people": [p.normalized for p in self.people],
            "teams": [t.normalized for t in self.teams],
            "projects": [p.normalized for p in self.projects],
            "topics": [t.normalized for t in self.topics],
            "message_id": self.message_id,
        }

    def all_entities(self) -> list[Entity]:
        """Get all entities as a single list."""
        return self.people + self.teams + self.projects + self.topics

    def count(self) -> int:
        """Get total count of entities."""
        return len(self.people) + len(self.teams) + len(self.projects) + len(self.topics)

    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return self.count() > 0
