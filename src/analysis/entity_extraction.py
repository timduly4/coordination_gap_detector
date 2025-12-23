"""
Entity extraction for coordination gap detection.

This module implements regex-based entity extraction to identify:
- People: @mentions, emails, names
- Teams: team mentions, channel-based inference
- Projects: feature names, project codes, technical terms
- Topics: keywords and discussion themes

Uses lightweight pattern matching for speed and determinism.
Future enhancement: Add spaCy/NLP for more sophisticated extraction.
"""

import logging
import re
from typing import Any, Optional

from src.analysis.entity_types import (
    ExtractedEntities,
    Person,
    Project,
    Team,
    Topic,
)

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extract entities from messages using regex patterns.

    This extractor uses lightweight pattern matching rather than
    heavy NLP models for speed and reliability. It handles:
    - Slack-style @mentions and #channels
    - Email addresses
    - Team name patterns
    - Technical terms and acronyms
    - Project codes (PROJ-123, EPIC-456)
    """

    # Regex patterns for entity extraction
    # MENTION_PATTERN: Matches @username but not @domain in emails
    # Uses negative lookbehind to avoid matching @domain.com from email@domain.com
    MENTION_PATTERN = re.compile(r"(?<![a-zA-Z0-9._-])@([a-zA-Z0-9_-]+)(?![a-zA-Z0-9._-])")
    EMAIL_PATTERN = re.compile(r"\b([a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b")
    CHANNEL_PATTERN = re.compile(r"#([a-zA-Z0-9_-]+)")
    TEAM_MENTION_PATTERN = re.compile(r"(?<![a-zA-Z0-9._-])@([a-zA-Z0-9_-]*team[a-zA-Z0-9_-]*)", re.IGNORECASE)
    PROJECT_CODE_PATTERN = re.compile(r"\b([A-Z]{2,10}-\d+)\b")
    ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,6})\b")

    # Common technical terms (can be extended)
    TECHNICAL_TERMS = {
        "oauth", "oauth2", "api", "rest", "graphql", "microservices",
        "kubernetes", "docker", "redis", "postgres", "elasticsearch",
        "authentication", "authorization", "jwt", "sso", "saml", "rbac",
        "ci/cd", "devops", "frontend", "backend", "database",
        "security", "encryption", "migration", "refactor", "deployment",
    }

    # Department/team keywords
    DEPARTMENT_KEYWORDS = {
        "engineering", "product", "design", "platform", "backend",
        "frontend", "mobile", "data", "security", "infrastructure",
        "devops", "qa", "testing", "operations", "marketing",
    }

    def __init__(self, domain: str = "company.com"):
        """
        Initialize entity extractor.

        Args:
            domain: Default email domain for normalizing @mentions
        """
        self.domain = domain
        logger.info(f"EntityExtractor initialized with domain: {domain}")

    def extract(
        self,
        message: dict[str, Any],
        extract_people: bool = True,
        extract_teams: bool = True,
        extract_projects: bool = True,
        extract_topics: bool = True,
    ) -> ExtractedEntities:
        """
        Extract all entities from a message.

        Args:
            message: Message dict with 'content', 'channel', 'author' fields
            extract_people: Extract person entities
            extract_teams: Extract team entities
            extract_projects: Extract project/feature entities
            extract_topics: Extract topic entities

        Returns:
            ExtractedEntities with all extracted entities
        """
        content = message.get("content", "")
        channel = message.get("channel", "")
        author = message.get("author", "")
        message_id = message.get("id") or message.get("message_id")

        entities = ExtractedEntities(message_id=message_id)

        if extract_people:
            entities.people = self.extract_people(content, author)

        if extract_teams:
            entities.teams = self.extract_teams(content, channel)

        if extract_projects:
            entities.projects = self.extract_projects(content)

        if extract_topics:
            entities.topics = self.extract_topics(content)

        logger.debug(
            f"Extracted {entities.count()} entities from message {message_id}: "
            f"{len(entities.people)} people, {len(entities.teams)} teams, "
            f"{len(entities.projects)} projects, {len(entities.topics)} topics"
        )

        return entities

    def extract_people(self, content: str, author: Optional[str] = None) -> list[Person]:
        """
        Extract person entities from content.

        Identifies:
        - @mentions: @alice, @bob
        - Email addresses: alice@company.com
        - Author attribution

        Args:
            content: Message content
            author: Message author (if available)

        Returns:
            List of Person entities (deduplicated by normalized form)
        """
        people = []
        seen = set()  # Track normalized forms to avoid duplicates

        # Extract email addresses
        for match in self.EMAIL_PATTERN.finditer(content):
            email = match.group(1).lower()
            if email not in seen:
                people.append(Person(
                    text=email,
                    normalized=email,
                    confidence=1.0,
                    metadata={"source": "email", "email": email}
                ))
                seen.add(email)

        # Extract @mentions
        for match in self.MENTION_PATTERN.finditer(content):
            username = match.group(1).lower()
            # Skip if it's a team mention (contains 'team')
            if "team" in username:
                continue

            mention = f"@{username}"
            # Normalize to email format
            normalized = f"{username}@{self.domain}"

            # Check normalized form to avoid duplicates
            if normalized not in seen:
                people.append(Person(
                    text=mention,
                    normalized=normalized,
                    confidence=0.9,  # Slightly lower since we're inferring email
                    metadata={"source": "mention", "username": username}
                ))
                seen.add(normalized)

        # Add author if provided
        if author:
            # Check if author is already an email
            if "@" in author:
                normalized = author.lower()
            else:
                normalized = f"{author}@{self.domain}"

            # Only add if not already seen
            if normalized not in seen:
                people.append(Person(
                    text=author,
                    normalized=normalized,
                    confidence=1.0,
                    metadata={"source": "author"}
                ))
                seen.add(normalized)

        return people

    def extract_teams(self, content: str, channel: Optional[str] = None) -> list[Team]:
        """
        Extract team entities from content and channel.

        Identifies:
        - Team mentions: @platform-team, @auth-team
        - Channel names: #platform → platform team
        - Department keywords: engineering, product

        Args:
            content: Message content
            channel: Channel name (if available)

        Returns:
            List of Team entities
        """
        teams = []
        seen = set()

        # Extract team mentions (@*team*)
        for match in self.TEAM_MENTION_PATTERN.finditer(content):
            team_name = match.group(1).lower()
            normalized = team_name.replace("_", "-")

            if normalized not in seen:
                teams.append(Team(
                    text=f"@{team_name}",
                    normalized=normalized,
                    confidence=1.0,
                    metadata={"source": "team_mention"}
                ))
                seen.add(normalized)

        # Infer team from channel name
        if channel:
            # Remove # prefix if present
            channel_clean = channel.lstrip("#").lower()

            # Check if channel name suggests a team
            # Skip generic channels like 'general', 'random'
            generic_channels = {"general", "random", "announcements", "social"}

            if channel_clean not in generic_channels:
                # Check if it matches department keywords
                for dept in self.DEPARTMENT_KEYWORDS:
                    if dept in channel_clean:
                        normalized = f"{dept}-team"
                        if normalized not in seen:
                            teams.append(Team(
                                text=channel,
                                normalized=normalized,
                                confidence=0.7,  # Lower confidence for inference
                                metadata={"source": "channel", "channel": channel_clean}
                            ))
                            seen.add(normalized)
                            break
                else:
                    # Use channel name directly as team
                    normalized = f"{channel_clean}-team"
                    if normalized not in seen:
                        teams.append(Team(
                            text=channel,
                            normalized=normalized,
                            confidence=0.6,  # Even lower for direct channel mapping
                            metadata={"source": "channel", "channel": channel_clean}
                        ))
                        seen.add(normalized)

        # Extract department keywords from content
        content_lower = content.lower()
        for dept in self.DEPARTMENT_KEYWORDS:
            # Look for "dept team" or just "dept"
            pattern = rf"\b{dept}(?:\s+team)?\b"
            if re.search(pattern, content_lower):
                normalized = f"{dept}-team"
                if normalized not in seen:
                    teams.append(Team(
                        text=dept,
                        normalized=normalized,
                        confidence=0.8,
                        metadata={"source": "department_keyword"}
                    ))
                    seen.add(normalized)

        return teams

    def extract_projects(self, content: str) -> list[Project]:
        """
        Extract project/feature entities from content.

        Identifies:
        - Project codes: PROJ-123, EPIC-456
        - Technical terms: OAuth, Kubernetes, API
        - Acronyms: SSO, RBAC, JWT

        Args:
            content: Message content

        Returns:
            List of Project entities
        """
        projects = []
        seen = set()

        # Extract project codes (JIRA-style)
        for match in self.PROJECT_CODE_PATTERN.finditer(content):
            code = match.group(1)
            normalized = code.lower()

            if normalized not in seen:
                projects.append(Project(
                    text=code,
                    normalized=normalized,
                    confidence=1.0,
                    metadata={"source": "project_code", "is_code": True}
                ))
                seen.add(normalized)

        # Extract technical terms
        content_lower = content.lower()
        for term in self.TECHNICAL_TERMS:
            # Use word boundary to avoid partial matches
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, content_lower):
                if term not in seen:
                    projects.append(Project(
                        text=term,
                        normalized=term.lower(),
                        confidence=0.9,
                        metadata={"source": "technical_term"}
                    ))
                    seen.add(term)

        # Extract acronyms (2-6 uppercase letters)
        for match in self.ACRONYM_PATTERN.finditer(content):
            acronym = match.group(1)
            # Skip common words that happen to be uppercase
            skip_words = {"I", "A", "AN", "THE", "OK", "PR", "MR"}
            if acronym in skip_words:
                continue

            normalized = acronym.lower()
            if normalized not in seen:
                # Lower confidence for acronyms as they could be anything
                projects.append(Project(
                    text=acronym,
                    normalized=normalized,
                    confidence=0.6,
                    metadata={"source": "acronym", "is_acronym": True}
                ))
                seen.add(normalized)

        return projects

    def extract_topics(self, content: str) -> list[Topic]:
        """
        Extract topic entities from content.

        Uses simple keyword extraction. Future enhancement:
        - TF-IDF for importance
        - Topic modeling
        - NLP-based key phrase extraction

        Args:
            content: Message content

        Returns:
            List of Topic entities
        """
        topics = []
        seen = set()

        # For now, use technical terms as topics
        content_lower = content.lower()
        for term in self.TECHNICAL_TERMS:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, content_lower):
                if term not in seen:
                    topics.append(Topic(
                        text=term,
                        normalized=term.lower(),
                        confidence=0.7,
                        metadata={"source": "keyword"}
                    ))
                    seen.add(term)

        # Extract common action verbs as topics (implementing, building, etc.)
        action_verbs = {
            "implementing", "building", "designing", "refactoring",
            "migrating", "deploying", "testing", "debugging",
            "optimizing", "scaling", "monitoring", "fixing",
        }

        for verb in action_verbs:
            pattern = rf"\b{verb}\b"
            if re.search(pattern, content_lower):
                if verb not in seen:
                    topics.append(Topic(
                        text=verb,
                        normalized=verb.lower(),
                        confidence=0.6,
                        metadata={"source": "action_verb"}
                    ))
                    seen.add(verb)

        return topics

    def normalize_entity(self, text: str, entity_type: str) -> str:
        """
        Normalize an entity to canonical form.

        Args:
            text: Original entity text
            entity_type: Type of entity (person, team, project, topic)

        Returns:
            Normalized entity string
        """
        text = text.lower().strip()

        if entity_type == "person":
            # Remove @ prefix
            text = text.lstrip("@")
            # Add domain if not an email
            if "@" not in text:
                text = f"{text}@{self.domain}"
            return text

        elif entity_type == "team":
            # Remove @ and # prefixes
            text = text.lstrip("@#")
            # Ensure -team suffix
            if not text.endswith("-team"):
                text = f"{text}-team"
            return text.replace("_", "-")

        elif entity_type in ("project", "topic"):
            # Simple lowercase normalization
            return text.lower()

        return text
