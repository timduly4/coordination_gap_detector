# Entity Extraction Guide

This guide explains how the system extracts people, teams, projects, and topics from organizational communication for gap detection.

## Table of Contents

- [Overview](#overview)
- [Entity Types](#entity-types)
- [Extraction Methods](#extraction-methods)
- [Entity Normalization](#entity-normalization)
- [Confidence Scoring](#confidence-scoring)
- [Performance Optimization](#performance-optimization)
- [Testing Entity Extraction](#testing-entity-extraction)
- [Advanced Topics](#advanced-topics)

---

## Overview

### What is Entity Extraction?

Entity extraction identifies key organizational entities from unstructured text:

- **People**: @mentions, email addresses, full names
- **Teams**: Team mentions, channel-based inference, department names
- **Projects**: Feature names, project codes, technical terms
- **Topics**: Discussion topics, technical stack, problem/solution identification

### Why Entity Extraction Matters

Accurate entity extraction is critical for gap detection:

1. **Team Identification**: Know which teams are involved in work
2. **Scope Determination**: Understand what's being worked on
3. **Overlap Detection**: Find multiple teams working on same projects
4. **Evidence Collection**: Attribute messages to specific teams/people

### Extraction Pipeline

```
Raw Message Text
      ↓
Pattern Matching (Regex)
      ↓
Context Analysis
      ↓
Normalization
      ↓
Confidence Scoring
      ↓
Structured Entities
```

---

## Entity Types

### 1. People / Authors

**Purpose**: Identify individuals involved in discussions

**Extraction Patterns**:

```python
# @mentions (Slack, GitHub, etc.)
@alice → "alice@company.com"
@bob.smith → "bob.smith@company.com"

# Email addresses
alice@company.com → "alice@company.com"

# Full names in text
"Alice Johnson said..." → "alice.johnson@company.com" (if mappable)

# Author field
{"author": "alice@company.com"} → "alice@company.com"
```

**Normalization**:
- All mentions normalized to canonical email
- Handle variations: @alice, alice@company.com, Alice Johnson
- Maintain alias mapping: alice ↔ alice@company.com

**Example**:
```python
from src.analysis.entity_extraction import EntityExtractor

extractor = EntityExtractor()
message = {
    "content": "@alice and @bob are working on OAuth",
    "author": "charlie@company.com"
}

people = extractor.extract_people(message)
# Result: ["alice@company.com", "bob@company.com", "charlie@company.com"]
```

### 2. Teams

**Purpose**: Identify which organizational teams are involved

**Extraction Methods**:

#### A. Direct Team Mentions
```python
# @team mentions
@platform-team → "platform-team"
@auth-team → "auth-team"
@security → "security-team"

# Hashtag teams
#platform → "platform-team"
#backend-eng → "backend-team"
```

#### B. Channel-Based Inference
```python
# Infer team from channel name
Channel: #platform → Team: "platform-team"
Channel: #auth-team → Team: "auth-team"
Channel: #backend-eng → Team: "backend-team"
Channel: #engineering → Team: "engineering" (general)
```

#### C. Metadata-Based
```python
# Team in message metadata
{
    "metadata": {
        "team": "platform-team"
    }
}
```

**Team Name Variations**:
```python
# These all normalize to "platform-team"
@platform-team
@platform_team
#platform
platform team
Platform Team
PlatformTeam
```

**Example**:
```python
message = {
    "content": "The @platform-team is collaborating with @auth-team on OAuth",
    "channel": "#platform",
    "metadata": {"team": "platform-team"}
}

teams = extractor.extract_teams(message)
# Result: ["platform-team", "auth-team"]
# (platform-team appears once even though mentioned in content and metadata)
```

### 3. Projects / Features

**Purpose**: Identify what's being worked on

**Extraction Patterns**:

#### A. Feature Names
```python
# Capitalized technical terms
OAuth, OAuth2, OpenID → "OAuth"
API Gateway, API gateway → "API Gateway"
Kubernetes, K8s → "Kubernetes"
```

#### B. Project Codes
```python
# Jira/ticket patterns
PROJ-123 → "PROJ-123"
EPIC-456 → "EPIC-456"
TICKET-789 → "TICKET-789"
```

#### C. Technical Terms
```python
# Common technical keywords
authentication, auth → "authentication"
microservices → "microservices"
database, DB → "database"
```

#### D. Acronyms
```python
# Technical acronyms
SSO → "SSO" (Single Sign-On)
RBAC → "RBAC" (Role-Based Access Control)
JWT → "JWT" (JSON Web Token)
API → "API" (Application Programming Interface)
```

**Example**:
```python
message = {
    "content": "Working on OAuth2 implementation for the API Gateway. " \
               "Using JWT tokens with RBAC for security. Tracking in PROJ-123."
}

projects = extractor.extract_projects(message)
# Result: ["OAuth2", "API Gateway", "JWT", "RBAC", "PROJ-123"]
```

### 4. Topics

**Purpose**: Understand discussion themes

**Extraction Methods**:

#### A. Keyword Extraction
```python
# Technical action keywords
implementation, integration, migration, refactoring
design, architecture, deployment, testing
```

#### B. Problem/Solution Pairs
```python
# Problem indicators
"issue with...", "bug in...", "error when..."

# Solution indicators
"fixed by...", "resolved with...", "workaround..."
```

#### C. Technology Stack
```python
# Languages, frameworks, tools
Python, React, Docker, Kubernetes, PostgreSQL
```

**Example**:
```python
message = {
    "content": "Implementation of OAuth2 authentication using Python and PostgreSQL. " \
               "Encountered issue with token expiration, fixed by adding refresh tokens."
}

topics = extractor.extract_topics(message)
# Result: {
#   "actions": ["implementation", "authentication"],
#   "technologies": ["OAuth2", "Python", "PostgreSQL"],
#   "problems": ["token expiration"],
#   "solutions": ["refresh tokens"]
# }
```

---

## Extraction Methods

### Pattern-Based (Regex) - Primary Approach

**Advantages**:
- ✅ Fast (<20ms per message)
- ✅ Deterministic and predictable
- ✅ No model downloads required
- ✅ Works well for structured mentions (@, #, email)
- ✅ Sufficient for Milestone 3 goals

**Best For**:
- @mentions and #channels
- Email addresses
- URLs and ticket IDs
- Known patterns (OAuth, JWT, etc.)

**Implementation**:
```python
import re

class RegexEntityExtractor:
    """Pattern-based entity extraction using regex."""

    # Mention pattern: @username or @team-name
    MENTION_PATTERN = r'@([a-zA-Z0-9._-]+)'

    # Email pattern
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # Team pattern: words ending in -team or #channel
    TEAM_PATTERN = r'(?:@|#)?([a-z-]+(?:-team|team))\b'

    # Project code pattern: PREFIX-123
    PROJECT_PATTERN = r'\b([A-Z]{2,}-\d+)\b'

    def extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        return re.findall(self.MENTION_PATTERN, text)

    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        return re.findall(self.EMAIL_PATTERN, text)

    def extract_teams(self, text: str) -> List[str]:
        """Extract team names from text."""
        return re.findall(self.TEAM_PATTERN, text.lower())

    def extract_project_codes(self, text: str) -> List[str]:
        """Extract project codes (JIRA, etc.)."""
        return re.findall(self.PROJECT_PATTERN, text)
```

### NLP-Based (Future Enhancement)

**Advantages**:
- Better context understanding
- Named Entity Recognition (NER)
- Handles unstructured text better
- Can identify entities without explicit patterns

**Disadvantages**:
- Slower (100-200ms per message)
- Requires model downloads (100MB+)
- More complex dependencies
- Less deterministic

**When to Add**:
- Pattern-based extraction insufficient
- Need to handle complex, unstructured text
- Willing to accept performance trade-off
- Have NER training data

**Example (Future)**:
```python
import spacy

class NLPEntityExtractor:
    """NLP-based entity extraction using spaCy."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using NER."""
        doc = self.nlp(text)

        return {
            "people": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
            "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
            "products": [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"],
        }
```

### Hybrid Approach (Recommended)

Combine both methods for best results:

```python
class HybridEntityExtractor:
    """Hybrid extractor combining regex and NLP."""

    def __init__(self):
        self.regex_extractor = RegexEntityExtractor()
        self.nlp_extractor = None  # Optional

    def extract_people(self, message: dict) -> List[str]:
        """Extract people using multiple methods."""
        people = set()

        # 1. Regex-based (fast, reliable)
        people.update(self.regex_extractor.extract_mentions(message["content"]))
        people.update(self.regex_extractor.extract_emails(message["content"]))

        # 2. Author field
        if message.get("author"):
            people.add(message["author"])

        # 3. Metadata mentions
        if message.get("metadata", {}).get("mentions"):
            people.update(message["metadata"]["mentions"])

        # 4. NLP-based (optional, if available)
        if self.nlp_extractor:
            nlp_people = self.nlp_extractor.extract_entities(message["content"])
            people.update(nlp_people.get("people", []))

        return list(people)
```

---

## Entity Normalization

### Why Normalize?

Different representations of the same entity need to be unified:

```
@alice
alice@company.com
Alice Johnson
alice.johnson@company.com

→ All normalize to: "alice@company.com"
```

### Normalization Strategies

#### 1. People Normalization

```python
class PeopleNormalizer:
    """Normalize people mentions to canonical email addresses."""

    def __init__(self):
        # Mapping: mention → email
        self.alias_map = {
            "alice": "alice@company.com",
            "bob": "bob@company.com",
            "charlie": "charlie@company.com",
        }

    def normalize(self, mention: str) -> str:
        """Normalize mention to canonical email."""
        # If already email, return as-is
        if "@" in mention:
            return mention.lower()

        # Look up in alias map
        normalized = self.alias_map.get(mention.lower())
        if normalized:
            return normalized

        # Default: assume @company.com
        return f"{mention.lower()}@company.com"
```

#### 2. Team Normalization

```python
class TeamNormalizer:
    """Normalize team names to canonical form."""

    def __init__(self):
        # Mapping: variation → canonical
        self.team_aliases = {
            "platform": "platform-team",
            "platform_team": "platform-team",
            "platformteam": "platform-team",
            "auth": "auth-team",
            "security": "security-team",
        }

    def normalize(self, team: str) -> str:
        """Normalize team name to canonical form."""
        # Remove @ and # prefixes
        clean = team.lstrip("@#").lower()

        # Look up alias
        normalized = self.team_aliases.get(clean)
        if normalized:
            return normalized

        # Ensure -team suffix
        if not clean.endswith("team"):
            clean += "-team"

        return clean
```

#### 3. Project Normalization

```python
class ProjectNormalizer:
    """Normalize project/feature names."""

    def __init__(self):
        # Synonyms: OAuth2 = OAuth = OpenAuth
        self.synonyms = {
            "oauth": "OAuth",
            "oauth2": "OAuth",
            "openauth": "OAuth",
            "k8s": "Kubernetes",
            "kubernetes": "Kubernetes",
        }

    def normalize(self, project: str) -> str:
        """Normalize project name."""
        # Check synonyms (case-insensitive)
        normalized = self.synonyms.get(project.lower())
        if normalized:
            return normalized

        # Title case for multi-word
        if " " in project:
            return project.title()

        # Keep capitalization for acronyms
        if project.isupper():
            return project

        # Default: capitalize first letter
        return project.capitalize()
```

---

## Confidence Scoring

### Why Score Confidence?

Not all extractions are equally certain:

```python
# High confidence (explicit mention)
"@platform-team is working on OAuth" → confidence: 0.95

# Medium confidence (inferred from channel)
Channel: #platform, Content: "working on OAuth" → confidence: 0.70

# Low confidence (ambiguous)
"the platform people" → confidence: 0.40
```

### Confidence Factors

```python
def calculate_extraction_confidence(
    entity: str,
    extraction_method: str,
    context: dict
) -> float:
    """Calculate confidence score for entity extraction."""

    confidence = 0.5  # Base confidence

    # Factor 1: Extraction method
    if extraction_method == "explicit_mention":
        confidence += 0.4  # @team-name
    elif extraction_method == "email_address":
        confidence += 0.4  # email@company.com
    elif extraction_method == "channel_inference":
        confidence += 0.2  # inferred from #channel
    elif extraction_method == "nlp":
        confidence += 0.1  # NLP extraction

    # Factor 2: Multiple confirmations
    if context.get("confirmed_by_multiple_methods"):
        confidence += 0.1

    # Factor 3: Metadata confirmation
    if context.get("in_metadata"):
        confidence += 0.1

    # Factor 4: Pattern match strength
    if context.get("exact_pattern_match"):
        confidence += 0.1

    return min(1.0, confidence)
```

### Confidence Thresholds

```python
# Only use high-confidence extractions
if confidence >= 0.7:
    # Use entity
    entities.append(entity)
else:
    # Discard or flag for review
    uncertain_entities.append((entity, confidence))
```

---

## Performance Optimization

### Target Performance

- **Entity extraction**: <20ms per message (p95)
- **Batch processing**: <100ms for 100 messages
- **Total overhead**: <5% of detection pipeline time

### Optimization Techniques

#### 1. Compile Regex Patterns Once

```python
class OptimizedExtractor:
    """Performance-optimized entity extractor."""

    def __init__(self):
        # Compile patterns once at initialization
        self.mention_re = re.compile(r'@([a-zA-Z0-9._-]+)')
        self.email_re = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.team_re = re.compile(r'(?:@|#)?([a-z-]+(?:-team|team))\b')

    def extract(self, text: str):
        # Use pre-compiled patterns (faster)
        mentions = self.mention_re.findall(text)
        emails = self.email_re.findall(text)
        teams = self.team_re.findall(text.lower())
        return mentions, emails, teams
```

#### 2. Cache Normalization Lookups

```python
from functools import lru_cache

class CachedNormalizer:
    """Normalizer with caching for repeated lookups."""

    @lru_cache(maxsize=1000)
    def normalize_person(self, mention: str) -> str:
        """Cached person normalization."""
        # Expensive normalization only happens once per unique mention
        return self._normalize_impl(mention)
```

#### 3. Batch Processing

```python
def extract_entities_batch(messages: List[dict]) -> List[dict]:
    """Extract entities from multiple messages efficiently."""
    extractor = EntityExtractor()

    results = []
    for msg in messages:
        # Extract entities
        entities = extractor.extract(msg)
        results.append(entities)

    return results

# Process 100 messages in one call
entities_list = extract_entities_batch(messages)
```

#### 4. Lazy NLP Loading

```python
class LazyNLPExtractor:
    """Only load NLP model if needed."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract(self, text: str):
        # NLP model only loaded on first use
        return self.nlp(text)
```

---

## Testing Entity Extraction

### Unit Tests

```python
import pytest
from src.analysis.entity_extraction import EntityExtractor

class TestEntityExtraction:
    """Unit tests for entity extraction."""

    def test_extract_mentions(self):
        """Test @mention extraction."""
        extractor = EntityExtractor()
        message = {"content": "@alice and @bob are working on this"}

        people = extractor.extract_people(message)

        assert "alice@company.com" in people
        assert "bob@company.com" in people

    def test_extract_teams(self):
        """Test team extraction."""
        extractor = EntityExtractor()
        message = {
            "content": "@platform-team collaborating with @auth-team",
            "channel": "#platform"
        }

        teams = extractor.extract_teams(message)

        assert "platform-team" in teams
        assert "auth-team" in teams

    def test_normalization(self):
        """Test entity normalization."""
        extractor = EntityExtractor()

        # Different representations → same canonical form
        assert extractor.normalize_person("alice") == "alice@company.com"
        assert extractor.normalize_person("alice@company.com") == "alice@company.com"
        assert extractor.normalize_person("@alice") == "alice@company.com"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_extraction_with_real_messages():
    """Test extraction with realistic message data."""
    from src.ingestion.slack.mock_client import MockSlackClient

    # Load realistic scenario
    client = MockSlackClient()
    messages = client.get_scenario_messages("oauth_duplication")

    extractor = EntityExtractor()

    # Extract from all messages
    all_teams = set()
    all_people = set()

    for msg in messages:
        entities = extractor.extract({
            "content": msg.content,
            "author": msg.author,
            "channel": msg.channel,
            "metadata": msg.metadata
        })

        all_teams.update(entities.get("teams", []))
        all_people.update(entities.get("people", []))

    # Verify expected entities found
    assert "platform-team" in all_teams
    assert "auth-team" in all_teams
    assert len(all_people) >= 4  # Multiple participants
```

### Performance Tests

```python
import time

def test_extraction_performance():
    """Test that extraction meets performance targets."""
    extractor = EntityExtractor()

    # Create test messages
    messages = [
        {"content": f"@user{i} working on project{i}"}
        for i in range(100)
    ]

    # Measure extraction time
    start = time.time()
    for msg in messages:
        extractor.extract(msg)
    elapsed = (time.time() - start) * 1000  # ms

    # Should be <100ms for 100 messages
    assert elapsed < 100, f"Took {elapsed}ms, expected <100ms"

    # Per-message average should be <1ms
    per_message = elapsed / 100
    assert per_message < 1.0, f"Took {per_message}ms per message"
```

---

## Advanced Topics

### Context-Aware Extraction

Use surrounding context to improve extraction:

```python
def extract_with_context(message: dict, thread: List[dict]) -> dict:
    """Extract entities using thread context."""

    # Extract from current message
    entities = extractor.extract(message)

    # Use thread context for disambiguation
    thread_teams = set()
    for msg in thread:
        thread_teams.update(extractor.extract_teams(msg))

    # If current message has ambiguous team reference,
    # use most common team from thread
    if not entities.get("teams") and thread_teams:
        entities["teams"] = [most_common(thread_teams)]

    return entities
```

### Custom Entity Types

Add domain-specific entity types:

```python
class CustomEntityExtractor(EntityExtractor):
    """Extractor with custom entity types."""

    def extract_metrics(self, text: str) -> List[dict]:
        """Extract metrics mentions (latency, throughput, etc.)."""
        metrics = []

        # Latency mentions
        latency_pattern = r'(\d+(?:\.\d+)?)\s*(ms|millisecond|second)s?\s+latency'
        for match in re.finditer(latency_pattern, text.lower()):
            metrics.append({
                "type": "latency",
                "value": float(match.group(1)),
                "unit": match.group(2)
            })

        return metrics

    def extract_costs(self, text: str) -> List[dict]:
        """Extract cost mentions ($X, Xk/month, etc.)."""
        cost_pattern = r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
        # ... implementation
```

### Multi-Language Support

Support entity extraction in multiple languages:

```python
class MultiLanguageExtractor:
    """Entity extractor supporting multiple languages."""

    def __init__(self, language="en"):
        self.language = language
        self.patterns = self._load_patterns(language)

    def _load_patterns(self, lang: str) -> dict:
        """Load language-specific patterns."""
        patterns = {
            "en": {
                "mention": r'@([a-zA-Z0-9._-]+)',
                "team": r'(?:team|group)\s+([a-z-]+)',
            },
            "es": {
                "mention": r'@([a-zA-Z0-9._-]+)',
                "team": r'(?:equipo|grupo)\s+([a-z-]+)',
            },
            # ... more languages
        }
        return patterns.get(lang, patterns["en"])
```

---

## Summary

**Key Takeaways**:

1. **Primary Method**: Pattern-based (regex) extraction is fast and sufficient
2. **Entity Types**: People, teams, projects, topics
3. **Normalization**: Critical for matching variations
4. **Performance**: <20ms per message is achievable
5. **Future**: Can add NLP if needed

**Best Practices**:
- Start with pattern-based extraction
- Normalize all entities to canonical forms
- Use confidence scoring to filter uncertain extractions
- Test with realistic scenarios
- Optimize for batch processing

**Next Steps**:
- [Gap Detection Methodology](GAP_DETECTION.md)
- [API Usage Examples](API_EXAMPLES.md)

---

**Last Updated**: December 2024
**Version**: 1.0
**Milestone**: 3H - Testing & Documentation
