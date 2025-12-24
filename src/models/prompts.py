"""
Prompt templates for Claude API interactions.

This module contains structured prompts for gap verification, insight generation,
and other LLM-powered reasoning tasks.
"""

# Gap Verification Prompt Template
GAP_VERIFICATION_PROMPT = """You are analyzing organizational communication to detect coordination gaps.

Given these clustered messages about "{topic}", determine if they represent duplicate work:

Messages:
{message_list}

Teams involved: {teams}
Timeframe: {timeframe}

Is this duplicate work? Consider:
1. Are multiple teams solving the same problem?
2. Is there temporal overlap (working simultaneously)?
3. Are they aware of each other's work?
4. Is there actual duplication of effort?

Respond with JSON:
{{
  "is_duplicate": boolean,
  "confidence": 0-1,
  "reasoning": "explanation",
  "evidence": ["key quotes"],
  "recommendation": "action to take",
  "overlap_ratio": 0-1
}}"""

# Insight Generation Prompt Template
INSIGHT_GENERATION_PROMPT = """You are analyzing a detected coordination gap.

Gap Details:
Type: {gap_type}
Teams: {teams}
Topic: {topic}
Evidence: {evidence}

Generate actionable insights for this coordination gap:

1. What went wrong?
2. Why did this happen?
3. What is the impact?
4. What should be done immediately?

Respond with JSON:
{{
  "summary": "brief summary",
  "root_cause": "why this happened",
  "impact": "organizational impact",
  "immediate_actions": ["action 1", "action 2"],
  "preventive_measures": ["measure 1", "measure 2"]
}}"""

# Missing Context Detection Prompt Template
MISSING_CONTEXT_PROMPT = """You are analyzing a decision-making discussion for missing stakeholders.

Decision Discussion:
{discussion_content}

Required Stakeholders (based on org structure): {required_stakeholders}
Actual Participants: {actual_participants}

Determine if critical stakeholders are missing:

1. Is this a significant decision?
2. Who should have been involved?
3. What context are they missing?
4. What is the risk?

Respond with JSON:
{{
  "is_missing_context": boolean,
  "confidence": 0-1,
  "missing_stakeholders": ["person 1", "person 2"],
  "reasoning": "explanation",
  "risk_level": "low|medium|high|critical",
  "recommendation": "action to take"
}}"""

# Stale Documentation Detection Prompt Template
STALE_DOCS_PROMPT = """You are analyzing documentation for staleness against current implementation.

Documentation:
{doc_content}
Last Updated: {doc_last_updated}

Current Implementation Evidence:
{implementation_evidence}
Recent Changes: {recent_changes}

Determine if documentation is stale:

1. Does the documentation match current implementation?
2. What specific details are outdated?
3. How critical are the discrepancies?
4. What should be updated?

Respond with JSON:
{{
  "is_stale": boolean,
  "confidence": 0-1,
  "stale_sections": ["section 1", "section 2"],
  "discrepancies": ["what's wrong 1", "what's wrong 2"],
  "criticality": "low|medium|high|critical",
  "update_recommendation": "what to update"
}}"""

# Entity Extraction Enhancement Prompt Template
ENTITY_EXTRACTION_PROMPT = """Extract key entities from this message:

Message: {message_content}
Channel: {channel}
Author: {author}

Extract:
1. People mentioned (names, @mentions, emails)
2. Teams mentioned (@team mentions, team names)
3. Projects/features discussed
4. Technical terms and concepts

Respond with JSON:
{{
  "people": ["person 1", "person 2"],
  "teams": ["team 1", "team 2"],
  "projects": ["project 1", "project 2"],
  "technical_terms": ["term 1", "term 2"],
  "topics": ["topic 1", "topic 2"]
}}"""

# Cluster Labeling Prompt Template
CLUSTER_LABELING_PROMPT = """Generate a concise label for this cluster of related messages:

Messages:
{message_summary}

Number of messages: {message_count}
Timespan: {timespan}
Participants: {participants}

Generate:
1. A short label (3-5 words) that captures the main topic
2. A one-sentence summary
3. Key themes (2-4 words each)

Respond with JSON:
{{
  "label": "short label",
  "summary": "one sentence summary",
  "themes": ["theme 1", "theme 2", "theme 3"]
}}"""

# Recommendation Generation Prompt Template
RECOMMENDATION_PROMPT = """Generate actionable recommendations for addressing this coordination gap.

Gap Details:
Type: {gap_type}
Teams: {teams}
Topic: {topic}
Impact Score: {impact_score}
Evidence: {evidence}

Generate specific, actionable recommendations:
1. Immediate actions (next 24 hours)
2. Short-term fixes (next week)
3. Long-term improvements (prevent recurrence)

Respond with JSON:
{{
  "immediate_actions": [
    {{"action": "what to do", "owner": "who should do it", "urgency": "critical|high|medium"}}
  ],
  "short_term_fixes": [
    {{"action": "what to do", "timeline": "when", "expected_impact": "outcome"}}
  ],
  "long_term_improvements": [
    {{"action": "what to do", "rationale": "why", "effort": "low|medium|high"}}
  ],
  "talking_points": ["point 1 for discussion", "point 2"]
}}"""


def format_message_list(messages: list[dict]) -> str:
    """Format a list of messages for prompt inclusion."""
    formatted = []
    for i, msg in enumerate(messages, 1):
        formatted.append(
            f"{i}. [{msg.get('timestamp', 'unknown')}] "
            f"{msg.get('author', 'unknown')} in {msg.get('channel', 'unknown')}:\n"
            f"   {msg.get('content', '')}"
        )
    return "\n\n".join(formatted)


def format_evidence(evidence: list[dict]) -> str:
    """Format evidence for prompt inclusion."""
    formatted = []
    for i, item in enumerate(evidence, 1):
        formatted.append(
            f"{i}. {item.get('source', 'unknown')} - "
            f"{item.get('content', '')[:200]}..."
        )
    return "\n".join(formatted)


def format_teams(teams: list[str]) -> str:
    """Format team list for prompt inclusion."""
    return ", ".join(teams) if teams else "unknown"


def format_timeframe(start_date: str, end_date: str, overlap_days: int = None) -> str:
    """Format timeframe for prompt inclusion."""
    base = f"{start_date} to {end_date}"
    if overlap_days:
        base += f" ({overlap_days} days overlap)"
    return base
