# Gap Detection Methodology

This document explains the coordination gap detection system, algorithms, and best practices for detecting and resolving organizational coordination failures.

## Table of Contents

- [Overview](#overview)
- [Detection Pipeline](#detection-pipeline)
- [Duplicate Work Detection](#duplicate-work-detection)
- [Confidence Scoring](#confidence-scoring)
- [Impact Assessment](#impact-assessment)
- [Tuning Parameters](#tuning-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What is a Coordination Gap?

A coordination gap occurs when organizational information flows break down, causing:

- **Duplicate Work**: Teams independently solving the same problem
- **Missing Context**: Decisions made without critical stakeholder input
- **Stale Documentation**: Documentation contradicting current implementation
- **Knowledge Silos**: Critical knowledge trapped within individual teams

### How Gap Detection Works

The system uses a multi-stage AI pipeline combining:
1. **Semantic Analysis**: Clustering similar discussions using embeddings
2. **Entity Extraction**: Identifying teams, people, and projects
3. **Temporal Analysis**: Detecting work happening simultaneously
4. **LLM Reasoning**: Using Claude to verify actual coordination failures
5. **Impact Scoring**: Quantifying organizational cost

---

## Detection Pipeline

### Stage 1: Data Retrieval

```python
# Retrieve messages from specified timeframe
messages = await retrieve_messages(
    timeframe_days=30,
    sources=["slack", "github", "google_docs"],
    channels=["#engineering", "#platform"]
)
```

**What happens**:
- Fetch messages from PostgreSQL database
- Filter by timeframe, sources, channels
- Load embeddings from vector store

### Stage 2: Semantic Clustering

```python
# Group similar technical discussions
clusters = semantic_clusterer.cluster(
    messages,
    similarity_threshold=0.85,  # 85% similarity required
    time_window_days=30,        # Within 30-day window
    min_cluster_size=2          # At least 2 messages
)
```

**What happens**:
- Compute cosine similarity between message embeddings
- Apply DBSCAN clustering algorithm
- Group messages with >85% semantic similarity
- Filter clusters within time window

**Key Insight**: Similar discussions about "OAuth" or "API design" get grouped together, even if in different channels.

### Stage 3: Entity Extraction

```python
# Identify teams, people, projects in each cluster
for cluster in clusters:
    entities = entity_extractor.extract(cluster.messages)

    # entities contains:
    # {
    #   "teams": ["platform-team", "auth-team"],
    #   "people": ["alice@company.com", "bob@company.com"],
    #   "projects": ["OAuth", "authentication"],
    #   "topics": ["implementation", "security"]
    # }
```

**What happens**:
- Extract @mentions (teams and people)
- Identify email addresses
- Detect channel-based teams (#platform → platform-team)
- Find project/feature mentions
- Extract technical terms

**Key Insight**: If multiple teams are found in a cluster, it's a potential coordination gap.

### Stage 4: Temporal Overlap Analysis

```python
# Check if teams are working simultaneously
def has_temporal_overlap(cluster, teams, min_overlap_days=3):
    team_timelines = {}

    for msg in cluster.messages:
        team = msg.metadata.get("team")
        if team not in team_timelines:
            team_timelines[team] = []
        team_timelines[team].append(msg.timestamp)

    # Calculate overlap between teams
    overlap = compute_timeline_overlap(team_timelines)
    return overlap >= min_overlap_days
```

**What happens**:
- Build timeline for each team's messages
- Calculate overlapping time periods
- Require minimum 3 days of simultaneous work

**Key Insight**: Teams working 2 months apart = knowledge sharing (good). Teams working same 2 weeks = potential duplication (investigate).

### Stage 5: LLM Verification

```python
# Use Claude to verify actual duplication
verification = await claude_client.verify_duplicate_work(
    messages=cluster.messages,
    teams=entities["teams"],
    topic=cluster.topic,
    context=cluster.summary
)

# verification contains:
# {
#   "is_duplicate": true,
#   "confidence": 0.87,
#   "reasoning": "Both teams independently implementing OAuth2...",
#   "evidence": ["Quote 1", "Quote 2"],
#   "recommendation": "Connect alice@ and bob@ immediately",
#   "overlap_ratio": 0.8  # 80% of work is duplicated
# }
```

**What happens**:
- Send cluster context to Claude API
- LLM analyzes if work is truly duplicated or intentionally coordinated
- Returns structured verification with reasoning
- Provides action recommendations

**Key Insight**: LLM can distinguish between:
- ✅ Duplication: "Starting OAuth implementation" (team A) + "Building OAuth support" (team B)
- ❌ Collaboration: "Working with @team-b on OAuth" + "@team-a is handling auth flow"

### Stage 6: Gap Creation & Storage

```python
if verification.is_duplicate and verification.confidence > 0.7:
    gap = CoordinationGap(
        id=generate_gap_id(),
        type="DUPLICATE_WORK",
        topic=cluster.topic,
        teams_involved=entities["teams"],
        evidence=collect_evidence(cluster, verification),
        impact_score=calculate_impact(cluster, entities),
        confidence=verification.confidence,
        verification=verification,
        recommendation=verification.recommendation,
        detected_at=datetime.utcnow()
    )

    # Store in PostgreSQL
    await db.save_gap(gap)
```

**What happens**:
- Only gaps with >70% confidence are saved
- Evidence is collected and ranked by relevance
- Impact is scored (0-1 scale)
- Gap stored in database for retrieval

---

## Duplicate Work Detection

### Detection Criteria

A cluster must meet **ALL** of these to be flagged as duplicate work:

1. ✅ **Semantic Similarity > 0.85**: Messages discuss similar topics
2. ✅ **Multiple Teams (≥2)**: At least 2 different teams involved
3. ✅ **Temporal Overlap (≥3 days)**: Teams working simultaneously
4. ✅ **LLM Verification (confidence >0.7)**: Claude confirms duplication

### Exclusion Rules

The following scenarios are **NOT** flagged as duplicate work:

- ❌ **Explicit Collaboration**: Cross-references present (@team mentions)
- ❌ **Different Scopes**: LLM identifies distinct goals (e.g., user auth vs service auth)
- ❌ **Sequential Work**: No temporal overlap (Team B starts after Team A completes)
- ❌ **Mentor/Mentee Pattern**: One team helping another learn
- ❌ **Intentional Redundancy**: Backup systems, A/B tests, disaster recovery

### Example: OAuth Duplication (Detected)

```
Platform Team (#platform channel):
  Dec 1, 09:00 - "Starting OAuth2 implementation for API gateway"
  Dec 3, 10:15 - "Finished OAuth flow research, going with authorization code"
  Dec 5, 14:20 - "Got token validation working"

Auth Team (#auth-team channel):
  Dec 1, 14:20 - "We're building OAuth support for auth service"
  Dec 3, 15:30 - "Decided on authorization code flow"
  Dec 5, 16:45 - "Token endpoint complete"

Detection Result:
✅ Semantic Similarity: 0.92 (both discussing OAuth2 implementation)
✅ Multiple Teams: 2 teams (platform-team, auth-team)
✅ Temporal Overlap: 14 days of parallel work
✅ LLM Confidence: 0.89
✅ No Cross-References: Teams not aware of each other

→ DUPLICATE WORK DETECTED (High Impact)
```

### Example: Collaboration (NOT Detected)

```
Platform Team:
  "Working with @auth-team on OAuth integration. We're handling the client flow."

Auth Team:
  "@platform-team is building the client, we're doing the server. Coordinating via sync meetings."

Detection Result:
✅ Semantic Similarity: 0.88
✅ Multiple Teams: 2 teams
❌ Collaboration Detected: Cross-references present
❌ LLM Verdict: "Teams are intentionally collaborating with clear division of labor"

→ NOT DUPLICATE WORK (Collaboration)
```

---

## Confidence Scoring

### Confidence Formula

```python
confidence = (
    0.3 * semantic_similarity_score +   # How similar are discussions?
    0.2 * team_separation_score +       # How isolated are teams?
    0.3 * temporal_overlap_score +      # How much simultaneous work?
    0.2 * llm_confidence                # LLM verification strength
)
```

### Confidence Tiers

| Score | Tier | Meaning | Action |
|-------|------|---------|--------|
| 0.9-1.0 | **Very High** | Extremely confident this is duplicate work | Immediate intervention |
| 0.8-0.9 | **High** | Strong evidence of duplication | Intervene within 24h |
| 0.7-0.8 | **Medium-High** | Likely duplicate, needs review | Flag for review |
| 0.6-0.7 | **Medium** | Possible duplicate, monitor | Monitor situation |
| <0.6 | **Low** | Insufficient evidence | No action (filtered out) |

### Calibration

Confidence scores are calibrated against:
- Known duplicate work scenarios (ground truth)
- False positive examples (collaboration, different scopes)
- Historical detection accuracy

**Target Performance**:
- Precision: >80% (low false positives)
- Recall: >70% (catch most duplicates)
- F1 Score: >75%

---

## Impact Assessment

### Impact Score Formula

```python
impact_score = (
    0.25 * team_size_score +            # How many people affected?
    0.25 * time_investment_score +      # How much time wasted?
    0.20 * project_criticality_score +  # How important is this?
    0.15 * velocity_impact_score +      # What's blocked?
    0.15 * duplicate_effort_score       # How much actual duplication?
)
```

Range: [0, 1] where 1 = catastrophic waste

### Impact Tiers

#### Critical (0.8 - 1.0) 🔴
- Multiple large teams (10+ people)
- 100+ engineering hours wasted
- Roadmap/OKR-critical work
- **Action**: Immediate intervention required

#### High (0.6 - 0.8) 🟠
- 5-10 people affected
- 40-100 hours wasted
- Important projects
- **Action**: Address within 1 week

#### Medium (0.4 - 0.6) 🟡
- 2-5 people affected
- 10-40 hours wasted
- Moderate importance
- **Action**: Monitor and advise teams

#### Low (0.0 - 0.4) 🟢
- Small scope (<10 hours)
- Low criticality
- **Action**: FYI notification only

### Cost Estimation

```python
# Organizational waste cost (not Claude API cost)
avg_hourly_rate = 100  # $100/hour loaded engineer cost
estimated_hours = calculate_time_investment(gap)
estimated_cost = estimated_hours * avg_hourly_rate

# Example: 60 hours × $100 = $6,000 organizational waste
# Claude API cost is separate: ~$0.01-0.05 per gap verification
```

---

## Tuning Parameters

### Similarity Threshold

```python
similarity_threshold = 0.85  # Default: 85% similarity
```

**Increase (0.90)** if:
- Too many false positives
- Need higher precision
- Want only extremely similar work

**Decrease (0.80)** if:
- Missing obvious duplicates
- Teams use different terminology
- Want higher recall

### Temporal Overlap Minimum

```python
min_overlap_days = 3  # Default: 3 days
```

**Increase (5-7 days)** if:
- Short overlaps causing false positives
- Want sustained parallel effort only

**Decrease (1-2 days)** if:
- Fast-moving teams (sprints)
- Short project cycles

### LLM Confidence Threshold

```python
min_llm_confidence = 0.7  # Default: 70%
```

**Increase (0.8)** if:
- Too many low-confidence gaps
- Want only clear-cut cases

**Decrease (0.6)** if:
- Missing edge cases
- Want to catch uncertain situations

### Impact Score Filter

```python
min_impact_score = 0.6  # Default: High+ impact only
```

**Increase (0.8)** if:
- Only want critical gaps
- High volume of gaps

**Decrease (0.4)** if:
- Want to see all gaps
- Learning mode

---

## Best Practices

### 1. Start Conservative

```python
# Recommended initial settings
{
    "similarity_threshold": 0.90,      # High precision
    "min_overlap_days": 5,             # Sustained effort
    "min_llm_confidence": 0.8,         # Clear cases only
    "min_impact_score": 0.7            # High+ impact
}
```

Gradually relax thresholds as you gain confidence.

### 2. Regular Monitoring

- Run detection weekly
- Review false positives/negatives
- Adjust thresholds based on results
- Track precision/recall over time

### 3. Act on High-Impact Gaps Immediately

- Critical gaps (0.8+): Intervene within 24h
- High gaps (0.6-0.8): Address within 1 week
- Document resolution outcomes
- Use for future calibration

### 4. Provide Context to Teams

When notifying teams about gaps:
- Share full evidence (messages, timestamps)
- Explain detection reasoning
- Provide recommendation
- Ask for feedback on accuracy

### 5. Iterate on Prompts

LLM verification prompts can be tuned:
- Add domain-specific context
- Include organizational patterns
- Refine verification questions
- A/B test prompt variations

---

## Troubleshooting

### Issue: No Gaps Detected (Expected Some)

**Possible Causes**:
1. Similarity threshold too high → Lower to 0.80-0.85
2. Temporal overlap too strict → Reduce min_overlap_days to 2-3
3. Impact filter excluding results → Lower min_impact_score to 0.4
4. LLM confidence too high → Lower to 0.6-0.7

**Debugging**:
```bash
# Check clustering output
curl -X POST /api/v1/gaps/detect \
  -d '{"timeframe_days": 30, "min_impact_score": 0.0}'

# Review logs for cluster statistics
docker compose logs api | grep "clusters_found"
```

### Issue: Too Many False Positives

**Possible Causes**:
1. Similarity threshold too low → Increase to 0.90-0.92
2. Collaboration not being detected → Review LLM prompt
3. Different scopes confused → Add scope detection to prompt

**Resolution**:
- Review false positives to find patterns
- Adjust exclusion rules
- Refine LLM verification prompt
- Add domain-specific heuristics

### Issue: Slow Detection Performance

**Performance Targets**:
- Entity extraction: <20ms per message
- Clustering: <500ms for 1000 messages
- LLM verification: <2s per gap
- Full detection: <5s for 30 days

**Optimization Steps**:
1. Enable embedding caching (24h TTL)
2. Batch LLM calls (verify multiple gaps together)
3. Reduce timeframe_days
4. Limit sources/channels
5. Use faster Claude model (Haiku for verification)

### Issue: LLM API Errors

**Common Errors**:
1. **Rate Limit (429)**: Exponential backoff, reduce request rate
2. **Invalid API Key (401)**: Check ANTHROPIC_API_KEY in .env
3. **Timeout (408)**: Increase timeout, use shorter prompts
4. **Quota Exceeded (429)**: Wait for quota reset, implement daily limits

**Monitoring**:
```bash
# Check API usage
docker compose logs api | grep "LLM_API"

# View retry attempts
docker compose logs api | grep "retry"
```

---

## Advanced Topics

### Multi-Gap Detection

Detect multiple gap types in one pass:

```python
response = await client.post("/api/v1/gaps/detect", json={
    "gap_types": [
        "duplicate_work",
        "missing_context",    # Future
        "stale_docs",         # Future
        "knowledge_silo"      # Future
    ]
})
```

### Cross-Source Detection

Combine evidence from multiple sources:

```python
# Detect across Slack + GitHub + Google Docs
{
    "sources": ["slack", "github", "google_docs"],
    "timeframe_days": 30
}

# Result: Evidence from multiple sources
{
    "evidence": [
        {"source": "slack", "content": "Starting OAuth..."},
        {"source": "github", "pr": "#123", "title": "Add OAuth..."},
        {"source": "google_docs", "doc": "OAuth Design Doc"}
    ]
}
```

### Historical Analysis

Analyze past gaps to prevent future ones:

```python
# Get gaps from last 90 days
gaps = await client.get("/api/v1/gaps", params={
    "start_date": "2024-10-01",
    "end_date": "2024-12-31"
})

# Analyze patterns
- Which teams frequently duplicate?
- What topics are commonly duplicated?
- What's the average resolution time?
```

---

## Summary

**Key Takeaways**:

1. **Multi-Stage Pipeline**: Clustering → Entity Extraction → Temporal Analysis → LLM Verification
2. **High Precision**: Default thresholds favor low false positives
3. **Tunable**: Adjust parameters based on your organization
4. **Evidence-Based**: Every gap includes detailed evidence and reasoning
5. **Actionable**: Recommendations help teams resolve issues

**Next Steps**:
- [Entity Extraction Guide](ENTITY_EXTRACTION.md)
- [API Usage Examples](API_EXAMPLES.md)
- [Troubleshooting Guide](../README.md#troubleshooting)

---

**Last Updated**: December 2024
**Version**: 1.0
**Milestone**: 3H - Testing & Documentation
