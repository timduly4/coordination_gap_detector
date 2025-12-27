# Gap Detection Scenarios

This directory contains realistic mock data scenarios for testing and demonstrating the coordination gap detection system.

## Overview

These scenarios are designed to test the gap detection algorithms under various conditions:
- **Positive cases**: Should detect as duplicate work
- **Edge cases**: Similar situations that might confuse the detector
- **Negative examples**: Should NOT detect (intentional collaboration)

## Scenarios

### Positive Cases (Should Detect)

#### 1. OAuth Duplication (`oauth_duplication`)
**Teams**: Platform Team, Auth Team
**Timeframe**: 14 days of parallel work
**Messages**: 24+ messages
**Expected Impact**: HIGH (0.85+)

Both teams independently implementing OAuth2:
- Platform team building OAuth client integration
- Auth team building full OAuth server
- Same design decisions (authorization code flow, PKCE, JWT tokens)
- Parallel timelines with significant overlap
- Discovery moment on Day 14

**Key Indicators**:
- ✅ Semantic similarity > 0.85
- ✅ Multiple teams (2)
- ✅ Temporal overlap (14 days)
- ✅ Similar technical decisions
- ❌ No cross-references

#### 2. API Redesign Duplication (`api_redesign_duplication`)
**Teams**: Platform Team, Backend Team
**Timeframe**: 18 days of parallel work
**Messages**: 18 messages
**Expected Impact**: HIGH (0.78+)

Both teams independently redesigning REST API structure:
- Same principles (RESTful, resource naming, HTTP verbs)
- Same versioning strategy (URL-based /v1, /v2)
- Same error handling format
- Same pagination approach (cursor-based)
- Same rate limiting strategy

**Key Indicators**:
- ✅ Architectural duplicate work
- ✅ Multiple implementation decisions align
- ✅ Long overlap period
- ❌ No coordination

#### 3. Auth Migration Duplication (`auth_migration_duplication`)
**Teams**: Security Team, Platform Team
**Timeframe**: 10 days of parallel work
**Messages**: 10 messages
**Expected Impact**: MEDIUM-HIGH (0.72+)

Both teams migrating from sessions to JWT:
- Same motivation (scaling issues)
- Same algorithm choice (RS256)
- Same token expiry strategy
- Same migration approach (gradual)

**Key Indicators**:
- ✅ Duplicate migration work
- ✅ Similar technical choices
- ✅ Temporal overlap
- ✅ Medium scope (fewer messages than others)

### Edge Cases (Should NOT Detect)

#### 4. Similar Topics, Different Scope (`similar_topics_different_scope`)
**Teams**: Frontend Team (user auth), Platform Team (service-to-service auth)
**Timeframe**: 8 days
**Messages**: 4 messages
**Expected**: Should NOT detect

Both discussing "authentication" but completely different:
- Frontend: Social logins, magic links (user authentication)
- Platform: mTLS, service certificates (service-to-service auth)
- Different problems, different solutions

**Why it shouldn't detect**:
- LLM should identify different scopes
- Semantic similarity might be moderate but context is different
- No actual duplication of work

#### 5. Sequential Work, No Overlap (`sequential_work`)
**Teams**: Web Team (Nov), Mobile Team (Jan)
**Timeframe**: 90 days total, 0 days overlap
**Messages**: 4 messages
**Expected**: Should NOT detect

OAuth implementation at different times:
- Web team: November (completed)
- Mobile team: January (60 days later)
- Mobile team explicitly references web team's work

**Why it shouldn't detect**:
- No temporal overlap (sequential, not parallel)
- Mobile team building on web team's work (knowledge sharing)
- Intentional learning from previous work

### Negative Examples (Intentional Collaboration)

#### 6. Intentional Collaboration (`intentional_collaboration`)
**Teams**: Platform Team, Auth Team
**Timeframe**: 12 days
**Messages**: 6 messages
**Expected**: Should NOT detect

OAuth implementation with clear collaboration:
- Explicit coordination announcement
- Clear division of labor (server vs client)
- Cross-references in messages
- Joint sync meetings mentioned
- Shared design docs

**Why it shouldn't detect**:
- ✅ Cross-references present (@auth-team, @platform-team)
- ✅ Explicit collaboration language
- ✅ Division of labor mentioned
- ✅ Shared planning

## Using Scenarios

### Load Specific Scenario
```bash
# Load individual scenario
uv run python scripts/generate_mock_data.py --scenarios oauth_duplication --clear

# Load multiple scenarios
uv run python scripts/generate_mock_data.py --scenarios oauth_duplication api_redesign_duplication

# Load all gap detection scenarios
uv run python scripts/generate_mock_data.py --scenarios oauth_duplication api_redesign_duplication auth_migration_duplication similar_topics_different_scope sequential_work intentional_collaboration
```

### Run Detection
```bash
# After loading scenarios, run gap detection
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6,
    "include_evidence": true
  }'
```

### Expected Results

**Should Detect (3 gaps)**:
1. OAuth Duplication (impact ~0.85, HIGH)
2. API Redesign Duplication (impact ~0.78, HIGH)
3. Auth Migration Duplication (impact ~0.72, MEDIUM-HIGH)

**Should NOT Detect (3 scenarios)**:
4. Similar Topics Different Scope (different problem domains)
5. Sequential Work (no temporal overlap)
6. Intentional Collaboration (explicit coordination)

## Scenario Design Principles

### Positive Cases
- **Temporal overlap**: Minimum 3-5 days of parallel work
- **Semantic similarity**: >0.85 in technical discussions
- **Team isolation**: No cross-references between teams
- **Similar decisions**: Same choices made independently
- **Sufficient volume**: Enough messages to show real work

### Edge Cases
- **Deceptive similarity**: Looks similar at first glance
- **Different context**: LLM reasoning needed to distinguish
- **Tests robustness**: Prevents false positives

### Negative Examples
- **Clear collaboration signals**: Cross-references, mentions
- **Explicit coordination**: Meeting mentions, shared docs
- **Division of labor**: Different scopes assigned

## Testing

Run integration tests with scenarios:
```bash
# Test detection with all scenarios
pytest tests/test_integration/test_duplicate_detection_scenarios.py -v

# Test specific scenario
pytest tests/test_integration/test_duplicate_detection_scenarios.py::test_oauth_duplication_scenario -v
```

## Metrics

Successful detection should achieve:
- **Precision**: >0.80 (few false positives)
- **Recall**: >0.70 (catch most duplicates)
- **F1 Score**: >0.75
- **False Positive Rate**: <0.20

Measured across all 6 scenarios:
- 3 true positives (should detect)
- 3 true negatives (should not detect)

## Adding New Scenarios

To add a new scenario:

1. **Add method to MockSlackClient** (`src/ingestion/slack/mock_client.py`):
```python
def _generate_your_scenario(self) -> List[MockMessage]:
    """Description of scenario."""
    base_time = datetime.utcnow() - timedelta(days=X)
    messages = [...]
    return messages
```

2. **Register in __init__**:
```python
self.scenarios = {
    ...
    "your_scenario": self._generate_your_scenario,
}
```

3. **Add description**:
```python
def get_scenario_descriptions(self):
    return {
        ...
        "your_scenario": "Description (expected result)",
    }
```

4. **Document here** in README.md with:
   - Teams involved
   - Timeframe
   - Message count
   - Expected detection result
   - Key indicators

5. **Add test** in `tests/test_integration/test_duplicate_detection_scenarios.py`

## Scenario Statistics

| Scenario | Teams | Days | Messages | Should Detect | Impact |
|----------|-------|------|----------|---------------|--------|
| oauth_duplication | 2 | 14 | 24 | ✅ Yes | HIGH |
| api_redesign_duplication | 2 | 18 | 18 | ✅ Yes | HIGH |
| auth_migration_duplication | 2 | 10 | 10 | ✅ Yes | MEDIUM-HIGH |
| similar_topics_different_scope | 2 | 8 | 4 | ❌ No | N/A |
| sequential_work | 2 | 0 overlap | 4 | ❌ No | N/A |
| intentional_collaboration | 2 | 12 | 6 | ❌ No | N/A |

## Notes

- **Timestamps**: Scenarios use `datetime.utcnow() - timedelta(...)` for consistent recent data
- **Team metadata**: Each message includes `team` in metadata for extraction testing
- **Thread IDs**: Messages grouped by logical threads
- **Reactions**: Included for realism but not used in detection
- **Discovery moments**: Some scenarios include explicit discovery (e.g., "Wait, are both teams working on this?")

---

Last Updated: December 2024
Milestone: 3G - Mock Data Scenarios for Duplicate Work
