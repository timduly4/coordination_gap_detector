# Milestone 3: Simple Gap Detection - PR Breakdown

## Overview
Milestone 3 implements the first coordination gap detection capability: identifying duplicate work across teams. This milestone demonstrates AI-powered pattern detection, LLM reasoning, and practical organizational insights. Breaking this into focused PRs shows methodical development of a complex AI system.

---

## Milestone 3A: Entity Extraction Foundation
**Branch**: `feat/entity-extraction`
**Time**: ~3-4 hours
**Files Changed**: ~6-8 files

### Changes:
```
coordination-gap-detector/
├── pyproject.toml              # Add spaCy, regex dependencies
└── src/
    ├── analysis/
    │   ├── __init__.py
    │   ├── entity_extraction.py   # Core extraction logic
    │   └── entity_types.py        # Entity type definitions
    ├── models/
    │   └── schemas.py             # Updated with entity schemas
    └── utils/
        └── text_processing.py     # Enhanced text utilities
tests/
└── test_analysis/
    ├── __init__.py
    └── test_entity_extraction.py
```

### Specific Tasks:
- [ ] Add spaCy or regex-based entity extraction
- [ ] Implement person/author extraction
- [ ] Implement team name detection
- [ ] Extract project/feature mentions
- [ ] Add technical term identification
- [ ] Create entity normalization (e.g., @alice → alice@company.com)
- [ ] Add confidence scoring for extractions
- [ ] Write comprehensive entity extraction tests
- [ ] Add entity extraction to message processing pipeline

### PR Description Template:
```markdown
## Entity Extraction Foundation

Extract people, teams, and projects from messages for gap detection.

### Changes
- Entity extraction for people, teams, projects
- Pattern-based and NLP extraction methods
- Entity normalization and deduplication
- Confidence scoring for extractions
- Integration with message processing
- Comprehensive tests

### Entity Types Extracted

**1. People/Authors**
- Direct mentions: @alice, @bob
- Email addresses: alice@company.com
- Full names: Alice Johnson
- Normalized to canonical form

**2. Teams**
- Team mentions: @platform-team, @auth-team
- Channel-based inference: #platform → platform team
- Department names: engineering, product, design
- Team name variations handled

**3. Projects/Features**
- Feature names: OAuth, authentication, API gateway
- Project codes: PROJ-123, EPIC-456
- Technical terms: microservices, Kubernetes, React
- Acronyms: SSO, RBAC, JWT

**4. Topics**
- Discussion topics via keyword extraction
- Technical stack mentions
- Problem/solution identification

### Extraction Methods

**Pattern-based (Regex)**
- Fast and deterministic
- Good for structured mentions (@mentions, #channels)
- Handles email addresses, URLs

**NLP-based (Optional: spaCy)**
- Named entity recognition (NER)
- Better for unstructured text
- Handles context and ambiguity

### Testing
- [ ] Extracts @mentions correctly
- [ ] Identifies team names from channels
- [ ] Finds project references
- [ ] Normalizes entity variations
- [ ] Handles edge cases (typos, formatting)
- [ ] Confidence scores are reasonable
- [ ] Performance acceptable (<20ms per message)

### Example Usage
```python
from src.analysis.entity_extraction import EntityExtractor

extractor = EntityExtractor()
message = {
    "content": "@alice and @bob from @platform-team are working on OAuth integration",
    "channel": "#platform",
    "author": "charlie@company.com"
}

entities = extractor.extract(message)
# Returns:
# {
#   "people": ["alice@company.com", "bob@company.com"],
#   "teams": ["platform-team"],
#   "projects": ["OAuth"],
#   "topics": ["integration"]
# }
```
```

**Commit Messages:**
```
feat: implement entity extraction for people and teams
feat: add project and topic extraction
feat: create entity normalization and deduplication
test: add comprehensive entity extraction tests
```

---

## Milestone 3B: Semantic Clustering
**Branch**: `feat/semantic-clustering`
**Time**: ~3-4 hours
**Files Changed**: ~6-8 files

### Changes:
```
src/
├── detection/
│   ├── __init__.py
│   ├── patterns.py              # Base detection interface
│   └── clustering.py            # Semantic clustering logic
├── analysis/
│   └── similarity.py            # Similarity computation
└── models/
    └── schemas.py               # Cluster schemas
tests/
└── test_detection/
    ├── __init__.py
    ├── test_clustering.py
    └── test_similarity.py
```

### Specific Tasks:
- [ ] Implement semantic similarity computation
- [ ] Create clustering algorithm (DBSCAN or hierarchical)
- [ ] Add temporal awareness (cluster by time window)
- [ ] Implement cluster quality metrics
- [ ] Add cluster labeling/summarization
- [ ] Create cluster visualization helpers
- [ ] Optimize clustering performance
- [ ] Write clustering algorithm tests
- [ ] Add integration tests with vector store

### PR Description Template:
```markdown
## Semantic Clustering for Pattern Detection

Group similar messages to identify coordination patterns.

### Changes
- Semantic similarity computation using embeddings
- DBSCAN clustering for message grouping
- Temporal clustering windows
- Cluster quality metrics (silhouette score)
- Cluster summarization and labeling
- Performance optimizations
- Comprehensive tests

### Clustering Approach

**Algorithm: DBSCAN (Density-Based Spatial Clustering)**

Why DBSCAN?
- No need to specify number of clusters upfront
- Handles noise and outliers
- Finds arbitrarily shaped clusters
- Works well with cosine similarity

Parameters:
- `eps`: Similarity threshold (default: 0.15)
- `min_samples`: Minimum messages per cluster (default: 2)
- `metric`: cosine distance

**Temporal Windowing**

Messages clustered within time windows:
- Default: 30-day rolling window
- Prevents matching very old discussions
- Configurable per detection type

### Cluster Quality Metrics

**Silhouette Score**
- Range: [-1, 1]
- >0.5: Good clustering
- Measures cluster cohesion vs separation

**Cluster Statistics**
- Size: Number of messages
- Density: Average intra-cluster similarity
- Timespan: First to last message
- Participant count: Unique authors

### Testing
- [ ] Clusters similar messages together
- [ ] Different topics stay separate
- [ ] Temporal windowing works correctly
- [ ] Outliers handled appropriately
- [ ] Quality metrics calculated correctly
- [ ] Performance acceptable (<500ms for 1000 messages)
- [ ] Empty/single message handling

### Example Usage
```python
from src.detection.clustering import SemanticClusterer

clusterer = SemanticClusterer(
    similarity_threshold=0.85,
    time_window_days=30,
    min_cluster_size=2
)

messages = [...]  # List of messages
clusters = clusterer.cluster(messages)

for cluster in clusters:
    print(f"Cluster: {cluster.label}")
    print(f"  Size: {cluster.size}")
    print(f"  Similarity: {cluster.avg_similarity:.2f}")
    print(f"  Timespan: {cluster.timespan_days} days")
    print(f"  Messages: {[m.content[:50] for m in cluster.messages]}")
```
```

**Commit Messages:**
```
feat: implement semantic clustering with DBSCAN
feat: add temporal windowing for clusters
feat: create cluster quality metrics
feat: add cluster labeling and summarization
test: add clustering algorithm tests
perf: optimize clustering for large message sets
```

---

## Milestone 3C: Claude API Integration
**Branch**: `feat/claude-api-integration`
**Time**: ~3-4 hours
**Files Changed**: ~7-9 files

### Changes:
```
pyproject.toml                   # Add anthropic SDK
.env.example                     # Add ANTHROPIC_API_KEY
src/
├── models/
│   ├── llm.py                   # Claude API wrapper
│   ├── reasoning.py             # Multi-step reasoning
│   └── prompts.py               # Prompt templates
├── config.py                    # Updated with Anthropic config
└── utils/
    └── token_utils.py           # Token counting (tiktoken)
tests/
└── test_models/
    ├── test_llm.py
    └── test_reasoning.py
```

### Specific Tasks:
- [ ] Add Anthropic SDK to dependencies
- [ ] Create Claude API client wrapper
- [ ] Implement retry logic with exponential backoff
- [ ] Add rate limiting and quota management
- [ ] Create prompt templates for gap verification
- [ ] Implement structured output parsing
- [ ] Add token counting and cost tracking
- [ ] Create reasoning chains for gap analysis
- [ ] Write tests (with mocked API calls)
- [ ] Add error handling for API failures

### PR Description Template:
```markdown
## Claude API Integration

LLM reasoning for gap verification and insight generation.

### Changes
- Anthropic Claude API client wrapper
- Retry logic with exponential backoff
- Rate limiting and quota management
- Prompt template system
- Structured output parsing
- Token counting and cost tracking
- Multi-step reasoning chains
- Comprehensive error handling
- Tests with mocked responses

### API Client Features

**Reliability**
- Exponential backoff for rate limits (2^n seconds)
- Automatic retries (max 3 attempts)
- Graceful degradation on API errors
- Request timeout handling

**Cost Management**
- Token counting with tiktoken
- Cost estimation per request
- Daily quota tracking
- Warning on high usage

**Structured Output**
- JSON schema validation
- Pydantic model parsing
- Type-safe responses
- Fallback on parse errors

### Prompt Templates

**Gap Verification Prompt**
```
You are analyzing organizational communication to detect coordination gaps.

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
{
  "is_duplicate": boolean,
  "confidence": 0-1,
  "reasoning": "explanation",
  "evidence": ["key quotes"],
  "recommendation": "action to take"
}
```

### Testing
- [ ] API client connects successfully
- [ ] Retries work on transient failures
- [ ] Rate limiting prevents excessive calls
- [ ] Token counting is accurate
- [ ] Prompt templates render correctly
- [ ] Structured output parsed properly
- [ ] Error handling works (network, API, parsing)
- [ ] Cost tracking accumulates correctly

### Example Usage
```python
from src.models.llm import ClaudeClient
from src.models.prompts import GAP_VERIFICATION_PROMPT

client = ClaudeClient()

response = client.complete(
    prompt=GAP_VERIFICATION_PROMPT.format(
        topic="OAuth implementation",
        message_list=messages,
        teams=["platform-team", "auth-team"],
        timeframe="Dec 1-15, 2024"
    ),
    model="claude-sonnet-4-5",
    max_tokens=1000
)

# Structured output
result = client.parse_json(response)
# {
#   "is_duplicate": true,
#   "confidence": 0.89,
#   "reasoning": "Both teams are implementing OAuth2...",
#   ...
# }
```

### Configuration
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-5
CLAUDE_MAX_TOKENS=4096
CLAUDE_TEMPERATURE=0.3
CLAUDE_DAILY_QUOTA_TOKENS=1000000
```
```

**Commit Messages:**
```
feat: add anthropic SDK and claude api client
feat: implement retry logic and rate limiting
feat: create prompt template system
feat: add structured output parsing
feat: implement token counting and cost tracking
test: add comprehensive LLM client tests
```

---

## Milestone 3D: Duplicate Work Detection Algorithm
**Branch**: `feat/duplicate-work-detection`
**Time**: ~4-5 hours
**Files Changed**: ~8-10 files

### Changes:
```
src/
├── detection/
│   ├── duplicate_work.py        # Core detection algorithm
│   ├── patterns.py              # Updated with base classes
│   └── validators.py            # Gap validation logic
├── models/
│   └── schemas.py               # Gap schemas
└── services/
    └── detection_service.py     # Detection orchestration
tests/
└── test_detection/
    ├── test_duplicate_work.py
    └── test_validators.py
```

### Specific Tasks:
- [ ] Implement duplicate work detection algorithm
- [ ] Create detection pipeline (cluster → extract → verify)
- [ ] Add team overlap detection
- [ ] Implement temporal overlap checking
- [ ] Create gap validation rules
- [ ] Add evidence collection and ranking
- [ ] Implement detection confidence scoring
- [ ] Create detection service for orchestration
- [ ] Write comprehensive detection tests
- [ ] Add integration tests with real scenarios

### PR Description Template:
```markdown
## Duplicate Work Detection Algorithm

Core algorithm for identifying parallel efforts across teams.

### Changes
- Duplicate work detection algorithm
- Multi-stage detection pipeline
- Team overlap analysis
- Temporal overlap checking
- Gap validation rules
- Evidence collection and ranking
- Confidence scoring
- Detection service orchestration
- Comprehensive tests

### Detection Algorithm

**Stage 1: Semantic Clustering**
```python
# Group similar technical discussions
clusters = semantic_clusterer.cluster(
    messages,
    similarity_threshold=0.85,
    time_window_days=30
)
```

**Stage 2: Team Detection**
```python
# Identify teams involved in each cluster
for cluster in clusters:
    teams = entity_extractor.extract_teams(cluster.messages)
    if len(teams) > 1:
        potential_gaps.append((cluster, teams))
```

**Stage 3: Temporal Overlap Check**
```python
# Verify teams working simultaneously
def has_temporal_overlap(cluster, min_overlap_days=3):
    team_timelines = build_team_timelines(cluster)
    overlap = compute_overlap(team_timelines)
    return overlap >= min_overlap_days
```

**Stage 4: LLM Verification**
```python
# Use Claude to verify actual duplication
verification = claude_client.verify_duplicate_work(
    messages=cluster.messages,
    teams=teams,
    context=cluster.context
)

if verification.is_duplicate and verification.confidence > 0.7:
    gaps.append(create_gap(cluster, teams, verification))
```

### Detection Rules

**Must Meet All:**
1. ✅ Semantic similarity > 0.85 (same problem domain)
2. ✅ Multiple teams involved (>1 team)
3. ✅ Temporal overlap (working within same timeframe)
4. ✅ LLM confirms duplication (confidence > 0.7)

**Exclusion Rules:**
- ❌ Teams explicitly collaborating (cross-references present)
- ❌ Different problem scopes (LLM identifies distinct goals)
- ❌ One team helping another (mentor/mentee pattern)
- ❌ Intentional redundancy (backup systems, A/B tests)

### Confidence Scoring

```python
confidence = (
    0.3 * semantic_similarity +      # How similar are discussions?
    0.2 * team_separation_score +    # How isolated are teams?
    0.3 * temporal_overlap_score +   # How much time overlap?
    0.2 * llm_confidence             # LLM verification confidence
)
```

### Evidence Collection

For each gap, collect:
- Key messages showing parallel work
- Timeline visualization
- Team participant lists
- Technical terms/features mentioned
- Cross-references (or lack thereof)

Ranked by:
1. Earliest mentions from each team
2. Messages with clearest intent
3. Decision points
4. Implementation details

### Testing
- [ ] Detects true duplicate work scenarios
- [ ] Rejects false positives (collaboration)
- [ ] Handles edge cases (single team, no overlap)
- [ ] Confidence scores are calibrated
- [ ] Evidence collection is complete
- [ ] Performance acceptable (<5s per detection run)
- [ ] Integration with clustering works
- [ ] LLM verification integrated correctly

### Example Output
```json
{
  "gap_type": "DUPLICATE_WORK",
  "gap_id": "dup_abc123",
  "confidence": 0.89,
  "teams": ["platform-team", "auth-team"],
  "topic": "OAuth2 integration",
  "evidence": [
    {
      "message_id": "msg_001",
      "team": "platform-team",
      "content": "Starting OAuth2 implementation...",
      "timestamp": "2024-12-01T09:00:00Z",
      "relevance_score": 0.95
    },
    {
      "message_id": "msg_045",
      "team": "auth-team",
      "content": "We're building OAuth support...",
      "timestamp": "2024-12-01T14:20:00Z",
      "relevance_score": 0.92
    }
  ],
  "temporal_overlap": {
    "start": "2024-12-01",
    "end": "2024-12-15",
    "overlap_days": 14
  },
  "verification": {
    "is_duplicate": true,
    "llm_confidence": 0.87,
    "reasoning": "Both teams are independently implementing OAuth2..."
  }
}
```
```

**Commit Messages:**
```
feat: implement duplicate work detection algorithm
feat: add team overlap and temporal analysis
feat: integrate LLM verification into detection
feat: create evidence collection and ranking
test: add comprehensive duplicate work detection tests
```

---

## Milestone 3E: Impact Scoring
**Branch**: `feat/impact-scoring`
**Time**: ~2-3 hours
**Files Changed**: ~5-7 files

### Changes:
```
src/
├── detection/
│   ├── impact_scoring.py        # Impact calculation
│   └── cost_estimation.py       # Cost estimation models
└── models/
    └── schemas.py               # Updated with impact schemas
tests/
└── test_detection/
    └── test_impact_scoring.py
```

### Specific Tasks:
- [ ] Implement impact scoring algorithm
- [ ] Add team size estimation
- [ ] Create time investment calculation
- [ ] Add project criticality scoring
- [ ] Implement cost estimation (engineering hours)
- [ ] Create impact visualization helpers
- [ ] Add impact tier classification (low/medium/high)
- [ ] Write impact scoring tests
- [ ] Add calibration with realistic scenarios

### PR Description Template:
```markdown
## Impact Scoring for Coordination Gaps

Quantify the organizational cost of coordination failures.

### Changes
- Multi-signal impact scoring algorithm
- Team size and seniority estimation
- Time investment calculation
- Project criticality assessment
- Engineering cost estimation
- Impact tier classification
- Visualization helpers
- Comprehensive tests

### Impact Scoring Model

**Formula**:
```python
impact_score = (
    0.25 * team_size_score +           # How many people affected?
    0.25 * time_investment_score +     # How much time wasted?
    0.20 * project_criticality_score + # How important is this?
    0.15 * velocity_impact_score +     # What's the opportunity cost?
    0.15 * duplicate_effort_score      # How much actual duplication?
)
```

Range: [0, 1], where 1 is catastrophic waste

### Scoring Components

**1. Team Size Score (0.25 weight)**
```python
# Estimate people involved
team_size = sum(unique_participants_per_team)

# Score based on total people-hours at risk
if team_size >= 10:
    score = 1.0
elif team_size >= 5:
    score = 0.7
elif team_size >= 2:
    score = 0.4
else:
    score = 0.2
```

**2. Time Investment Score (0.25 weight)**
```python
# Estimate from message volume, timespan, commit activity
message_count = len(cluster.messages)
timespan_days = cluster.timespan_days
commit_count = count_related_commits(cluster)

# Heuristic: 1 message ≈ 30 min, 1 commit ≈ 2 hours
estimated_hours = (
    message_count * 0.5 +
    commit_count * 2.0
)

# Normalize to [0, 1]
# 100+ hours = 1.0 (critical waste)
score = min(1.0, estimated_hours / 100)
```

**3. Project Criticality Score (0.20 weight)**
```python
# Based on project tags, mentions, channel importance
signals = {
    "roadmap_item": 0.9,      # Mentioned in roadmap
    "okr_related": 0.85,       # Tied to OKRs
    "security": 0.8,           # Security-related
    "production": 0.75,        # Production system
    "customer_facing": 0.7,    # Customer impact
    "internal_tool": 0.3,      # Internal only
}

score = max(signals.get(tag, 0.5) for tag in project_tags)
```

**4. Velocity Impact Score (0.15 weight)**
```python
# What's blocked by this duplicate work?
blocking_work = count_blocked_items(gap)
score = min(1.0, blocking_work / 5)  # 5+ blocked items = max
```

**5. Duplicate Effort Score (0.15 weight)**
```python
# How much actual overlap vs complementary work?
overlap_ratio = llm_verification.overlap_ratio
score = overlap_ratio  # Direct from LLM assessment
```

### Impact Tiers

**Critical (0.8 - 1.0)**
- 🔴 Multiple large teams
- 100+ engineering hours wasted
- Roadmap/OKR impact
- Immediate intervention needed

**High (0.6 - 0.8)**
- 🟠 5-10 people affected
- 40-100 hours wasted
- Important projects
- Address within week

**Medium (0.4 - 0.6)**
- 🟡 2-5 people affected
- 10-40 hours wasted
- Moderate importance
- Monitor and advise

**Low (0.0 - 0.4)**
- 🟢 Small scope
- <10 hours wasted
- Low criticality
- FYI only

### Cost Estimation

```python
# Engineering cost calculation
avg_hourly_rate = 100  # $100/hour (loaded cost)
estimated_hours = calculate_time_investment(gap)
estimated_cost = estimated_hours * avg_hourly_rate

# Example: 60 hours × $100 = $6,000 wasted
```

### Testing
- [ ] Impact scores in valid range [0, 1]
- [ ] Larger gaps score higher
- [ ] Calibrated against realistic scenarios
- [ ] All components contribute appropriately
- [ ] Edge cases handled (missing data)
- [ ] Cost estimates are reasonable
- [ ] Tier classification correct

### Example Output
```json
{
  "gap_id": "dup_abc123",
  "impact_score": 0.89,
  "impact_tier": "CRITICAL",
  "breakdown": {
    "team_size_score": 0.9,
    "time_investment_score": 0.85,
    "project_criticality_score": 0.9,
    "velocity_impact_score": 0.8,
    "duplicate_effort_score": 0.95
  },
  "estimated_cost": {
    "engineering_hours": 85,
    "dollar_value": 8500,
    "explanation": "2 teams × ~40 hours each + coordination overhead"
  },
  "details": {
    "people_affected": 8,
    "timespan_days": 14,
    "messages_analyzed": 32,
    "commits_found": 12
  }
}
```
```

**Commit Messages:**
```
feat: implement multi-signal impact scoring
feat: add team size and time investment estimation
feat: create project criticality assessment
feat: add engineering cost estimation
test: add impact scoring tests with calibration
```

---

## Milestone 3F: Gap Detection API Endpoint
**Branch**: `feat/gap-detection-endpoint`
**Time**: ~3-4 hours
**Files Changed**: ~7-9 files

### Changes:
```
src/
├── api/
│   └── routes/
│       ├── gaps.py              # Gap detection endpoints
│       └── __init__.py          # Updated router
├── services/
│   └── gap_service.py           # Gap service orchestration
├── models/
│   └── schemas.py               # Request/response models
└── main.py                      # Updated with gaps router
tests/
└── test_api/
    └── test_gaps.py
```

### Specific Tasks:
- [ ] Create gap detection endpoint (POST /api/v1/gaps/detect)
- [ ] Add gap retrieval endpoints (GET /api/v1/gaps, GET /api/v1/gaps/{id})
- [ ] Implement GapService for business logic
- [ ] Add request/response validation
- [ ] Create filtering and pagination
- [ ] Add error handling and logging
- [ ] Implement async processing for long-running detections
- [ ] Write comprehensive API tests
- [ ] Add OpenAPI documentation

### PR Description Template:
```markdown
## Gap Detection API Endpoints

RESTful API for detecting and managing coordination gaps.

### Changes
- POST /api/v1/gaps/detect - Run gap detection
- GET /api/v1/gaps - List detected gaps
- GET /api/v1/gaps/{id} - Get specific gap
- GapService for business logic
- Request/response validation
- Filtering and pagination
- Async processing support
- Error handling
- OpenAPI documentation

### API Endpoints

#### **POST /api/v1/gaps/detect**

Detect coordination gaps across specified timeframe and sources.

**Request**:
```json
{
  "timeframe_days": 30,
  "sources": ["slack", "github"],
  "gap_types": ["duplicate_work"],
  "teams": ["engineering"],
  "min_impact_score": 0.6,
  "include_evidence": true
}
```

**Response**:
```json
{
  "gaps": [
    {
      "id": "gap_abc123",
      "type": "DUPLICATE_WORK",
      "title": "Two teams building OAuth integration",
      "impact_score": 0.89,
      "impact_tier": "CRITICAL",
      "teams_involved": ["platform-team", "auth-team"],
      "topic": "OAuth2 integration",
      "timeframe": {
        "start": "2024-12-01T00:00:00Z",
        "end": "2024-12-15T00:00:00Z",
        "overlap_days": 14
      },
      "evidence": [
        {
          "source": "slack",
          "channel": "#platform",
          "message": "Starting OAuth2 implementation...",
          "author": "alice@company.com",
          "timestamp": "2024-12-01T09:00:00Z",
          "relevance_score": 0.95
        },
        {
          "source": "slack",
          "channel": "#auth",
          "message": "Building OAuth support...",
          "author": "bob@company.com",
          "timestamp": "2024-12-01T14:20:00Z",
          "relevance_score": 0.92
        }
      ],
      "insight": "Platform and Auth teams independently implementing OAuth2. Platform started 4 hours before Auth. High overlap in scope.",
      "recommendation": "Connect alice@company.com and bob@company.com immediately. Consider consolidating under one team lead.",
      "estimated_cost": {
        "engineering_hours": 85,
        "dollar_value": 8500
      },
      "confidence": 0.87,
      "detected_at": "2024-12-19T10:30:00Z"
    }
  ],
  "metadata": {
    "total_gaps": 1,
    "critical_gaps": 1,
    "high_gaps": 0,
    "detection_time_ms": 3842,
    "messages_analyzed": 1250,
    "clusters_found": 15
  }
}
```

#### **GET /api/v1/gaps**

List all detected gaps with filtering and pagination.

**Query Parameters**:
- `gap_type`: Filter by type (duplicate_work, missing_context, etc.)
- `min_impact_score`: Minimum impact score (0-1)
- `teams`: Filter by team involvement
- `start_date`, `end_date`: Timeframe filter
- `page`, `limit`: Pagination

**Example**:
```bash
GET /api/v1/gaps?gap_type=duplicate_work&min_impact_score=0.7&limit=10
```

#### **GET /api/v1/gaps/{gap_id}**

Get detailed information about a specific gap.

**Response**: Full gap object with all evidence and metadata

### Request Validation

**Pydantic Schemas**:
```python
class GapDetectionRequest(BaseModel):
    timeframe_days: int = Field(default=30, ge=1, le=365)
    sources: List[str] = Field(default=["slack"])
    gap_types: List[str] = Field(default=["duplicate_work"])
    teams: Optional[List[str]] = None
    min_impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    include_evidence: bool = Field(default=True)

class GapResponse(BaseModel):
    id: str
    type: str
    title: str
    impact_score: float
    impact_tier: str
    teams_involved: List[str]
    topic: str
    # ... all fields
```

### Error Handling

**400 Bad Request**:
- Invalid timeframe
- Unknown gap type
- Invalid impact score range

**500 Internal Server Error**:
- Detection algorithm failure
- LLM API error
- Database error

**503 Service Unavailable**:
- LLM rate limit exceeded
- System overload

### Async Processing

For long-running detections (>10s), return job ID:

```bash
POST /api/v1/gaps/detect
→ 202 Accepted
{
  "job_id": "job_xyz789",
  "status": "processing",
  "estimated_time_seconds": 30
}

GET /api/v1/gaps/jobs/{job_id}
→ 200 OK (when complete)
```

### Testing
- [ ] Detection endpoint works end-to-end
- [ ] Request validation catches invalid input
- [ ] Filtering works correctly
- [ ] Pagination works
- [ ] Error handling covers all cases
- [ ] OpenAPI docs generated correctly
- [ ] Performance acceptable (<5s for 30-day scan)

### OpenAPI Documentation

Access at: http://localhost:8000/docs

Includes:
- Full request/response schemas
- Example requests
- Error response formats
- Authentication (future)
```

**Commit Messages:**
```
feat: create gap detection API endpoint
feat: add gap listing and retrieval endpoints
feat: implement GapService orchestration
feat: add request validation and error handling
test: add comprehensive gap API tests
docs: add OpenAPI documentation for gap endpoints
```

---

## Milestone 3G: Mock Data Scenarios for Duplicate Work
**Branch**: `feat/duplicate-work-scenarios`
**Time**: ~3-4 hours
**Files Changed**: ~6-8 files

### Changes:
```
scripts/
└── generate_mock_data.py        # Updated with gap scenarios
data/
└── scenarios/
    ├── duplicate_work/
    │   ├── oauth_scenario.json
    │   ├── api_redesign_scenario.json
    │   └── auth_migration_scenario.json
    └── README.md
tests/
├── fixtures/
│   └── gap_scenarios.py         # Test fixtures
└── test_integration/
    └── test_duplicate_detection.py
```

### Specific Tasks:
- [ ] Create realistic duplicate work scenarios
- [ ] Generate OAuth integration scenario (2 teams)
- [ ] Create API redesign scenario (platform + backend)
- [ ] Add auth migration scenario (security + platform)
- [ ] Include edge cases (false positives)
- [ ] Add collaboration scenarios (should NOT detect)
- [ ] Create scenario documentation
- [ ] Write integration tests using scenarios
- [ ] Add scenario visualization helpers

### PR Description Template:
```markdown
## Mock Data Scenarios for Duplicate Work

Realistic scenarios for testing and demonstrating gap detection.

### Changes
- 3 realistic duplicate work scenarios
- 2 edge case scenarios (false positives)
- 1 collaboration scenario (negative example)
- Scenario generation scripts
- Integration tests using scenarios
- Scenario documentation
- Visualization helpers

### Scenarios Included

#### **Scenario 1: OAuth Integration Duplication**

**Setup**:
- **Teams**: Platform Team, Auth Team
- **Timeframe**: Dec 1-15, 2024
- **Messages**: 24 messages across 2 Slack channels
- **Overlap**: 14 days of parallel work

**Story**:
```
Dec 1, 09:00 - Alice (Platform): "Starting OAuth2 implementation for our API gateway"
Dec 1, 14:20 - Bob (Auth): "We're building OAuth support for the new auth service"
Dec 3, 10:15 - Alice: "Finished research on OAuth flows, going with authorization code"
Dec 3, 15:30 - Bob: "Decided on authorization code flow for OAuth"
... [continuing parallel implementation]
Dec 10, 11:00 - Charlie: "Wait, are both teams working on OAuth??" [discovery]
```

**Expected Detection**:
- ✅ Should detect as duplicate work
- Impact Score: ~0.85 (HIGH)
- Teams: 2 (platform-team, auth-team)
- Estimated waste: ~80 hours
- Confidence: >0.8

#### **Scenario 2: API Redesign Duplication**

**Setup**:
- **Teams**: Platform Team, Backend Team
- **Timeframe**: Nov 15 - Dec 5, 2024
- **Messages**: 32 messages + 8 GitHub issues
- **Overlap**: 20 days

**Story**:
Both teams independently designing RESTful API restructuring with different approaches but solving same problems.

**Expected Detection**:
- ✅ Should detect as duplicate work
- Impact Score: ~0.78 (HIGH)
- Evidence includes Slack + GitHub cross-source

#### **Scenario 3: Auth Migration Duplication**

**Setup**:
- **Teams**: Security Team, Platform Team
- **Timeframe**: Dec 10-20, 2024
- **Messages**: 18 messages
- **Overlap**: 10 days

**Story**:
Both teams migrating from legacy auth to modern system, unaware of each other's efforts.

**Expected Detection**:
- ✅ Should detect as duplicate work
- Impact Score: ~0.72 (MEDIUM-HIGH)

### Edge Case Scenarios

#### **Edge Case 1: Similar Topics, Different Scope**

**Setup**: Two teams discussing "authentication" but different aspects
- Team A: User authentication
- Team B: Service-to-service auth (completely different)

**Expected**: Should NOT detect (LLM identifies different scopes)

#### **Edge Case 2: Sequential Work (Not Parallel)**

**Setup**: Team A implements OAuth in Nov, Team B in Jan
- No temporal overlap

**Expected**: Should NOT detect (no temporal overlap)

### Collaboration Scenario (Negative Example)

#### **Intentional Collaboration**

**Setup**:
- Platform team leads OAuth implementation
- Auth team contributing specific components
- Explicit cross-references in messages

**Messages Include**:
- "Working with @auth-team on token validation..."
- "Platform team is handling the main flow, we're doing refresh logic"
- Cross-channel mentions and coordination

**Expected**: Should NOT detect as duplicate work
- Cross-references present
- Clear division of labor
- Explicit collaboration

### Scenario Data Format

```json
{
  "scenario_id": "oauth_duplication_001",
  "name": "OAuth Integration Duplication",
  "description": "Two teams independently implementing OAuth2",
  "should_detect": true,
  "expected_impact_tier": "HIGH",
  "expected_confidence": 0.85,
  "messages": [
    {
      "id": "msg_001",
      "source": "slack",
      "channel": "#platform",
      "author": "alice@company.com",
      "team": "platform-team",
      "content": "Starting OAuth2 implementation for API gateway",
      "timestamp": "2024-12-01T09:00:00Z",
      "metadata": {
        "thread_id": "thread_001",
        "is_thread_root": true
      }
    },
    ...
  ],
  "expected_gap": {
    "type": "DUPLICATE_WORK",
    "teams": ["platform-team", "auth-team"],
    "topic": "OAuth2 integration",
    "overlap_days": 14
  }
}
```

### Testing
- [ ] All positive scenarios detected correctly
- [ ] Edge cases handled properly (no false positives)
- [ ] Collaboration scenario correctly excluded
- [ ] Impact scores in expected ranges
- [ ] Evidence collection works
- [ ] Cross-source scenarios (Slack + GitHub) work
- [ ] Scenarios load and seed correctly

### Usage

**Generate Scenarios**:
```bash
# Generate all scenarios
uv run python scripts/generate_mock_data.py --scenarios duplicate_work

# Specific scenario
uv run python scripts/generate_mock_data.py --scenario oauth_duplication

# Load into database
uv run python scripts/generate_mock_data.py --load
```

**Run Detection**:
```bash
# Detect gaps using scenarios
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -d '{"timeframe_days": 30}'

# Should detect 3 gaps with HIGH impact
```

**Visualize Scenario**:
```bash
# Generate timeline visualization
uv run python scripts/visualize_scenario.py oauth_duplication

# Outputs timeline showing parallel work
```
```

**Commit Messages:**
```
feat: create OAuth duplication scenario
feat: add API redesign and auth migration scenarios
feat: create edge case and collaboration scenarios
feat: add scenario loading and visualization
test: add integration tests using realistic scenarios
docs: document all gap detection scenarios
```

---

## Milestone 3H: Testing & Documentation
**Branch**: `feat/milestone-3-testing-docs`
**Time**: ~3-4 hours
**Files Changed**: ~12-15 files

### Changes:
```
tests/
├── test_detection/
│   ├── test_duplicate_work.py   # Enhanced
│   ├── test_clustering.py       # Enhanced
│   ├── test_impact_scoring.py   # Enhanced
│   └── test_integration.py      # Full pipeline tests
├── test_api/
│   └── test_gaps.py             # Enhanced
└── conftest.py                  # Updated fixtures
README.md                         # Updated with Milestone 3
docs/
├── GAP_DETECTION.md             # Detection methodology
├── ENTITY_EXTRACTION.md         # Entity extraction guide
└── API_EXAMPLES.md              # API usage examples
notebooks/
└── gap_analysis_demo.ipynb      # Interactive demo
```

### Specific Tasks:
- [ ] Add comprehensive integration tests
- [ ] Enhance unit tests for all components
- [ ] Add end-to-end detection tests
- [ ] Create gap detection documentation
- [ ] Write entity extraction guide
- [ ] Add API usage examples
- [ ] Create interactive Jupyter notebook demo
- [ ] Update README with Milestone 3
- [ ] Add troubleshooting guide
- [ ] Document LLM prompt engineering

### PR Description Template:
```markdown
## Milestone 3: Testing & Documentation

Comprehensive tests and documentation for gap detection system.

### Changes
- Integration tests for full detection pipeline
- Enhanced unit tests with edge cases
- End-to-end scenario tests
- Gap detection methodology docs
- Entity extraction guide
- API usage examples
- Interactive Jupyter demo
- Updated README
- Troubleshooting guide
- Prompt engineering documentation

### Test Coverage

**Component Coverage**:
- Entity extraction: 92%
- Semantic clustering: 95%
- Duplicate work detection: 93%
- Impact scoring: 90%
- Gap API: 88%
- LLM integration: 85% (mocked)
- **Overall Milestone 3**: 90%+

**Test Categories**:
- Unit tests: 145 tests
- Integration tests: 23 tests
- API tests: 18 tests
- Scenario tests: 6 tests
- **Total**: 192 tests

### Testing
- [ ] All tests pass: `pytest`
- [ ] Coverage meets threshold: `pytest --cov=src`
- [ ] Integration tests complete: `pytest tests/test_integration/`
- [ ] Scenario tests pass: `pytest tests/test_scenarios/`
- [ ] API tests pass: `pytest tests/test_api/test_gaps.py`
- [ ] End-to-end detection works with mock data

### Documentation Added

#### **GAP_DETECTION.md**
- Detection methodology overview
- Algorithm explanation (clustering → extraction → verification)
- Confidence scoring details
- Impact scoring breakdown
- When to use different gap types
- Tuning parameters guide
- Best practices

#### **ENTITY_EXTRACTION.md**
- Extraction methods (regex vs NLP)
- Entity types and examples
- Normalization strategies
- Handling ambiguity
- Performance optimization
- Testing entity extraction

#### **API_EXAMPLES.md**
- Complete API usage guide
- Request/response examples
- Filtering and pagination
- Error handling
- Rate limiting
- Integration patterns

#### **README Updates**
- Milestone 3 quickstart
- Gap detection examples
- Running detection locally
- Interpreting results
- Configuration guide

### Jupyter Notebook Demo

**gap_analysis_demo.ipynb**:
```python
# Interactive demonstration
1. Load mock scenarios
2. Run entity extraction
3. Visualize semantic clusters
4. Execute gap detection
5. Analyze results
6. Explore evidence
7. Understand impact scoring
```

Features:
- Step-by-step walkthrough
- Visualizations (clusters, timelines)
- Interactive parameter tuning
- Real API calls
- Result exploration

### End-to-End Test Example

```python
async def test_full_duplicate_work_detection():
    """Test complete detection pipeline with OAuth scenario."""

    # 1. Load scenario
    scenario = load_scenario("oauth_duplication_001")
    await seed_database(scenario.messages)

    # 2. Run detection
    response = await client.post("/api/v1/gaps/detect", json={
        "timeframe_days": 30,
        "gap_types": ["duplicate_work"],
        "min_impact_score": 0.6
    })

    # 3. Verify detection
    assert response.status_code == 200
    gaps = response.json()["gaps"]
    assert len(gaps) == 1

    gap = gaps[0]
    assert gap["type"] == "DUPLICATE_WORK"
    assert gap["impact_score"] >= 0.8
    assert "platform-team" in gap["teams_involved"]
    assert "auth-team" in gap["teams_involved"]
    assert gap["topic"] == "OAuth2 integration"
    assert len(gap["evidence"]) >= 5

    # 4. Verify impact details
    assert gap["estimated_cost"]["engineering_hours"] > 50
    assert gap["confidence"] > 0.7
```

### Performance Benchmarks

**Detection Pipeline**:
- Entity extraction: <20ms per message (p95)
- Clustering 1000 messages: <500ms
- LLM verification: <2s per gap
- Full detection (30 days): <5s (p95)

**API Latency**:
- POST /api/v1/gaps/detect: <5s (p95)
- GET /api/v1/gaps: <100ms (p95)
- GET /api/v1/gaps/{id}: <50ms (p95)

### Troubleshooting Guide

**Common Issues**:

1. **No gaps detected despite obvious duplication**
   - Check similarity threshold (default: 0.85)
   - Verify temporal overlap (min 3 days)
   - Review LLM verification confidence
   - Check team extraction accuracy

2. **Too many false positives**
   - Increase similarity threshold (0.85 → 0.90)
   - Raise min impact score filter
   - Adjust LLM confidence threshold
   - Review exclusion rules

3. **LLM API errors**
   - Check API key validity
   - Verify rate limits not exceeded
   - Review retry logic
   - Check network connectivity

4. **Slow detection performance**
   - Reduce timeframe_days
   - Enable caching for embeddings
   - Batch LLM calls
   - Optimize clustering parameters

### Example: Complete Detection Workflow

```bash
# 1. Start services
docker compose up -d

# 2. Load mock scenarios
uv run python scripts/generate_mock_data.py --scenarios duplicate_work --load

# 3. Run detection
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6,
    "include_evidence": true
  }'

# 4. View results
# Should detect 3 gaps with detailed evidence and recommendations

# 5. Explore in Jupyter
jupyter notebook notebooks/gap_analysis_demo.ipynb
```

### Next Steps

After Milestone 3:
- ✅ Understand gap detection fundamentals
- ✅ Can detect duplicate work end-to-end
- ✅ Ready for Milestone 4: Real Slack integration
- ✅ Foundation for additional gap types (Milestone 6)
```

**Commit Messages:**
```
test: add comprehensive integration tests
test: add end-to-end scenario tests
docs: create gap detection methodology guide
docs: add entity extraction documentation
docs: create API usage examples
docs: add interactive Jupyter demo notebook
docs: update README with Milestone 3 examples
```

---

## Summary: Milestone 3 PRs

| PR | Branch | Time | Files | Focus |
|----|--------|------|-------|-------|
| **3A** | `feat/entity-extraction` | 3-4h | ~8 | People, teams, projects |
| **3B** | `feat/semantic-clustering` | 3-4h | ~8 | DBSCAN clustering |
| **3C** | `feat/claude-api-integration` | 3-4h | ~9 | LLM verification |
| **3D** | `feat/duplicate-work-detection` | 4-5h | ~10 | Core algorithm |
| **3E** | `feat/impact-scoring` | 2-3h | ~7 | Cost estimation |
| **3F** | `feat/gap-detection-endpoint` | 3-4h | ~9 | REST API |
| **3G** | `feat/duplicate-work-scenarios` | 3-4h | ~8 | Mock scenarios |
| **3H** | `feat/milestone-3-testing-docs` | 3-4h | ~15 | Tests, docs |

**Total**: 24-35 hours across 8 PRs

## PR Workflow

### For Each PR:

1. **Create branch from main**
```bash
git checkout main
git pull origin main
git checkout -b feat/entity-extraction
```

2. **Make changes & commit frequently**
```bash
git add .
git commit -m "feat: implement entity extraction for people and teams"
# ... more commits ...
```

3. **Push and create PR**
```bash
git push origin feat/entity-extraction
# Create PR on GitHub
```

4. **Self-review checklist**
- [ ] Code runs without errors
- [ ] Tests pass with good coverage
- [ ] LLM integration works (or properly mocked)
- [ ] Commit messages are clear
- [ ] PR description is complete
- [ ] Documentation updated

5. **Merge and tag (if completing milestone)**
```bash
git checkout main
git merge feat/entity-extraction
git tag v0.3.0  # After PR 3H
git push origin main --tags
```

## Dependencies Between PRs

```
3A (Entity Extraction) ──→ 3B (Clustering) ──→ 3D (Detection)
         ↓                                            ↓
    3C (Claude API) ─────────────────────────→ 3D (Detection)
                                                      ↓
                                               3E (Impact) ──→ 3F (API)
                                                      ↓              ↓
                                               3G (Scenarios) ──→ 3H (Testing)
```

## Key Technical Concepts Demonstrated

After Milestone 3, you can discuss:

### AI/ML Techniques
- **Semantic Clustering**: DBSCAN with cosine similarity for pattern detection
- **Entity Extraction**: NLP and pattern-based extraction from unstructured text
- **LLM Reasoning**: Using Claude for verification and insight generation
- **Multi-signal Scoring**: Combining multiple features for impact assessment

### System Design
- **Multi-stage Pipeline**: Cluster → Extract → Verify → Score
- **Confidence Scoring**: Probabilistic reasoning about detections
- **Evidence-based Analysis**: Collecting and ranking supporting evidence
- **Cost Estimation**: Quantifying organizational impact

### LLM Integration
- **Prompt Engineering**: Structured prompts for gap verification
- **Structured Output**: JSON schema validation from LLM responses
- **Error Handling**: Retries, rate limiting, graceful degradation
- **Cost Management**: Token counting and quota tracking

## Interview Questions You Can Answer

**Q: How do you detect duplicate work across teams?**
> Multi-stage pipeline: (1) Semantic clustering to group similar discussions, (2) Entity extraction to identify teams, (3) Temporal overlap check, (4) LLM verification to confirm actual duplication vs. collaboration. Confidence scoring combines all signals.

**Q: How do you prevent false positives in gap detection?**
> Exclusion rules (cross-references = collaboration), LLM verification to understand context, confidence thresholds (>0.7), and impact scoring to prioritize. Also validate with realistic scenarios during testing.

**Q: How do you integrate LLMs into a production system?**
> Structured prompts with JSON schemas, retry logic with exponential backoff, rate limiting, token counting for cost control, graceful degradation on failures, and comprehensive error handling. Mock LLM responses in tests.

**Q: How do you quantify the impact of coordination failures?**
> Multi-signal scoring: team size (people affected), time investment (hours wasted), project criticality (business importance), velocity impact (blocking work), and duplicate effort ratio. Normalize to 0-1 scale and classify into tiers.

## Benefits of This Approach

✅ **AI-Native Development** - Real LLM integration with Claude
✅ **Practical ML** - Clustering, entity extraction, confidence scoring
✅ **Production Patterns** - Error handling, cost management, monitoring
✅ **Realistic Use Case** - Solves real organizational problem
✅ **End-to-End System** - Detection → Verification → Impact → Action
✅ **Interview-Ready** - Can discuss AI system design confidently

## Success Criteria for Milestone 3

After completing all PRs, you should have:

✅ **Working System**
- Entity extraction from messages
- Semantic clustering operational
- Claude API integrated and working
- Duplicate work detection functional
- Impact scoring calibrated
- REST API endpoints ready

✅ **Detectable Gaps**
- Can detect OAuth scenario (>0.8 confidence)
- Can detect API redesign scenario
- Can detect auth migration scenario
- Properly excludes collaboration scenarios
- No false positives on edge cases

✅ **Production-Quality Code**
- 90%+ test coverage
- Performance targets met (<5s detection)
- Comprehensive documentation
- Realistic mock scenarios
- Clean architecture

✅ **Interview Readiness**
- Can explain gap detection pipeline
- Understand LLM integration patterns
- Discuss impact scoring methodology
- Show real working system
- Prepared for behavioral questions about AI projects

---

**Last Updated**: December 2024
**Dependencies**: Milestone 1 & 2 completed
**Next**: Milestone 4 - Real Slack Integration
