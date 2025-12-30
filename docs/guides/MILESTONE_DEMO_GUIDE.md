# Milestone Demo Guide

A practical guide for demonstrating the Coordination Gap Detector's capabilities across Milestones 1-3.

## Table of Contents

1. [Overview](#overview)
2. [Demo Setup](#demo-setup)
3. [Milestone 1: Foundation & Search](#milestone-1-foundation--search)
4. [Milestone 2: Ranking & Search Quality](#milestone-2-ranking--search-quality)
5. [Milestone 3: Gap Detection](#milestone-3-gap-detection)
6. [Complete Demo Script](#complete-demo-script)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide demonstrates the complete system evolution through three major milestones:

**Milestone 1**: Basic search infrastructure with semantic similarity
**Milestone 2**: Advanced ranking with BM25, hybrid search, and evaluation metrics
**Milestone 3**: AI-powered coordination gap detection with LLM reasoning

### What You'll Demonstrate

- ✅ **Semantic search** over organizational communications
- ✅ **Hybrid search** combining semantic + keyword matching
- ✅ **Ranking quality** with industry-standard metrics (MRR, NDCG)
- ✅ **Gap detection** identifying duplicate work across teams
- ✅ **Impact scoring** quantifying organizational cost
- ✅ **LLM integration** with Claude for verification and insights

### Time Commitment

- **Quick Demo** (10 minutes): Core capabilities from each milestone
- **Standard Demo** (20 minutes): Detailed walkthrough with explanations
- **Deep Dive** (45 minutes): Architecture, algorithms, and code review

---

## Demo Setup

### Prerequisites

```bash
# 1. Ensure all services are running
docker compose up -d

# 2. Wait for services to be healthy (30 seconds)
docker compose ps

# Expected output:
# NAME                          STATUS
# elasticsearch                 Up (healthy)
# postgres                      Up (healthy)
# redis                         Up (healthy)
# chromadb                      Up (healthy)
# api                           Up (healthy)
```

### Verify System Health

```bash
# Check API health endpoint
curl http://localhost:8000/health | jq

# Expected output:
# {
#   "status": "healthy",
#   "services": {
#     "postgres": "connected",
#     "redis": "connected",
#     "chromadb": "connected",
#     "elasticsearch": "connected"
#   }
# }
```

### Load Demo Data

```bash
# Clear existing data and load OAuth duplication scenario
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios oauth_duplication \
  --clear

# Verify data loaded
docker compose exec postgres psql -U coordination_user -d coordination \
  -c "SELECT COUNT(*) FROM messages;"

# Expected: 20+ messages
```

---

## Milestone 1: Foundation & Search

**What Was Built**: Core search infrastructure with semantic similarity using ChromaDB.

### Demo 1.1: Basic Semantic Search

**Goal**: Show that the system can find semantically similar messages.

```bash
# Search for OAuth-related discussions
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "limit": 5,
    "threshold": 0.7
  }' | jq '.results[] | {content: .content[:80], score: .score}'
```

**Expected Output**:
```json
{
  "content": "Starting work on OAuth2 integration for our API gateway",
  "score": 0.94
}
{
  "content": "We're building OAuth support for the new auth service",
  "score": 0.91
}
{
  "content": "Finished research on OAuth flows, going with authorization code",
  "score": 0.88
}
```

**Key Points to Highlight**:
- ✅ Semantic understanding (finds "OAuth" even if query says "authentication")
- ✅ Relevance scoring (0-1 scale)
- ✅ Vector embeddings enable similarity search

### Demo 1.2: Cross-Source Search

**Goal**: Show search works across different communication channels.

```bash
# Search across Slack channels
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "sources": ["slack"],
    "limit": 5
  }' | jq '.results[] | {source: .source, channel: .channel, author: .author, score: .score}'
```

**Expected Output**:
```json
{
  "channel": "#platform",
  "author": "alice@company.com",
  "score": 0.30
}
{
  "channel": "#auth-team",
  "author": "diana@company.com",
  "score": 0.28
}
{
  "channel": "#platform",
  "author": "alice@company.com",
  "score": 0.21
}
```

**Key Points to Highlight**:
- ✅ Multi-channel search (results from both #platform and #auth-team channels)
- ✅ Author attribution across different teams
- ✅ Source filtering (only Slack results)

### Demo 1.3: ChromaDB Vector Store

**Architecture Explanation**:

```
User Query: "OAuth implementation"
      ↓
1. Generate embedding (768-dim vector)
      ↓
2. ChromaDB cosine similarity search
      ↓
3. Return top-k similar messages
      ↓
4. Format and return results
```

**Show the Code** (if doing deep dive):

```python
# src/db/vector_store.py
def search(self, query: str, limit: int = 10):
    # Generate query embedding
    query_embedding = self.embedding_model.encode(query)

    # Search ChromaDB
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=limit
    )

    return results
```

---

## Milestone 2: Ranking & Search Quality

**What Was Built**: Advanced ranking algorithms, BM25 scoring, hybrid search, and evaluation metrics.

### Demo 2.1: BM25 Keyword Search

**Goal**: Show keyword-based ranking with term frequency and IDF.

```bash
# Search using BM25 (keyword matching)
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation authorization code",
    "ranking_strategy": "bm25",
    "limit": 5
  }' | jq '.results[] | {content: .content[:80], bm25_score: .score}'
```

**Expected Output**:
```json
{
  "content": "Decided on authorization code flow for OAuth",
  "bm25_score": 12.45
}
{
  "content": "OAuth implementation using authorization code grant type",
  "bm25_score": 11.23
}
```

**Key Points to Highlight**:
- ✅ **BM25 scoring**: Probabilistic ranking function
- ✅ **Term frequency saturation**: k1 parameter prevents over-weighting repeated terms
- ✅ **Length normalization**: b parameter adjusts for document length
- ✅ **IDF weighting**: Rare terms (like "OAuth") rank higher than common terms (like "the")

**BM25 Formula Explanation**:
```
score(D,Q) = Σ IDF(qi) · [f(qi,D) · (k1 + 1)] / [f(qi,D) + k1 · (1 - b + b · |D|/avgdl)]

Where:
- f(qi,D) = term frequency in document
- k1 = 1.5 (term frequency saturation)
- b = 0.75 (length normalization)
- IDF = log(N / df) (inverse document frequency)
```

### Demo 2.2: Hybrid Search (Semantic + BM25)

**Goal**: Show how combining semantic and keyword search improves results.

```bash
# Search using hybrid RRF (Reciprocal Rank Fusion)
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth security best practices",
    "ranking_strategy": "hybrid_rrf",
    "limit": 5
  }' | jq '.results[] | {
    content: .content[:60],
    score: .score,
    ranking_details: .ranking_details
  }'
```

**Expected Output**:
```json
{
  "content": "OAuth security considerations for authorization flow",
  "score": 0.89,
  "ranking_details": {
    "semantic_score": 0.91,
    "bm25_score": 8.5,
    "semantic_rank": 1,
    "bm25_rank": 2,
    "fusion_method": "rrf"
  }
}
```

**Key Points to Highlight**:
- ✅ **Reciprocal Rank Fusion (RRF)**: Combines rankings, not scores
- ✅ **Formula**: `score(d) = Σ 1/(k + rank_i(d))` where k=60
- ✅ **Benefits**: No score normalization needed, robust to distribution differences
- ✅ **Best of both worlds**: Semantic understanding + keyword precision

**RRF Explanation**:
```
Document appears at:
- Rank 1 in semantic results → 1/(60+1) = 0.0164
- Rank 3 in BM25 results    → 1/(60+3) = 0.0159
- Combined RRF score        → 0.0323

Higher combined score = better overall ranking
```

### Demo 2.3: Ranking Metrics Evaluation

**Goal**: Show how we measure search quality with industry-standard metrics.

```bash
# Run evaluation on test queries
docker compose exec api python scripts/evaluate_ranking.py \
  --strategies semantic,bm25,hybrid_rrf \
  --metrics mrr,ndcg,precision
```

**Expected Output**:
```
Strategy Comparison:
┌─────────────────┬───────┬──────────┬──────────┐
│ Strategy        │  MRR  │ NDCG@10  │  P@5     │
├─────────────────┼───────┼──────────┼──────────┤
│ semantic        │ 0.68  │  0.72    │  0.64    │
│ bm25            │ 0.62  │  0.65    │  0.58    │
│ hybrid_rrf      │ 0.74  │  0.79    │  0.72    │
└─────────────────┴───────┴──────────┴──────────┘

Best: hybrid_rrf (+8.8% MRR improvement vs semantic)
```

**Metrics Explained**:

**MRR (Mean Reciprocal Rank)**:
- Measures rank of first relevant result
- Formula: `MRR = (1/|Q|) · Σ 1/rank_i`
- Range: [0, 1], higher is better
- **Use case**: "Find the best answer" (navigational queries)

**NDCG@10 (Normalized Discounted Cumulative Gain)**:
- Graded relevance with position discount
- Formula: `DCG@k = Σ (2^rel_i - 1) / log2(i + 1)`
- Normalized to [0, 1]
- **Use case**: "Rank all relevant results" (informational queries)

**Precision@5**:
- Percentage of relevant items in top 5
- Formula: `P@5 = (relevant in top 5) / 5`
- **Use case**: "How many results are useful?"

**Key Points to Highlight**:
- ✅ Hybrid search outperforms single methods across all metrics
- ✅ Industry-standard evaluation (same metrics Google/Bing use)
- ✅ Quantifiable improvements (+8.8% MRR)
- ✅ Graded relevance (not just binary relevant/not-relevant)

### Demo 2.4: Feature Engineering

**Goal**: Show the 40+ ranking features extracted for each result.

```bash
# Search with feature details
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "ranking_strategy": "hybrid_rrf",
    "include_features": true,
    "limit": 3
  }' | jq '.results[0].features'
```

**Expected Output**:
```json
{
  "semantic_score": 0.92,
  "bm25_score": 11.5,
  "exact_match": 0.0,
  "term_coverage": 0.75,
  "recency": 0.85,
  "thread_depth": 0.60,
  "participant_count": 0.40,
  "author_seniority": 0.70,
  "channel_importance": 0.80
}
```

**Feature Categories**:

1. **Query-Document Similarity** (6 features)
   - semantic_score, bm25_score, exact_match, term_coverage

2. **Temporal Features** (5 features)
   - recency, activity_burst, temporal_relevance

3. **Engagement Features** (6 features)
   - thread_depth, participant_count, reaction_count

4. **Source Authority** (4 features)
   - author_seniority, channel_importance, team_influence

**Key Points to Highlight**:
- ✅ Multi-signal ranking (not just semantic similarity)
- ✅ Normalized to [0, 1] range for combining
- ✅ Extensible architecture (easy to add new features)
- ✅ Foundation for learning-to-rank (future work)

---

## Milestone 3: Gap Detection

**What Was Built**: AI-powered coordination gap detection with Claude API integration, entity extraction, clustering, and impact scoring.

### Demo 3.1: Entity Extraction

**Goal**: Show how the system extracts people, teams, and projects from messages.

```bash
# Extract entities from sample message
curl -X POST http://localhost:8000/api/v1/analysis/entities \
  -H "Content-Type: application/json" \
  -d '{
    "content": "@alice and @bob from @platform-team are working on OAuth integration",
    "channel": "#platform",
    "author": "charlie@company.com"
  }' | jq
```

**Expected Output**:
```json
{
  "people": [
    "alice@company.com",
    "bob@company.com"
  ],
  "teams": [
    "platform-team"
  ],
  "projects": [
    "OAuth"
  ],
  "topics": [
    "integration"
  ],
  "confidence": 0.92
}
```

**Key Points to Highlight**:
- ✅ **Person extraction**: @mentions, emails, names
- ✅ **Team detection**: @team mentions, channel inference
- ✅ **Project identification**: Technical terms, acronyms, feature names
- ✅ **Normalization**: @alice → alice@company.com

### Demo 3.2: Semantic Clustering

**Goal**: Show how similar discussions are grouped together.

```bash
# View clusters for OAuth discussions
curl -X POST http://localhost:8000/api/v1/analysis/clusters \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "similarity_threshold": 0.85,
    "min_cluster_size": 2
  }' | jq '.clusters[] | {
    id: .cluster_id,
    size: .size,
    topic: .label,
    similarity: .avg_similarity,
    teams: .teams
  }'
```

**Expected Output**:
```json
{
  "id": "cluster_001",
  "size": 12,
  "topic": "OAuth2 implementation",
  "similarity": 0.89,
  "teams": ["platform-team", "auth-team"]
}
{
  "id": "cluster_002",
  "size": 8,
  "topic": "API authentication",
  "similarity": 0.87,
  "teams": ["backend-team"]
}
```

**Clustering Algorithm - DBSCAN**:
```
Why DBSCAN?
✅ No need to specify cluster count upfront
✅ Handles noise and outliers
✅ Finds arbitrarily shaped clusters
✅ Works well with cosine similarity

Parameters:
- eps = 0.15 (similarity threshold)
- min_samples = 2 (minimum cluster size)
- metric = cosine distance
```

**Key Points to Highlight**:
- ✅ Automatically groups similar technical discussions
- ✅ Multi-team clusters indicate potential coordination gaps
- ✅ Temporal windowing (only cluster recent messages)
- ✅ Quality metrics (silhouette score, density)

### Demo 3.3: Duplicate Work Detection (Full Pipeline)

**Goal**: Demonstrate end-to-end gap detection identifying parallel efforts.

```bash
# Run gap detection
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "sources": ["slack"],
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6,
    "include_evidence": true
  }' | jq
```

**Expected Output**:
```json
{
  "gaps": [
    {
      "id": "gap_abc123",
      "type": "DUPLICATE_WORK",
      "title": "Two teams building OAuth integration simultaneously",
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
          "message": "Starting OAuth2 implementation for API gateway",
          "author": "alice@company.com",
          "timestamp": "2024-12-01T09:00:00Z",
          "team": "platform-team",
          "relevance_score": 0.95
        },
        {
          "source": "slack",
          "channel": "#auth",
          "message": "We're building OAuth support for auth service",
          "author": "bob@company.com",
          "timestamp": "2024-12-01T14:20:00Z",
          "team": "auth-team",
          "relevance_score": 0.92
        }
      ],
      "insight": "Platform and Auth teams are independently implementing OAuth2. Platform team started 4 hours before Auth opened their work. High overlap in technical scope - both implementing authorization code flow.",
      "recommendation": "Connect alice@company.com and bob@company.com immediately. Consider consolidating efforts with one team leading and the other contributing specific components (e.g., token validation, refresh logic).",
      "estimated_cost": {
        "engineering_hours": 85,
        "dollar_value": 8500,
        "explanation": "2 teams × ~40 hours each + coordination overhead"
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

**Detection Pipeline Explained**:

```
Stage 1: Semantic Clustering
  ↓
Messages grouped by similarity (threshold: 0.85)
15 clusters found

Stage 2: Team Detection
  ↓
Extract teams from each cluster
Cluster #3 has 2+ teams → potential gap

Stage 3: Temporal Overlap Check
  ↓
Platform: Dec 1-15
Auth: Dec 1-15
Overlap: 14 days ✅

Stage 4: LLM Verification (Claude API)
  ↓
Prompt: "Are these teams duplicating work?"
Response: {is_duplicate: true, confidence: 0.87}

Stage 5: Impact Scoring
  ↓
Team size: 0.9
Time investment: 0.85
Project criticality: 0.9
→ Overall impact: 0.89 (CRITICAL)

Stage 6: Evidence Collection
  ↓
Rank messages by relevance
Select top evidence from each team
Include timestamps, authors, channels
```

**Key Points to Highlight**:
- ✅ **Multi-stage pipeline**: Cluster → Extract → Verify → Score
- ✅ **LLM reasoning**: Claude verifies actual duplication vs collaboration
- ✅ **Evidence-based**: Provides concrete examples from each team
- ✅ **Actionable insights**: Specific recommendations for resolution
- ✅ **Cost quantification**: Estimates organizational waste ($8,500)

### Demo 3.4: Impact Scoring Breakdown

**Goal**: Explain how organizational cost is calculated.

```bash
# Get detailed impact breakdown for a gap
curl http://localhost:8000/api/v1/gaps/gap_abc123 | jq '.impact_breakdown'
```

**Expected Output**:
```json
{
  "impact_score": 0.89,
  "impact_tier": "CRITICAL",
  "breakdown": {
    "team_size_score": 0.90,
    "time_investment_score": 0.85,
    "project_criticality_score": 0.90,
    "velocity_impact_score": 0.80,
    "duplicate_effort_score": 0.95
  },
  "details": {
    "people_affected": 8,
    "estimated_hours": 85,
    "timespan_days": 14,
    "messages_analyzed": 32,
    "commits_found": 12,
    "criticality_tags": ["roadmap_item", "security"]
  },
  "estimated_cost": {
    "engineering_hours": 85,
    "dollar_value": 8500,
    "hourly_rate": 100,
    "calculation": "85 hours × $100/hour = $8,500",
    "note": "This represents organizational waste, not Claude API cost"
  }
}
```

**Impact Scoring Formula**:
```python
impact_score = (
    0.25 * team_size_score +           # How many people affected?
    0.25 * time_investment_score +     # How much time wasted?
    0.20 * project_criticality_score + # How important is this?
    0.15 * velocity_impact_score +     # What's blocked?
    0.15 * duplicate_effort_score      # How much overlap?
)
```

**Impact Tiers**:
- 🔴 **CRITICAL (0.8-1.0)**: Multiple large teams, 100+ hours, roadmap impact
- 🟠 **HIGH (0.6-0.8)**: 5-10 people, 40-100 hours, important projects
- 🟡 **MEDIUM (0.4-0.6)**: 2-5 people, 10-40 hours, moderate importance
- 🟢 **LOW (0.0-0.4)**: Small scope, <10 hours, low criticality

**Key Points to Highlight**:
- ✅ **Multi-signal scoring**: Combines 5 different factors
- ✅ **Quantifiable cost**: Estimates dollars wasted, not just "high impact"
- ✅ **Prioritization**: Tier system helps focus on critical gaps first
- ✅ **Transparent**: Shows exactly how score is calculated

### Demo 3.5: LLM Integration with Claude

**Goal**: Show how Claude API is used for verification and insights.

**Prompt Template Used**:
```
You are analyzing organizational communication to detect coordination gaps.

Given these clustered messages about "OAuth2 integration", determine if they represent duplicate work:

Messages:
1. [Dec 1, 09:00] alice@company.com (#platform): "Starting OAuth2 implementation..."
2. [Dec 1, 14:20] bob@company.com (#auth): "We're building OAuth support..."
...

Teams involved: platform-team, auth-team
Timeframe: Dec 1-15, 2024 (14 days overlap)

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
  "overlap_ratio": 0-1,
  "recommendation": "action to take"
}
```

**Claude Response**:
```json
{
  "is_duplicate": true,
  "confidence": 0.87,
  "reasoning": "Both teams are independently implementing OAuth2 authorization code flow for their respective services (API gateway vs auth service). They started within 5 hours of each other and show identical technical decisions (authorization code flow choice, token handling). No cross-references or coordination messages detected.",
  "evidence": [
    "Both chose authorization code flow independently (Dec 3)",
    "Both implementing token validation separately (Dec 5-8)",
    "No mentions of collaboration or awareness of other team's work",
    "Overlapping scope in refresh token logic"
  ],
  "overlap_ratio": 0.75,
  "recommendation": "Immediate coordination needed. Platform team started first - suggest they lead the OAuth implementation with Auth team contributing specific security components (token validation, security auditing). Consolidate duplicate work on refresh token logic."
}
```

**Key Points to Highlight**:
- ✅ **Structured prompts**: Clear instructions and expected output format
- ✅ **Context-aware reasoning**: Uses message content, timing, teams
- ✅ **Explainable AI**: Provides reasoning, not just yes/no
- ✅ **Actionable recommendations**: Specific next steps
- ✅ **Production-ready**: Retry logic, rate limiting, error handling

**LLM Integration Architecture**:
```
┌─────────────────────────────────────┐
│  Gap Detection Service              │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Claude API Client                  │
│  - Retry logic (exponential backoff)│
│  - Rate limiting (quota management) │
│  - Token counting (cost tracking)   │
│  - Structured output parsing        │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Anthropic Claude API               │
│  Model: claude-sonnet-4-5           │
└─────────────────────────────────────┘
```

---

## Complete Demo Script

A 20-minute walkthrough demonstrating the full system evolution.

### Setup (2 minutes)

```bash
# Start all services
docker compose up -d

# Wait for health check
sleep 30

# Verify system health
curl http://localhost:8000/health | jq '.status'

# Load OAuth duplication scenario
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios oauth_duplication \
  --clear

echo "✅ System ready for demo"
```

### Part 1: Milestone 1 - Basic Search (5 minutes)

**Narrative**: "Let's start with basic semantic search. The system can understand meaning, not just keywords."

```bash
# 1.1: Semantic search
echo "=== Searching for 'OAuth implementation' ==="
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "limit": 3
  }' | jq '.results[] | {content: .content[:60], score: .score}'

# Key point: Shows semantic similarity in action
```

**What to say**:
> "Notice how it finds messages about OAuth even though they use different phrasing. That's semantic search - it understands meaning through vector embeddings, not just keyword matching."

### Part 2: Milestone 2 - Advanced Ranking (6 minutes)

**Narrative**: "Now let's see how we improved search quality with advanced ranking algorithms."

```bash
# 2.1: Compare search strategies
echo "=== BM25 Keyword Search ==="
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth authorization code",
    "ranking_strategy": "bm25",
    "limit": 3
  }' | jq '.results[] | {content: .content[:60], bm25_score: .score}'

echo ""
echo "=== Hybrid Search (Semantic + BM25) ==="
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth authorization code",
    "ranking_strategy": "hybrid_rrf",
    "limit": 3
  }' | jq '.results[] | {
    content: .content[:50],
    score: .score,
    semantic_rank: .ranking_details.semantic_rank,
    bm25_rank: .ranking_details.bm25_rank
  }'
```

**What to say**:
> "BM25 is a probabilistic ranking function - it's better than simple TF-IDF because it handles term frequency saturation and document length normalization. But the real power comes from combining it with semantic search using Reciprocal Rank Fusion. Notice how the hybrid results combine the best of both worlds."

```bash
# 2.2: Show evaluation metrics
echo "=== Search Quality Metrics ==="
docker compose exec api python scripts/evaluate_ranking.py \
  --strategies hybrid_rrf \
  --quick
```

**What to say**:
> "We measure search quality with the same metrics used by Google and Bing: MRR for finding the best result, NDCG for overall ranking quality. Our hybrid approach achieves 0.74 MRR and 0.79 NDCG@10 - that's production-quality search."

### Part 3: Milestone 3 - Gap Detection (7 minutes)

**Narrative**: "Now for the interesting part - using AI to detect coordination failures."

```bash
# 3.1: Entity extraction
echo "=== Entity Extraction ==="
curl -X POST http://localhost:8000/api/v1/analysis/entities \
  -H "Content-Type: application/json" \
  -d '{
    "content": "@alice from @platform-team is working on OAuth integration"
  }' | jq '{people, teams, projects}'
```

**What to say**:
> "First, we extract entities - people, teams, and projects - from messages. This helps us understand who's working on what."

```bash
# 3.2: Clustering
echo "=== Semantic Clustering ==="
curl -X POST http://localhost:8000/api/v1/analysis/clusters \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "similarity_threshold": 0.85
  }' | jq '.clusters[] | {topic: .label, size: .size, teams: .teams}'
```

**What to say**:
> "We cluster similar discussions using DBSCAN. Notice this cluster has two teams - that's a red flag for potential duplicate work."

```bash
# 3.3: Full gap detection
echo "=== Detecting Coordination Gaps ==="
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6,
    "include_evidence": true
  }' | jq '{
    total_gaps: .metadata.total_gaps,
    gap: .gaps[0] | {
      type,
      title,
      impact_score,
      impact_tier,
      teams_involved,
      confidence,
      estimated_cost
    }
  }'
```

**What to say**:
> "The system detected duplicate work between Platform and Auth teams. Impact score of 0.89 means this is critical - we estimate $8,500 in wasted engineering effort. The confidence is 0.87 because Claude verified that both teams are actually solving the same problem, not collaborating."

```bash
# 3.4: Show evidence
echo "=== Evidence and Recommendations ==="
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"]
  }' | jq '.gaps[0] | {
    evidence: .evidence[0:2] | .[] | {team, message: .message[:60], timestamp},
    insight: .insight,
    recommendation: .recommendation
  }'
```

**What to say**:
> "Here's the evidence: Platform team started at 9 AM, Auth team at 2:20 PM the same day. Both implementing OAuth, no cross-references. Claude's recommendation: connect the teams immediately and consolidate under one lead."

---

## Troubleshooting

### Services Not Starting

```bash
# Check logs
docker compose logs api
docker compose logs elasticsearch

# Common fixes
docker compose down -v  # Clear volumes
docker compose up -d --build  # Rebuild images
```

### No Search Results

```bash
# Verify data exists
docker compose exec postgres psql -U user -d coordination \
  -c "SELECT COUNT(*) FROM messages;"

# Reload data
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios oauth_duplication \
  --clear
```

### Gap Detection Returns Empty

```bash
# Check Claude API key
docker compose exec api env | grep ANTHROPIC_API_KEY

# Lower detection threshold
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -d '{"min_impact_score": 0.3}'

# Check clustering found multi-team clusters
curl -X POST http://localhost:8000/api/v1/analysis/clusters \
  -d '{"timeframe_days": 30}' | jq '.clusters[] | select(.teams | length > 1)'
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Increase Docker memory (Settings → Resources → Memory: 8GB+)

# Reduce timeframe for faster detection
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -d '{"timeframe_days": 7}'
```

---

## Demo Variations

### Quick Demo (10 minutes)

1. ✅ Semantic search (2 min)
2. ✅ Hybrid search comparison (3 min)
3. ✅ Gap detection with one example (5 min)

### Standard Demo (20 minutes)

Follow the complete demo script above.

### Deep Dive (45 minutes)

Add:
- Code walkthrough (detection algorithm, BM25 implementation)
- Architecture diagrams
- Test coverage demonstration
- Jupyter notebook exploration
- Database schema review
- LLM prompt engineering discussion

---

## Key Talking Points

### Technical Depth

**Information Retrieval**:
- "We implement BM25, the same algorithm used by Elasticsearch and Lucene"
- "Hybrid search with RRF is production-grade, not a toy implementation"
- "We measure quality with industry-standard metrics: MRR, NDCG"

**AI/ML Integration**:
- "Entity extraction uses both pattern-based and NLP approaches"
- "DBSCAN clustering doesn't require knowing cluster count upfront"
- "Claude integration has production patterns: retry logic, rate limiting, structured output"

**System Design**:
- "Multi-stage pipeline: cluster → extract → verify → score"
- "Three-database architecture: Postgres for structured data, ChromaDB for vectors, Elasticsearch for full-text search"
- "Async processing with FastAPI for performance"

### Business Value

**Measurable Impact**:
- "Detects coordination failures costing thousands in wasted effort"
- "Quantifies impact with dollar estimates, not just 'high priority'"
- "Provides actionable recommendations, not just alerts"

**Production Quality**:
- "85%+ test coverage across all components"
- "Performance targets: <5s gap detection, <200ms search"
- "Comprehensive error handling and monitoring"

---

## Next Steps After Demo

1. **Show the codebase**: Walk through key files
2. **Discuss architecture**: Three-database design, async processing
3. **Review tests**: Unit, integration, end-to-end coverage
4. **Explore extensibility**: How to add new gap types, ranking features
5. **Production considerations**: Scaling, monitoring, cost management

---

**Last Updated**: December 2024
**Milestones**: 1, 2, 3 (Complete)
**Next**: Milestone 4 - Real Slack Integration
