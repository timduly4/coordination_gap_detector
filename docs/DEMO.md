# Coordination Gap Detector - Interactive Demo

Welcome! This guide provides a hands-on walkthrough of the Coordination Gap Detector's key features. In 15 minutes, you'll see how it identifies duplicate work, enables semantic search, and provides actionable insights.

## What You'll Learn

By the end of this demo, you'll understand how to:
- ✅ Set up and run the system
- ✅ Generate realistic mock collaboration data
- ✅ Search across conversations using semantic similarity
- ✅ Detect coordination gaps (duplicate work)
- ✅ Interpret gap detection results and recommendations

## Prerequisites

- Docker and Docker Compose installed
- ~5GB free disk space
- 10-15 minutes of time

**Optional but helpful:**
- `jq` for pretty JSON formatting: `brew install jq` (macOS) or `apt-get install jq` (Linux)
- `curl` for API testing (usually pre-installed)

## Part 1: Setup (2 minutes)

### 1.1 Start the System

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/timduly4/coordination_gap_detector.git
cd coordination_gap_detector

# Start all services
docker compose up -d

# Check that all services are running
docker compose ps
```

**Expected output:** You should see 5 services running:
- `coordination-postgres` (database)
- `coordination-redis` (cache)
- `coordination-chromadb` (vector store)
- `coordination-elasticsearch` (search engine)
- `coordination-api` (main application)

### 1.2 Verify Health

```bash
# Check API health
curl http://localhost:8000/health

# Detailed health check (shows all services)
curl http://localhost:8000/health/detailed | jq
```

**Expected:** `"status": "healthy"` for all services.

### 1.3 Run Database Migrations

```bash
docker compose exec api alembic upgrade head
```

**What this does:** Sets up the PostgreSQL database schema for messages, sources, and coordination gaps.

---

## Part 2: Load Demo Data (2 minutes)

### 2.1 Generate Mock Conversations

The system includes realistic mock Slack conversations that simulate duplicate work scenarios.

```bash
# Generate OAuth duplication scenario
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios oauth_duplication \
  --clear
```

**What this creates:**
- 25 messages across 3 Slack channels (#platform, #auth-team, #engineering)
- 2 teams independently implementing OAuth2
- 14 days of conversation history
- ~40 hours of duplicated engineering effort (simulated)

**Output:**
```
✓ Inserted 25 messages into Postgres
✓ Created 25 embeddings in ChromaDB
✓ Indexed 25 messages in Elasticsearch
```

### 2.2 Verify Data Load

```bash
# Check message count
docker compose exec postgres psql -U coordination_user -d coordination \
  -c "SELECT channel, COUNT(*) FROM messages GROUP BY channel;"
```

**Expected output:**
```
   channel    | count
--------------+-------
 #platform    |    10
 #auth-team   |    10
 #engineering |     5
```

---

## Part 3: Search API Demo (5 minutes)

The system supports three search strategies. Let's try them all!

### 3.1 Semantic Search (Conceptual Understanding)

Search for messages about authentication using natural language:

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user authentication and authorization",
    "ranking_strategy": "semantic",
    "limit": 3
  }' | jq '.results[] | {channel, author, score, content}'
```

**What you'll see:**
- Top 3 messages semantically similar to "authentication"
- Messages about OAuth, even though the word "authentication" might not appear
- Similarity scores (0.0-1.0, higher = more similar)

**Key insight:** Semantic search understands concepts, not just keywords.

### 3.2 BM25 Keyword Search (Exact Terms)

Search for exact technical terms:

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth2 PKCE authorization",
    "ranking_strategy": "bm25",
    "limit": 3
  }' | jq '.results[] | {channel, author, content}'
```

**What you'll see:**
- Messages containing "OAuth2", "PKCE", "authorization"
- BM25 scores (higher = better keyword match)
- Exact technical term matching

**Key insight:** BM25 is best for technical terms and acronyms.

### 3.3 Hybrid Search (Best of Both Worlds)

Combine semantic understanding with keyword matching:

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "ranking_strategy": "hybrid_rrf",
    "limit": 5
  }' | jq '.results[] | {
    channel,
    author,
    score,
    ranking_details: {
      semantic_score,
      bm25_score
    },
    snippet: .content[:80]
  }'
```

**What you'll see:**
- Combined ranking using Reciprocal Rank Fusion (RRF)
- Both semantic_score and bm25_score for each result
- Usually the best overall results

**Key insight:** Hybrid search gives the best of both worlds - conceptual understanding + exact matching.

### 3.4 Compare Search Strategies

Let's see how they differ on the same query:

```bash
# Create a comparison script
cat > /tmp/compare_search.sh << 'EOF'
#!/bin/bash
QUERY="OAuth implementation decisions"

echo "=== SEMANTIC ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"semantic\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.channel) - \(.content[:60])..."'

echo -e "\n=== BM25 ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"bm25\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.channel) - \(.content[:60])..."'

echo -e "\n=== HYBRID ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"hybrid_rrf\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.channel) - \(.content[:60])..."'
EOF

chmod +x /tmp/compare_search.sh
/tmp/compare_search.sh
```

**Observe:** Different strategies rank results differently!

---

## Part 4: Gap Detection Demo (4 minutes)

Now for the main feature: detecting coordination gaps.

### 4.1 Run Gap Detection

```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.0,
    "include_evidence": true
  }' | jq
```

**What this does:**
1. Analyzes last 30 days of messages
2. Clusters semantically similar discussions
3. Identifies teams working on the same thing
4. Checks for temporal overlap (working simultaneously)
5. Verifies actual duplication (not intentional collaboration)
6. Calculates impact and cost estimates

### 4.2 Understanding the Results

The response will show detected gaps. Let's examine key fields:

```bash
# View gap summary
curl -s -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.0,
    "include_evidence": true
  }' | jq '.gaps[0] | {
    title,
    teams_involved,
    impact_tier,
    confidence,
    estimated_cost: .cost_estimate.engineering_hours,
    recommendation: .recommendation[:100]
  }'
```

**Key metrics explained:**

- **`impact_score`** (0.0-1.0): How significant is this gap?
  - 0.8-1.0: CRITICAL - Large teams, major duplication
  - 0.6-0.8: HIGH - Significant wasted effort
  - 0.4-0.6: MEDIUM - Moderate coordination issue
  - 0.0-0.4: LOW - Minor overlap

- **`confidence`** (0.0-1.0): How sure are we this is real duplication?
  - Combines semantic similarity + team separation + temporal overlap + LLM verification

- **`teams_involved`**: Which teams are duplicating work?

- **`cost_estimate`**: Estimated wasted engineering hours

- **`recommendation`**: What should you do about it?

### 4.3 View Evidence

Each gap includes evidence - the actual messages that indicate duplication:

```bash
curl -s -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.0,
    "include_evidence": true
  }' | jq '.gaps[0].evidence[] | {
    channel,
    author,
    timestamp,
    content: .content[:80]
  }' | head -20
```

**What you're seeing:**
- Specific messages from each team
- Who said what, when, and where
- Evidence that both teams are working on the same thing

### 4.4 Detection Metadata

```bash
curl -s -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.0
  }' | jq '.metadata'
```

**Key fields:**
- `total_gaps`: How many gaps were found
- `messages_analyzed`: How many messages were examined
- `clusters_found`: How many semantic clusters were formed
- `detection_time_ms`: How long detection took

---

## Part 5: Advanced Features (2 minutes)

### 5.1 Filter by Impact

Only show high-impact gaps:

```bash
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.6
  }' | jq '.metadata.total_gaps'
```

### 5.2 Adjust Detection Sensitivity

Make detection more or less strict:

```bash
# More strict (fewer false positives)
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.0,
    "similarity_threshold": 0.90
  }' | jq '.metadata.total_gaps'

# Less strict (catch more potential gaps)
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{
    "timeframe_days": 30,
    "gap_types": ["duplicate_work"],
    "min_impact_score": 0.0,
    "similarity_threshold": 0.65
  }' | jq '.metadata.total_gaps'
```

### 5.3 List All Detected Gaps

```bash
# Get paginated list of all gaps
curl "http://localhost:8000/api/v1/gaps?limit=10" | jq '.gaps[] | {
  id,
  type,
  teams_involved,
  impact_tier,
  detected_at
}'
```

### 5.4 Get Specific Gap Details

```bash
# First, get a gap ID
GAP_ID=$(curl -s "http://localhost:8000/api/v1/gaps?limit=1" | jq -r '.gaps[0].id')

# Then fetch full details
curl "http://localhost:8000/api/v1/gaps/$GAP_ID" | jq
```

---

## Part 6: Understanding How It Works (Optional)

### 6.1 The Detection Pipeline

Here's what happens when you run gap detection:

```
1. Data Retrieval
   ↓ Fetch messages from last N days

2. Semantic Clustering (DBSCAN)
   ↓ Group similar messages (similarity > 0.70)
   ↓ Uses 384-dimensional embeddings from sentence-transformers

3. Entity Extraction
   ↓ Identify teams, people, projects in each cluster

4. Multi-Team Filter
   ↓ Keep only clusters with 2+ teams

5. Temporal Overlap Check
   ↓ Verify teams working simultaneously (≥3 days overlap)

6. LLM Verification (Optional)
   ↓ Claude confirms actual duplication vs collaboration

7. Impact Scoring
   ↓ Calculate organizational cost
   ↓ = team_size × time_investment × project_criticality

8. Gap Creation
   ↓ Store verified gaps with evidence and recommendations
```

### 6.2 Check the Vector Store

See what embeddings are stored:

```bash
curl http://localhost:8000/api/v1/search/health | jq '.vector_store'
```

### 6.3 View Database Contents

```bash
# See all messages
docker compose exec postgres psql -U coordination_user -d coordination \
  -c "SELECT id, channel, author, LEFT(content, 50) as content_preview FROM messages LIMIT 5;"

# View message metadata
docker compose exec postgres psql -U coordination_user -d coordination \
  -c "SELECT channel, metadata FROM messages LIMIT 3;"
```

---

## Part 7: Load More Scenarios

Want to see different types of gaps? Load more mock scenarios:

### 7.1 API Redesign Duplication

```bash
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios api_redesign_duplication \
  --clear
```

**Scenario:** Mobile and Backend teams independently redesigning the API.

### 7.2 Multiple Scenarios

```bash
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios oauth_duplication,api_redesign_duplication,auth_migration_duplication
```

**Result:** Multiple gaps to detect across different topics.

### 7.3 Edge Cases (Should NOT Detect)

```bash
# Sequential work (no temporal overlap)
docker compose exec api python scripts/generate_mock_data.py \
  --scenarios sequential_work \
  --clear

# Run detection - should find 0 gaps
curl -s -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{"timeframe_days": 90, "min_impact_score": 0.0}' \
  | jq '.metadata.total_gaps'
```

**Expected:** 0 gaps (teams worked sequentially, not in parallel)

---

## Part 8: Cleanup

When you're done exploring:

```bash
# Stop all services
docker compose down

# Remove all data (optional - if you want to start fresh)
docker compose down -v
```

---

## Next Steps

Now that you've seen the demo, explore further:

📚 **Read the Docs:**
- [GAP_DETECTION.md](./GAP_DETECTION.md) - Deep dive into detection algorithms
- [RANKING.md](./RANKING.md) - Search quality and ranking strategies
- [ENTITY_EXTRACTION.md](./ENTITY_EXTRACTION.md) - How teams and projects are identified
- [API_EXAMPLES.md](./API_EXAMPLES.md) - Comprehensive API usage guide

🔬 **Try Advanced Features:**
- Adjust similarity thresholds to tune detection
- Filter by specific channels or teams
- Export gaps to JSON for reporting
- Integrate with webhooks for real-time alerts

💻 **Customize for Your Org:**
- Load real Slack data (see ingestion docs)
- Adjust impact scoring weights for your context
- Add custom entity extraction patterns
- Fine-tune detection parameters

🧪 **Run Tests:**
```bash
# See the full test suite
docker compose exec api pytest -v

# Run specific test categories
docker compose exec api pytest tests/test_detection/ -v
docker compose exec api pytest tests/test_integration/ -v
```

---

## Troubleshooting

**Services won't start?**
```bash
# Check logs
docker compose logs api
docker compose logs postgres

# Restart everything
docker compose down && docker compose up -d
```

**Search returns no results?**
```bash
# Verify embeddings were created
curl http://localhost:8000/api/v1/search/health | jq '.vector_store.document_count'

# Should be > 0. If not, regenerate:
docker compose exec api python scripts/generate_mock_data.py --scenarios oauth_duplication --clear
```

**Gap detection returns 0 gaps?**
```bash
# Check message count
docker compose exec postgres psql -U coordination_user -d coordination -c "SELECT COUNT(*) FROM messages;"

# If 0, load data:
docker compose exec api python scripts/generate_mock_data.py --scenarios oauth_duplication

# Try with lower thresholds:
curl -X POST http://localhost:8000/api/v1/gaps/detect \
  -H "Content-Type: application/json" \
  -d '{"timeframe_days": 90, "min_impact_score": 0.0, "similarity_threshold": 0.65}'
```

---

## Summary

In this demo, you learned how to:

✅ Set up the Coordination Gap Detector
✅ Load realistic mock data (Slack conversations)
✅ Search using semantic, BM25, and hybrid strategies
✅ Detect duplicate work coordination gaps
✅ Interpret impact scores, confidence, and recommendations
✅ Understand the detection pipeline

**Key Takeaways:**

1. **Semantic search** understands concepts, not just keywords
2. **Hybrid search** combines the best of semantic + keyword matching
3. **Gap detection** identifies teams unknowingly duplicating work
4. **Impact scoring** helps prioritize which gaps matter most
5. **LLM verification** reduces false positives

**The Big Picture:**

This system helps organizations answer:
- Are multiple teams working on the same thing?
- How much time/money are we wasting on duplicate work?
- Which coordination gaps should we address first?
- How can we prevent this from happening again?

---

**Questions or issues?** Check the [main README](../README.md) or open an issue on GitHub.

Happy exploring! 🚀
