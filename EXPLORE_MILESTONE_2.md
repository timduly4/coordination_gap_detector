# Exploring Milestone 2: Ranking & Search Quality

This guide helps you understand what was built in Milestone 2 through hands-on exploration.

## Quick Reference: What Milestone 2 Built

✅ **Elasticsearch Integration** - Keyword search with BM25
✅ **BM25 Scoring** - Probabilistic ranking algorithm
✅ **Hybrid Search** - Semantic + keyword fusion (RRF and weighted)
✅ **Ranking Metrics** - MRR, NDCG, DCG, Precision/Recall
✅ **Feature Engineering** - 40+ ranking signals
✅ **Evaluation Framework** - Offline testing and comparison

---

## 🎯 Learning Path (Recommended Order)

### Level 1: Run the System (15-20 minutes)
Get the system running and see it in action.

### Level 2: Understand the Code (30-45 minutes)
Read key files and understand the implementation.

### Level 3: Run Tests & Experiments (30-45 minutes)
Execute tests and experiment with parameters.

### Level 4: Deep Dive (1-2 hours)
Explore advanced features and create custom scenarios.

---

## Level 1: Run the System 🚀

### Step 1: Start All Services

```bash
# Start Docker services (Postgres, Redis, Elasticsearch, ChromaDB)
docker compose up -d

# Wait for services to be healthy (check logs)
docker compose ps

# Verify Elasticsearch is running
curl http://localhost:9200
# Should return JSON with cluster info
```

**What to look for**:
- ✅ All services show "healthy" or "running"
- ✅ Elasticsearch returns cluster name and version
- ✅ No error messages in logs

### Step 2: Load Mock Data

```bash
# Generate and load mock Slack messages (all scenarios)
uv run python scripts/generate_mock_data.py --scenarios all --clear

# Verify data loaded
docker compose exec postgres psql -U coordination_user -d coordination -c "SELECT COUNT(*) FROM messages;"
# Should show 24 messages loaded

# Verify Elasticsearch index
curl -s http://localhost:9200/messages/_count | jq .
# Should show {"count": 24}
```

**What to look for**:
- ✅ Mock data generation completes without errors
- ✅ Messages inserted into database (24 messages)
- ✅ Embeddings created in ChromaDB (24 embeddings)
- ✅ Messages indexed in Elasticsearch (24 documents)

### Step 3: Start the API

```bash
# Start FastAPI server
uv run uvicorn src.main:app --reload --port 8000

# In another terminal, test health endpoint
curl http://localhost:8000/health
```

**Expected response**:
```json
{
  "status": "healthy",
  "database": "connected",
  "elasticsearch": "connected",
  "vector_store": "connected"
}
```

### Step 4: Test Basic Search

```bash
# Simple semantic search
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation",
    "limit": 5
  }'
```

**What to look for**:
- ✅ Returns 5 results
- ✅ Each result has content, score, source
- ✅ Scores are between 0 and 1

### Step 5: Test Hybrid Search

```bash
# Hybrid search with RRF fusion
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation decisions",
    "ranking_strategy": "hybrid_rrf",
    "limit": 10
  }' | jq .
```

**What to look for**:
- ✅ Results include `ranking_details` with both semantic and BM25 scores
- ✅ Results show `fusion_method: "rrf"`
- ✅ Both `semantic_rank` and `keyword_rank` present

### Step 6: Compare Ranking Strategies

```bash
# Create a comparison script
cat > compare_strategies.sh << 'EOF'
#!/bin/bash
echo "=== Semantic Only ==="
curl -s -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth", "ranking_strategy": "semantic", "limit": 3}' \
  | jq '.results[] | {content: .content[:50], score: .score}'

echo ""
echo "=== BM25 Only ==="
curl -s -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth", "ranking_strategy": "bm25", "limit": 3}' \
  | jq '.results[] | {content: .content[:50], score: .score}'

echo ""
echo "=== Hybrid RRF ==="
curl -s -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth", "ranking_strategy": "hybrid_rrf", "limit": 3}' \
  | jq '.results[] | {content: .content[:50], score: .score}'
EOF

chmod +x compare_strategies.sh
./compare_strategies.sh
```

**What to observe**:
- Different strategies rank results differently
- Hybrid often has best coverage
- BM25 better for exact keyword matches
- Semantic better for conceptual matches

---

## Level 2: Understand the Code 📚

### Key Files to Read (in order)

#### 1. BM25 Implementation

```bash
# Read the BM25 scorer
cat src/ranking/scoring.py | less
```

**Focus on**:
- `BM25Scorer` class initialization (k1, b parameters)
- `calculate_idf()` method - how IDF is computed
- `score()` method - the actual BM25 formula
- `score_with_explanation()` - see how scoring is broken down

**Questions to answer**:
- What is k1 and what does it control? (term frequency saturation)
- What is b and what does it control? (length normalization)
- How is IDF calculated? (log formula with smoothing)

#### 2. Hybrid Search Fusion

```bash
# Read hybrid search implementation
cat src/search/hybrid_search.py | less
```

**Focus on**:
- `HybridSearchFusion` class
- `_reciprocal_rank_fusion()` method - RRF algorithm
- `_weighted_score_fusion()` method - weighted combination
- How results are merged and deduplicated

**Questions to answer**:
- What's the RRF formula? (1/(k + rank))
- Why use RRF vs weighted scores? (no normalization needed)
- What's the default RRF k value? (60)

#### 3. Ranking Metrics

```bash
# Read metrics implementation
cat src/ranking/metrics.py | less
```

**Focus on**:
- `calculate_mrr()` - Mean Reciprocal Rank
- `calculate_ndcg()` - Normalized DCG
- `calculate_dcg()` - Discounted Cumulative Gain
- How graded relevance works (0-3 scale)

**Questions to answer**:
- What does MRR measure? (rank of first relevant result)
- How does DCG discount position? (log2(i + 1))
- What's the range of NDCG? (0 to 1)

#### 4. Feature Engineering

```bash
# Read feature extraction
cat src/ranking/features.py | less
```

**Focus on**:
- What features are extracted
- How features are normalized
- Feature categories (similarity, temporal, engagement, authority)

#### 5. Search Service Integration

```bash
# Read how it all comes together
cat src/services/search_service.py | less
```

**Focus on**:
- How different strategies are dispatched
- How results are formatted
- Error handling

---

## Level 3: Run Tests & Experiments 🧪

### Step 1: Run All Ranking Tests

```bash
# Run all ranking-related tests
uv run pytest tests/test_ranking/ -v

# Run specific test files
uv run pytest tests/test_ranking/test_metrics.py -v
uv run pytest tests/test_ranking/test_bm25.py -v
uv run pytest tests/test_ranking/test_features.py -v
```

**What to observe**:
- All tests should pass ✅
- Read test names to understand what's being tested
- Look at test coverage numbers

### Step 2: Run Tests with Coverage

```bash
# See what's actually tested
uv run pytest tests/test_ranking/ --cov=src/ranking --cov-report=term-missing

# Generate HTML coverage report
uv run pytest tests/test_ranking/ --cov=src/ranking --cov-report=html
open htmlcov/index.html
```

**What to look for**:
- Coverage > 90% for ranking modules
- What lines are NOT covered (edge cases?)
- Which functions are most thoroughly tested

### Step 3: Understand Tests by Reading Them

```bash
# Read MRR tests
cat tests/test_ranking/test_metrics.py | grep -A 20 "test_mrr"
```

**Example test to understand**:
```python
def test_mrr_first_result_relevant():
    # First result is relevant (rank 1)
    queries = [[1, 0, 0]]
    mrr = calculate_mrr(queries)
    assert mrr == 1.0  # Perfect MRR
```

**Try creating your own test scenarios**:
```python
# What's the MRR here?
queries = [
    [0, 1, 0, 0],  # First relevant at rank 2 → 1/2 = 0.5
    [1, 0, 0, 0],  # First relevant at rank 1 → 1/1 = 1.0
    [0, 0, 1, 0],  # First relevant at rank 3 → 1/3 = 0.333
]
# MRR = (0.5 + 1.0 + 0.333) / 3 = 0.611
```

### Step 4: Experiment with BM25 Parameters

Create an experiment script:

```python
# experiment_bm25.py
from src.ranking.scoring import BM25Scorer

# Test different k1 values (term frequency saturation)
for k1 in [0.5, 1.2, 1.5, 2.0]:
    scorer = BM25Scorer(k1=k1, b=0.75)

    # Mock collection stats
    term_idfs = {"oauth": 2.5, "implementation": 1.8}

    score = scorer.score(
        query_terms=["oauth", "implementation"],
        document="We are implementing OAuth for authentication",
        document_length=6,
        avg_doc_length=10,
        term_idfs=term_idfs
    )

    print(f"k1={k1}: score={score:.3f}")

print("\n" + "="*50 + "\n")

# Test different b values (length normalization)
for b in [0.0, 0.5, 0.75, 1.0]:
    scorer = BM25Scorer(k1=1.5, b=b)

    score = scorer.score(
        query_terms=["oauth", "implementation"],
        document="We are implementing OAuth for authentication",
        document_length=6,
        avg_doc_length=10,
        term_idfs=term_idfs
    )

    print(f"b={b}: score={score:.3f}")
```

Run it:
```bash
uv run python experiment_bm25.py
```

**What to observe**:
- How k1 affects score (higher k1 = less saturation)
- How b affects length normalization (b=0: no penalty, b=1: full penalty)

### Step 5: Run the Evaluation Script

```bash
# Check if evaluation script exists
ls -la scripts/evaluate_ranking.py

# See what it does
cat scripts/evaluate_ranking.py | head -100

# If test queries exist, run evaluation
# (may need to create test queries first)
```

---

## Level 4: Deep Dive 🔬

### Activity 1: Interactive Ranking Exploration

Create a Jupyter notebook:

```bash
# Install jupyter if needed
uv pip install jupyter

# Create notebook
jupyter notebook
```

**Notebook: `explore_ranking.ipynb`**

```python
# Cell 1: Setup
import sys
sys.path.insert(0, '/Users/duly/coordination_gap_detector')

from src.ranking.scoring import BM25Scorer
from src.ranking.metrics import calculate_mrr, calculate_ndcg
from src.search.hybrid_search import HybridSearchFusion

# Cell 2: BM25 Exploration
scorer = BM25Scorer(k1=1.5, b=0.75)

documents = [
    "OAuth implementation for API gateway",
    "Implementing authentication with OAuth2",
    "API gateway configuration guide",
    "OAuth tutorial and examples"
]

query = ["oauth", "implementation"]

# Score all documents
for doc in documents:
    result = scorer.score_with_explanation(
        query_terms=query,
        document=doc,
        document_length=len(doc.split()),
        avg_doc_length=5,
        term_idfs={"oauth": 2.0, "implementation": 1.5}
    )
    print(f"Doc: {doc}")
    print(f"Score: {result['total_score']:.3f}")
    print(f"Details: {result['term_scores']}")
    print()

# Cell 3: Ranking Metrics
# Simulate search results with relevance judgments
relevance_scores = [3, 2, 1, 0, 0]  # Graded 0-3
ndcg = calculate_ndcg(relevance_scores, k=5)
print(f"NDCG@5: {ndcg:.3f}")

# Compare different rankings
perfect_ranking = [3, 2, 1, 0, 0]
poor_ranking = [0, 0, 1, 2, 3]

print(f"Perfect: NDCG={calculate_ndcg(perfect_ranking, k=5):.3f}")
print(f"Poor: NDCG={calculate_ndcg(poor_ranking, k=5):.3f}")

# Cell 4: Hybrid Fusion
semantic_results = [
    {"id": "doc1", "content": "OAuth impl", "semantic_score": 0.92},
    {"id": "doc2", "content": "Auth guide", "semantic_score": 0.85},
    {"id": "doc3", "content": "API setup", "semantic_score": 0.78}
]

keyword_results = [
    {"id": "doc3", "content": "API setup", "keyword_score": 0.95},
    {"id": "doc1", "content": "OAuth impl", "keyword_score": 0.88},
    {"id": "doc4", "content": "OAuth spec", "keyword_score": 0.82}
]

# Try RRF fusion
fusion = HybridSearchFusion(strategy="rrf")
fused = fusion.fuse(semantic_results, keyword_results)

for result in fused:
    print(f"ID: {result['id']}, Score: {result['score']:.3f}")
    print(f"  Ranks: semantic={result['ranking_details']['semantic_rank']}, "
          f"keyword={result['ranking_details']['keyword_rank']}")
```

### Activity 2: Create Custom Test Scenarios

```bash
# Create a test scenario file
cat > test_ranking_scenario.py << 'EOF'
"""
Custom ranking scenario to test understanding.

Scenario: Search for "OAuth security best practices"
We have 5 documents with known relevance.
"""

from src.ranking.metrics import calculate_mrr, calculate_ndcg, calculate_precision_at_k

# Relevance judgments (0-3 scale)
# 3 = highly relevant, 2 = relevant, 1 = somewhat relevant, 0 = not relevant
ground_truth = {
    "doc1": 3,  # "OAuth 2.0 Security Best Practices"
    "doc2": 2,  # "OAuth implementation guide"
    "doc3": 1,  # "API security overview"
    "doc4": 0,  # "Database optimization"
    "doc5": 2,  # "OAuth security vulnerabilities"
}

# Test different ranking strategies
def test_ranking(ranked_doc_ids, name):
    """Test a ranking and compute metrics."""
    relevance_scores = [ground_truth[doc_id] for doc_id in ranked_doc_ids]

    # Binary relevance for MRR (3,2,1 = relevant, 0 = not relevant)
    binary = [1 if r > 0 else 0 for r in relevance_scores]

    mrr = calculate_mrr([binary])
    ndcg = calculate_ndcg(relevance_scores, k=5)
    precision = calculate_precision_at_k(binary, k=3)

    print(f"\n{name}:")
    print(f"  Ranking: {ranked_doc_ids}")
    print(f"  Relevance: {relevance_scores}")
    print(f"  MRR: {mrr:.3f}")
    print(f"  NDCG@5: {ndcg:.3f}")
    print(f"  P@3: {precision:.3f}")

    return mrr, ndcg, precision

# Test scenarios
print("="*60)
print("Testing Different Rankings")
print("="*60)

# Perfect ranking (best to worst)
test_ranking(["doc1", "doc2", "doc5", "doc3", "doc4"], "Perfect Ranking")

# Good ranking (mostly correct)
test_ranking(["doc1", "doc5", "doc2", "doc3", "doc4"], "Good Ranking")

# Poor ranking (reversed)
test_ranking(["doc4", "doc3", "doc2", "doc5", "doc1"], "Poor Ranking")

# Random ranking
test_ranking(["doc3", "doc1", "doc4", "doc5", "doc2"], "Random Ranking")

print("\n" + "="*60)
EOF

uv run python test_ranking_scenario.py
```

**What to observe**:
- How NDCG changes with ranking quality
- MRR only cares about first relevant result
- Precision@3 only looks at top 3

### Activity 3: Trace a Request Through the System

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Start server with logging
uv run uvicorn src.main:app --reload --log-level debug

# In another terminal, make a request
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth", "ranking_strategy": "hybrid_rrf", "limit": 3}'

# Watch the server logs to see:
# 1. Request received
# 2. Query parsing
# 3. Semantic search execution
# 4. BM25 search execution
# 5. Fusion algorithm
# 6. Response formatting
```

### Activity 4: Read Git History

```bash
# See how Milestone 2 was built incrementally
git log --oneline --grep="Milestone 2" --all

# See specific PR
git log --oneline ee2f1e8..d19aab8

# Look at what changed in each PR
git show d638157  # BM25 implementation
git show ee2f1e8  # Hybrid search

# See file evolution
git log --follow -p src/ranking/scoring.py
```

---

## 🎓 Knowledge Check: Can You Answer These?

### BM25 & Keyword Search
- [ ] What does k1 control in BM25? (term frequency saturation)
- [ ] What does b control in BM25? (length normalization)
- [ ] How is IDF calculated? (log formula)
- [ ] Why do common words get lower IDF? (appear in many docs)
- [ ] When would you increase k1? (long queries, technical terms)
- [ ] When would you decrease b? (documents vary in length naturally)

### Hybrid Search
- [ ] What's the RRF formula? (1/(k + rank))
- [ ] Why is RRF better than score averaging? (no normalization needed)
- [ ] What's the default RRF k constant? (60)
- [ ] When would you use weighted fusion instead? (when scores are already normalized)
- [ ] How does deduplication work? (by message_id/external_id)

### Ranking Metrics
- [ ] What does MRR measure? (rank of first relevant result)
- [ ] What's the range of NDCG? (0 to 1)
- [ ] How does DCG discount position? (log2(i + 1))
- [ ] What's the difference between binary and graded relevance? (0-1 vs 0-3)
- [ ] When would NDCG@10 be better than MRR? (when multiple results matter)

### System Architecture
- [ ] How does hybrid search work? (parallel semantic + BM25, then fuse)
- [ ] Where are embeddings stored? (ChromaDB)
- [ ] Where are keyword indices? (Elasticsearch)
- [ ] What's the search flow? (parse → retrieve → score → rank → return)

---

## 🚀 Milestone 2 → Milestone 3 Readiness

### You're Ready for Milestone 3 When:

✅ **Understanding**:
- [ ] Can explain BM25 algorithm
- [ ] Understand hybrid search fusion
- [ ] Know what MRR and NDCG measure
- [ ] Understand the search pipeline

✅ **Practical Skills**:
- [ ] Can run the system locally
- [ ] Can query different ranking strategies
- [ ] Can read and understand the implementation
- [ ] Can run and interpret tests

✅ **Conceptual Links**:
- [ ] Understand how Milestone 3 will use search (finding similar discussions)
- [ ] See how clustering relates to search (grouping search results)
- [ ] Understand how features will be used (in gap detection scoring)

---

## 📊 Quick Reference Commands

```bash
# Start everything
docker compose up -d
uv run uvicorn src.main:app --reload

# Test search strategies
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth", "ranking_strategy": "hybrid_rrf"}'

# Run tests
uv run pytest tests/test_ranking/ -v

# Check coverage
uv run pytest tests/test_ranking/ --cov=src/ranking --cov-report=term

# Read key files
cat src/ranking/scoring.py
cat src/search/hybrid_search.py
cat src/ranking/metrics.py

# Explore git history
git log --oneline --all | grep -i milestone
```

---

## 🔗 How Milestone 2 Connects to Milestone 3

**Milestone 2 Built:**
- Search infrastructure (semantic + keyword)
- Ranking algorithms (BM25, hybrid)
- Evaluation metrics (MRR, NDCG)
- Feature extraction patterns

**Milestone 3 Will Use:**
- ✅ Semantic search → Find similar discussions (clustering)
- ✅ Feature extraction → Extract entities, compute gap scores
- ✅ Ranking patterns → Rank gap evidence by relevance
- ✅ Metrics → Evaluate gap detection quality

**Key Insight**: Milestone 3 is applying search/ranking techniques to a different problem (coordination gaps vs. document search).

---

## Next Steps

1. ✅ Complete at least Level 1 & 2 (run system + understand code)
2. ✅ Try Level 3 experiments (tests + parameters)
3. ✅ Answer knowledge check questions
4. ✅ Read MILESTONE_3_BREAKDOWN.md
5. ✅ Start Milestone 3 PR 3A (Entity Extraction)

Good luck! 🚀
