# Milestone 2: Ranking & Search Quality - PR Breakdown

## Overview
Milestone 2 focuses on implementing information retrieval fundamentals and advanced ranking capabilities. Breaking this into focused PRs demonstrates expertise in search quality, ranking algorithms, and evaluation methodologies. Each PR should be independently reviewable and build upon the foundation from Milestone 1.

---

## Milestone 2A: Elasticsearch Integration & Setup
**Branch**: `feat/elasticsearch-integration`
**Time**: ~2-3 hours
**Files Changed**: ~6-8 files

### Changes:
```
coordination-gap-detector/
├── docker-compose.yml          # Updated with Elasticsearch
├── pyproject.toml              # Add elasticsearch dependency
└── src/
    ├── db/
    │   ├── elasticsearch.py    # ES client and operations
    │   └── __init__.py         # Updated exports
    ├── config.py               # ES configuration
    └── api/routes/
        └── health.py           # Updated health check
```

### Specific Tasks:
- [ ] Add Elasticsearch to docker-compose.yml (single-node cluster)
- [ ] Add elasticsearch Python package to pyproject.toml
- [ ] Create elasticsearch.py with client initialization
- [ ] Implement index creation and mapping definitions
- [ ] Add ES connection to health check endpoint
- [ ] Create indexing methods for messages
- [ ] Add error handling for ES connection failures
- [ ] Update .env.example with ES configuration

### PR Description Template:
```
## Elasticsearch Integration

Add Elasticsearch for keyword search and BM25 ranking.

### Changes
- Elasticsearch 8.x service in Docker Compose
- ES client with connection management
- Index creation with proper mappings
- Health check integration
- Environment configuration

### Testing
- [ ] `docker compose up` starts Elasticsearch successfully
- [ ] ES available at http://localhost:9200
- [ ] Health endpoint shows ES connectivity
- [ ] Can create indices and mappings
- [ ] Connection error handling works


### Index Mapping

{
  "messages": {
    "mappings": {
      "properties": {
        "content": {"type": "text", "analyzer": "standard"},
        "source": {"type": "keyword"},
        "channel": {"type": "keyword"},
        "author": {"type": "keyword"},
        "timestamp": {"type": "date"},
        "metadata": {"type": "object"}
      }
    }
  }
}
```

### Next Steps
- Implement BM25 scoring (PR 2B)

**Commit Messages:**

```
feat: add elasticsearch service to docker compose
feat: implement elasticsearch client and connection management
feat: create message index with text search mappings
feat: integrate elasticsearch health check
```

---

## Milestone 2B: BM25 Scoring Implementation
**Branch**: `feat/bm25-scoring`
**Time**: ~3-4 hours
**Files Changed**: ~6-8 files

### Changes:
```
src/
├── ranking/
│   ├── __init__.py
│   ├── scoring.py              # BM25 implementation
│   └── constants.py            # BM25 parameters (k1, b)
├── search/
│   ├── __init__.py
│   └── retrieval.py            # Search retrieval logic
└── db/
    └── elasticsearch.py        # Updated with BM25 queries
tests/
├── test_ranking/
│   ├── __init__.py
│   └── test_bm25.py
```

### Specific Tasks:
- [ ] Implement BM25 scoring algorithm
- [ ] Add configurable parameters (k1=1.5, b=0.75)
- [ ] Create term frequency and IDF calculations
- [ ] Implement ES query builder for BM25
- [ ] Add document length normalization
- [ ] Create retrieval service for keyword search
- [ ] Add score explanation feature
- [ ] Write unit tests for BM25 calculations
- [ ] Add integration tests with ES

### PR Description Template:
```
## BM25 Scoring Implementation

Probabilistic ranking function for keyword-based search.

### Changes
- BM25 scoring algorithm implementation
- Configurable parameters (k1, b)
- Term frequency and IDF calculations
- Document length normalization
- Elasticsearch query integration
- Score explanations

### BM25 Formula
score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1 + 1)) / (f(qi,D) + k1 · (1 - b + b · |D|/avgdl))

Where:
- `f(qi,D)` = term frequency of qi in document D
- `|D|` = document length
- `avgdl` = average document length
- `k1` = term frequency saturation (default: 1.5)
- `b` = length normalization (default: 0.75)

### Testing
- [ ] BM25 scores calculated correctly
- [ ] IDF weights common terms lower
- [ ] Length normalization works
- [ ] Configurable parameters applied
- [ ] ES queries return expected results
- [ ] Score explanations are accurate

### Example Usage
```python
from src.ranking.scoring import BM25Scorer

scorer = BM25Scorer(k1=1.5, b=0.75)
score = scorer.score(query="OAuth implementation", document=doc)
# Returns BM25 relevance score
```

**Commit Messages:**

```
feat: implement BM25 scoring algorithm
feat: add IDF and term frequency calculations
feat: integrate BM25 with elasticsearch queries
test: add comprehensive BM25 unit tests
```

---

## Milestone 2C: Hybrid Search (Semantic + BM25)
**Branch**: `feat/hybrid-search`
**Time**: ~3-4 hours
**Files Changed**: ~7-9 files

### Changes:
```
src/
├── search/
│   ├── hybrid_search.py        # Fusion of semantic + keyword
│   ├── query_parser.py         # Query understanding
│   └── filters.py              # Search filters
├── ranking/
│   └── scoring.py              # Updated with fusion strategies
└── services/
    └── search_service.py       # Updated to support hybrid
tests/
├── test_search/
│   ├── __init__.py
│   ├── test_hybrid_search.py
│   └── test_query_parser.py
```

### Specific Tasks:
- [ ] Implement reciprocal rank fusion (RRF)
- [ ] Add weighted score fusion strategy
- [ ] Create query parser for search intent
- [ ] Implement result deduplication
- [ ] Add configurable fusion weights
- [ ] Update search service to use hybrid search
- [ ] Add ranking strategy parameter to API
- [ ] Write tests for fusion algorithms
- [ ] Add performance benchmarks

### PR Description Template:
```
## Hybrid Search Implementation

Combines semantic similarity and keyword matching for better search quality.

### Changes
- Reciprocal Rank Fusion (RRF) implementation
- Weighted score fusion strategy
- Query parsing and intent detection
- Result deduplication and merging
- Configurable fusion parameters
- Updated search API with ranking strategies

### Fusion Strategies

**1. Reciprocal Rank Fusion (RRF)**

score(d) = Σ 1/(k + rank_i(d))

- Combines rankings from multiple sources
- `k=60` constant (standard value)
- Rank-based, not score-based

**2. Weighted Score Fusion**

score(d) = α·semantic_score(d) + β·bm25_score(d)

- Default: α=0.7, β=0.3
- Normalizes scores before combining

### Testing
- [ ] RRF produces sensible combined rankings
- [ ] Weighted fusion respects alpha/beta parameters
- [ ] Deduplication removes exact matches
- [ ] Query parser identifies intent correctly
- [ ] Hybrid search outperforms single-method
- [ ] API accepts ranking_strategy parameter

### API Example
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation decisions",
    "ranking_strategy": "hybrid_rrf",
    "limit": 10
  }'

### Response
{
  "results": [{
    "content": "We decided to use Auth0...",
    "score": 0.89,
    "ranking_details": {
      "semantic_score": 0.92,
      "bm25_score": 0.85,
      "semantic_rank": 1,
      "bm25_rank": 3,
      "fusion_method": "rrf"
    }
  }]
}
```

**Commit Messages:**
```
feat: implement reciprocal rank fusion
feat: add weighted score fusion strategy
feat: create query parser for search intent
feat: update search API with hybrid ranking
test: add hybrid search integration tests
```

---

## Milestone 2D: Ranking Metrics Implementation
**Branch**: `feat/ranking-metrics`
**Time**: ~3-4 hours
**Files Changed**: ~5-7 files

### Changes:
```
src/
├── ranking/
│   ├── metrics.py              # MRR, NDCG, DCG calculations
│   └── evaluation.py           # Evaluation utilities
└── models/
    └── schemas.py              # Updated with relevance judgments
tests/
├── test_ranking/
│   ├── test_metrics.py         # Comprehensive metric tests
│   └── test_evaluation.py
└── fixtures/
    └── labeled_queries.json    # Test data with relevance labels
```

### Specific Tasks:
- [ ] Implement Mean Reciprocal Rank (MRR)
- [ ] Implement NDCG (Normalized Discounted Cumulative Gain)
- [ ] Implement DCG and iDCG calculations
- [ ] Add Precision@k and Recall@k
- [ ] Create evaluation framework
- [ ] Add relevance judgment data structures
- [ ] Implement graded relevance (0-3 scale)
- [ ] Write comprehensive metric tests
- [ ] Add metric calculation examples

### PR Description Template:
```
## Ranking Metrics Implementation

Information retrieval metrics for evaluating search quality.

### Changes
- Mean Reciprocal Rank (MRR) implementation
- NDCG@k (Normalized Discounted Cumulative Gain)
- DCG and iDCG calculations
- Precision@k and Recall@k
- Evaluation framework
- Graded relevance support (0-3 scale)

### Metrics Explained

**Mean Reciprocal Rank (MRR)**

MRR = (1/|Q|) · Σ 1/rank_i

- Measures rank of first relevant result
- Range: [0, 1], higher is better
- Use case: Finding the best match

**NDCG@k (Normalized DCG)**

DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
NDCG@k = DCG@k / iDCG@k

- Graded relevance (not binary)
- Position-aware with logarithmic discount
- Range: [0, 1], higher is better
- Use case: Ranking quality with graded judgments

**Precision@k**

P@k = (relevant items in top k) / k


**Recall@k**

R@k = (relevant items in top k) / (total relevant items)


### Relevance Scale
- **3**: Highly relevant (perfect match)
- **2**: Relevant (good match)
- **1**: Partially relevant (somewhat related)
- **0**: Not relevant

### Testing
- [ ] MRR calculated correctly for test queries
- [ ] NDCG@k matches known values
- [ ] DCG calculation is accurate
- [ ] Precision/Recall computed correctly
- [ ] Handles edge cases (no results, all relevant)
- [ ] Graded relevance works properly

### Example Usage

from src.ranking.metrics import calculate_mrr, calculate_ndcg

# MRR example
queries = [
    {"results": [0, 1, 0, 0]},  # First relevant at rank 2
    {"results": [1, 0, 0, 0]}   # First relevant at rank 1
]
mrr = calculate_mrr(queries)  # = 0.75

# NDCG example
relevance_scores = [3, 2, 1, 0]  # Graded relevance
ndcg = calculate_ndcg(relevance_scores, k=4)
```

**Commit Messages:**
```
feat: implement MRR calculation
feat: implement NDCG and DCG metrics
feat: add precision and recall at k
feat: create evaluation framework with graded relevance
test: add comprehensive ranking metrics tests
```

---

## Milestone 2E: Feature Engineering for Ranking
**Branch**: `feat/ranking-features`
**Time**: ~3-4 hours
**Files Changed**: ~6-8 files

### Changes:
```
src/
├── ranking/
│   ├── features.py             # Feature extraction
│   └── feature_config.py       # Feature definitions
├── search/
│   └── retrieval.py            # Updated with features
└── utils/
    └── time_utils.py           # Temporal feature utilities
tests/
└── test_ranking/
    └── test_features.py
```

### Specific Tasks:
- [ ] Implement query-document similarity features
- [ ] Add temporal features (recency, activity burst)
- [ ] Create engagement features (thread depth, participants)
- [ ] Add source authority features
- [ ] Implement feature normalization
- [ ] Create feature configuration system
- [ ] Add feature importance logging
- [ ] Write feature extraction tests
- [ ] Document all features

### PR Description Template:
```
## Ranking Feature Engineering

40+ ranking signals for improved search quality.

### Changes
- Query-document similarity features
- Temporal recency and activity features
- Engagement metrics (threads, participants)
- Source authority signals
- Feature normalization
- Feature configuration system
- Comprehensive feature tests

### Feature Categories

**1. Query-Document Similarity (6 features)**
- `semantic_score`: Cosine similarity of embeddings
- `bm25_score`: BM25 keyword relevance
- `exact_match`: Exact phrase match bonus
- `term_coverage`: % of query terms in document
- `title_match`: Query match in title/channel
- `entity_overlap`: Shared entities (people, teams)

**2. Temporal Features (5 features)**
- `recency`: Time since message (exponential decay)
- `activity_burst`: Recent activity spike detection
- `temporal_relevance`: Time alignment with query context
- `edit_freshness`: Time since last edit
- `response_velocity`: Reply rate in thread

**3. Engagement Features (6 features)**
- `thread_depth`: Number of replies
- `participant_count`: Unique participants
- `reaction_count`: Total reactions
- `reaction_diversity`: Unique reaction types
- `cross_team_engagement`: Multiple teams involved
- `view_count`: Message views (if available)

**4. Source Authority (4 features)**
- `author_seniority`: Author's position/tenure
- `channel_importance`: Channel activity/membership
- `team_influence`: Team's organizational importance
- `domain_expertise`: Author expertise in topic area

**5. Content Features (3 features)**
- `message_length`: Character count (normalized)
- `code_snippet_present`: Contains code blocks
- `link_count`: External references

### Feature Normalization
- Min-max scaling to [0, 1] range
- Per-feature statistics tracked
- Outlier handling with percentile clipping

### Testing
- [ ] All features compute without errors
- [ ] Feature values in expected ranges
- [ ] Normalization works correctly
- [ ] Handles missing data gracefully
- [ ] Feature importance logged
- [ ] Performance acceptable (<50ms per doc)

### Example Usage
```python
from src.ranking.features import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract(
    query="OAuth implementation",
    document=message,
    context=search_context
)

# Returns:
# {
#   "semantic_score": 0.92,
#   "bm25_score": 0.85,
#   "recency": 0.95,
#   "thread_depth": 0.60,
#   "author_seniority": 0.78,
#   ...
# }
```

**Commit Messages:**
```
feat: implement query-document similarity features
feat: add temporal and engagement features
feat: create source authority features
feat: add feature normalization and configuration
test: add feature extraction tests
docs: document all ranking features
```

---

## Milestone 2F: Evaluation Framework
**Branch**: `feat/evaluation-framework`
**Time**: ~3-4 hours
**Files Changed**: ~7-9 files

### Changes:
```
scripts/
├── evaluate_ranking.py         # Main evaluation script
└── generate_test_queries.py    # Create test set
data/
└── test_queries/
    ├── queries.jsonl           # Test queries
    └── relevance_judgments.jsonl
src/
├── ranking/
│   └── evaluation.py           # Evaluation utilities
└── services/
    └── evaluation_service.py   # Evaluation API
tests/
└── test_ranking/
    └── test_evaluation_framework.py
```

### Specific Tasks:
- [ ] Create evaluation script for offline testing
- [ ] Implement test query generation
- [ ] Add relevance judgment collection
- [ ] Create comparison framework (A/B strategy comparison)
- [ ] Add result export (CSV, JSON)
- [ ] Implement statistical significance testing
- [ ] Add visualization helpers
- [ ] Create evaluation API endpoint
- [ ] Write evaluation documentation

### PR Description Template:
```
## Evaluation Framework

Offline evaluation system for ranking quality assessment.

### Changes
- Evaluation script for offline testing
- Test query generation from mock data
- Relevance judgment framework
- Strategy comparison (semantic vs BM25 vs hybrid)
- Statistical significance testing
- Result export and visualization
- Evaluation API endpoint

### Evaluation Workflow

1. **Generate Test Queries**

uv run python scripts/generate_test_queries.py \
  --output data/test_queries/queries.jsonl \
  --count 50


2. **Label Relevance (Manual)**

{
  "query": "OAuth implementation",
  "results": [
    {"doc_id": "msg_123", "relevance": 3},
    {"doc_id": "msg_456", "relevance": 2},
    {"doc_id": "msg_789", "relevance": 0}
  ]
}


3. **Run Evaluation**

uv run python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/relevance_judgments.jsonl \
  --strategies semantic,bm25,hybrid_rrf,hybrid_weighted


4. **View Results**

Strategy Comparison:
┌─────────────────┬───────┬──────────┬──────────┬──────────┐
│ Strategy        │  MRR  │ NDCG@10  │  P@5     │  R@10    │
├─────────────────┼───────┼──────────┼──────────┼──────────┤
│ semantic        │ 0.68  │  0.72    │  0.64    │  0.58    │
│ bm25            │ 0.62  │  0.65    │  0.58    │  0.52    │
│ hybrid_rrf      │ 0.74  │  0.79    │  0.72    │  0.66    │
│ hybrid_weighted │ 0.72  │  0.77    │  0.70    │  0.64    │
└─────────────────┴───────┴──────────┴──────────┴──────────┘

Best: hybrid_rrf (+8.8% MRR vs semantic)


### Test Query Categories
- Factual queries: "Who approved OAuth decision?"
- Technical queries: "OAuth implementation code"
- Temporal queries: "Recent discussions about auth"
- Multi-source queries: "Auth mentioned in Slack and GitHub"
- Ambiguous queries: "Login issues"

### Testing
- [ ] Evaluation script runs without errors
- [ ] Metrics calculated for all strategies
- [ ] Statistical tests work correctly
- [ ] Export formats valid (CSV, JSON)
- [ ] API endpoint returns evaluation results
- [ ] Handles missing judgments gracefully

### API Endpoint

POST /api/v1/evaluate
{
  "queries": [...],
  "strategies": ["semantic", "hybrid_rrf"],
  "metrics": ["mrr", "ndcg@10"]
}
```

**Commit Messages:**
```
feat: create offline evaluation script
feat: implement test query generation
feat: add strategy comparison framework
feat: add statistical significance testing
feat: create evaluation API endpoint
docs: add evaluation framework documentation
```

---

## Milestone 2G: Testing & Documentation
**Branch**: `feat/milestone-2-testing-docs`
**Time**: ~2-3 hours
**Files Changed**: ~10-12 files

### Changes:
```
tests/
├── test_ranking/
│   ├── test_integration.py     # End-to-end ranking tests
│   ├── test_bm25.py            # Enhanced
│   ├── test_metrics.py         # Enhanced
│   └── test_features.py        # Enhanced
├── test_search/
│   └── test_hybrid_search.py   # Enhanced
└── conftest.py                 # Additional fixtures
README.md                        # Updated with Milestone 2
docs/
├── RANKING.md                   # Ranking documentation
└── EVALUATION.md                # Evaluation guide
pytest.ini                       # Updated configuration
```

### Specific Tasks:
- [ ] Add comprehensive integration tests
- [ ] Enhance existing unit tests
- [ ] Add performance benchmarks
- [ ] Create ranking documentation
- [ ] Write evaluation guide
- [ ] Update README with Milestone 2 examples
- [ ] Add troubleshooting section
- [ ] Document all ranking strategies
- [ ] Add metric interpretation guide
- [ ] Create feature importance analysis

### PR Description Template:
```
## Milestone 2: Testing & Documentation

Comprehensive tests and documentation for ranking and search quality.

### Changes
- Integration tests for complete ranking pipeline
- Enhanced unit tests for all components
- Performance benchmarks
- Ranking strategy documentation
- Evaluation methodology guide
- Updated README with examples
- Troubleshooting guide

### Test Coverage
- BM25 scoring: 95%
- Hybrid search: 92%
- Ranking metrics: 98%
- Feature extraction: 90%
- Evaluation framework: 88%
- **Overall Milestone 2**: 90%+

### Testing
- [ ] All tests pass: `pytest`
- [ ] Coverage meets threshold: `pytest --cov=src --cov-report=html`
- [ ] Integration tests complete: `pytest tests/test_ranking/test_integration.py`
- [ ] Performance acceptable: `pytest tests/test_performance/ -v`
- [ ] Evaluation script works end-to-end

### Performance Benchmarks
- Search latency (hybrid): <200ms (p95)
- Feature extraction: <50ms per document
- BM25 scoring: <100ms for 1000 documents
- NDCG calculation: <10ms for 50 results

### Documentation Added

**RANKING.md**
- Overview of ranking strategies
- When to use semantic vs keyword vs hybrid
- Feature descriptions and importance
- Parameter tuning guide
- Best practices

**EVALUATION.md**
- How to create test queries
- Relevance judgment guidelines
- Running offline evaluations
- Interpreting metrics
- A/B testing methodology

**README Updates**
- Milestone 2 quickstart
- API examples with hybrid search
- Ranking strategy selection guide
- Feature configuration examples


### Example: Complete Ranking Pipeline

from src.search.hybrid_search import HybridSearch
from src.ranking.features import FeatureExtractor
from src.ranking.metrics import calculate_ndcg

# Search with hybrid ranking
search = HybridSearch(strategy="rrf")
results = search.query(
    "OAuth implementation decisions",
    limit=10
)

# Extract features for top results
extractor = FeatureExtractor()
for result in results[:5]:
    features = extractor.extract(query, result)
    print(f"Score: {result.score}, Features: {features}")

# Evaluate with test set
relevance = [3, 3, 2, 1, 1, 0, 0, 0, 0, 0]
ndcg = calculate_ndcg(relevance, k=10)
print(f"NDCG@10: {ndcg:.3f}")


### Troubleshooting Added
- Elasticsearch connection issues
- BM25 scores all zeros
- Hybrid search not improving results
- Feature extraction performance
- NDCG calculation errors
```

**Commit Messages:**
```
test: add comprehensive integration tests for ranking
test: enhance unit tests with edge cases
perf: add performance benchmarks
docs: create ranking strategy documentation
docs: add evaluation methodology guide
docs: update README with Milestone 2 examples
```

---

## Summary: Milestone 2 PRs

| PR | Branch | Time | Files | Focus |
|----|--------|------|-------|-------|
| **2A** | `feat/elasticsearch-integration` | 2-3h | ~8 | Elasticsearch setup |
| **2B** | `feat/bm25-scoring` | 3-4h | ~8 | BM25 implementation |
| **2C** | `feat/hybrid-search` | 3-4h | ~9 | Semantic + BM25 fusion |
| **2D** | `feat/ranking-metrics` | 3-4h | ~7 | MRR, NDCG, metrics |
| **2E** | `feat/ranking-features` | 3-4h | ~8 | Feature engineering |
| **2F** | `feat/evaluation-framework` | 3-4h | ~9 | Offline evaluation |
| **2G** | `feat/milestone-2-testing-docs` | 2-3h | ~12 | Tests, documentation |

**Total**: 19-29 hours across 7 PRs

## PR Workflow

### For Each PR:

1. **Create branch from main**
```bash
git checkout main
git pull origin main
git checkout -b feat/elasticsearch-integration
```

2. **Make changes & commit frequently**
```bash
git add .
git commit -m "feat: add elasticsearch service to docker compose"
# ... more commits ...
```

3. **Push and create PR**
```bash
git push origin feat/elasticsearch-integration
# Create PR on GitHub
```

4. **Self-review checklist**
- [ ] Code runs without errors
- [ ] Tests pass with good coverage
- [ ] Elasticsearch/services start correctly
- [ ] Commit messages are clear
- [ ] PR description is complete
- [ ] Documentation updated

5. **Merge and tag (if completing milestone)**
```bash
git checkout main
git merge feat/elasticsearch-integration
git tag v0.2.0  # After PR 2G
git push origin main --tags
```

## Benefits of This Approach

✅ **Demonstrates IR expertise** - Shows deep understanding of search fundamentals
✅ **Production-quality ranking** - Not just basic search, but sophisticated multi-signal ranking
✅ **Evaluation rigor** - Proper metrics and offline testing like real search teams
✅ **Clean progression** - Each PR builds logically on the previous
✅ **Interview-ready** - Can discuss BM25, NDCG, hybrid search confidently
✅ **Realistic workflow** - Mirrors how search quality teams actually work

## Key Technical Concepts Demonstrated

After Milestone 2, you can discuss:

### Information Retrieval
- **BM25**: Probabilistic ranking function, term frequency saturation, length normalization
- **IDF**: Why "meeting" should rank lower than "OAuth"
- **Hybrid Search**: Combining semantic and lexical matching
- **Reciprocal Rank Fusion**: Rank-based combination vs score-based

### Search Quality
- **MRR**: When you care about the first relevant result
- **NDCG**: Graded relevance and position-aware metrics
- **Precision/Recall**: Trade-offs in search systems
- **Offline Evaluation**: Test sets, relevance judgments, statistical significance

### Feature Engineering
- **Multi-signal ranking**: 40+ features across query-doc, temporal, engagement, authority
- **Feature normalization**: Why it matters for combining signals
- **Feature importance**: Understanding which signals drive ranking

### Architecture
- **Multi-stage retrieval**: Candidate retrieval → feature extraction → ranking
- **Performance**: Latency targets for each stage
- **Flexibility**: Configurable strategies, A/B testable

## Interview Questions You Can Answer

**Q: Explain BM25 and why it's better than TF-IDF.**
> BM25 adds term frequency saturation (diminishing returns after k1) and document length normalization (parameter b). Unlike TF-IDF, it's probabilistic and handles both very short and very long documents better.

**Q: How do you combine semantic and keyword search?**
> Reciprocal Rank Fusion for rank-based combination (robust to score distribution differences) or weighted score fusion with normalized scores. RRF typically performs better without tuning.

**Q: What's the difference between MRR and NDCG?**
> MRR assumes binary relevance and only considers rank of first relevant item - good for navigational queries. NDCG supports graded relevance (0-3) and position discounting - better for informational queries where multiple results matter.

**Q: How do you evaluate search quality offline?**
> Create test query set, collect relevance judgments (0-3 scale), compute metrics (MRR, NDCG@k), compare strategies, run statistical significance tests. Need at least 50+ queries for reliable results.

## Dependencies Between PRs

```
2A (Elasticsearch) ──→ 2B (BM25) ──→ 2C (Hybrid Search)
                              ↓
                         2D (Metrics)
                              ↓
                       2E (Features) ──→ 2F (Evaluation)
                              ↓               ↓
                         2G (Testing & Docs)
```

## Optional Enhancements (Post-Milestone 2)

If time permits, consider these additions:
- Query expansion with synonyms
- Learning to rank (LambdaMART) - save for Milestone 7
- Cross-encoder reranking
- Query performance prediction
- Personalized ranking
- Diversity in results

## Git Commit Convention

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding tests
- `perf:` - Performance improvement
- `refactor:` - Code restructuring
- `chore:` - Maintenance tasks

Examples:
```
feat: implement BM25 scoring with configurable parameters
feat: add reciprocal rank fusion for hybrid search
docs: document ranking metrics and evaluation methodology
test: add integration tests for hybrid search pipeline
perf: optimize feature extraction to <50ms per document
```

## Success Criteria for Milestone 2

After completing all PRs, you should have:

✅ **Working System**
- Elasticsearch running and integrated
- Hybrid search (semantic + BM25) operational
- Multiple ranking strategies available via API
- Feature extraction pipeline functional
- Evaluation framework ready to use

✅ **Measurable Quality**
- Baseline metrics established (MRR, NDCG)
- Hybrid search outperforms single methods
- Can demonstrate quality improvements
- Statistical comparison of strategies

✅ **Production-Quality Code**
- 90%+ test coverage
- Performance benchmarks met
- Comprehensive documentation
- Clean, maintainable architecture

✅ **Interview Readiness**
- Can explain IR fundamentals confidently
- Understand ranking metrics deeply
- Discuss trade-offs in search systems
- Show real implementation, not just theory

---

**Last Updated**: December 2024
**Dependencies**: Milestone 1 completed
**Next**: Milestone 3 - Gap Detection
