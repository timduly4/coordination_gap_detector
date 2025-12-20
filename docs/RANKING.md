# Ranking Strategies Guide

## Overview

This document provides comprehensive guidance on the ranking strategies available in the Coordination Gap Detector system. Understanding these strategies is essential for optimizing search quality and choosing the right approach for your use case.

## Table of Contents

1. [Ranking Strategy Overview](#ranking-strategy-overview)
2. [Strategy Details](#strategy-details)
3. [Feature Descriptions](#feature-descriptions)
4. [When to Use Each Strategy](#when-to-use-each-strategy)
5. [Parameter Tuning](#parameter-tuning)
6. [Best Practices](#best-practices)
7. [Performance Considerations](#performance-considerations)

---

## Ranking Strategy Overview

The system supports three primary ranking strategies:

| Strategy | Type | Best For | Latency | Complexity |
|----------|------|----------|---------|------------|
| **Semantic** | Dense retrieval | Conceptual matches, paraphrasing | Low | Simple |
| **BM25** | Sparse retrieval | Exact keyword matches, technical terms | Low | Simple |
| **Hybrid** | Combined | General-purpose search | Medium | Moderate |

### Quick Decision Tree

```
┌─ Need exact keyword matching? ────────────────────────► BM25
│
├─ Need to understand concepts/paraphrasing? ───────────► Semantic
│
├─ Need best overall results? ──────────────────────────► Hybrid (RRF)
│
└─ Have custom requirements? ───────────────────────────► Hybrid (Weighted)
```

---

## Strategy Details

### 1. Semantic Search

**Description**: Uses dense vector embeddings to find documents semantically similar to the query.

**How it Works**:
- Query and documents are embedded into high-dimensional vectors
- Cosine similarity measures distance between vectors
- Returns documents with highest similarity scores

**Strengths**:
- Understands synonyms and paraphrasing
- Captures conceptual similarity
- Language-agnostic (within model's training)
- Handles query reformulations well

**Weaknesses**:
- May miss exact keyword matches
- Can return false positives with similar topics
- Embedding computation required
- Less interpretable than keyword matching

**Example**:

```python
from src.services.search_service import SearchService

# Query that benefits from semantic understanding
query = "how to authenticate users"

results = await search_service.search(
    query=query,
    ranking_strategy="semantic",
    limit=10
)

# Will match:
# - "OAuth implementation for user login"
# - "Authentication best practices"
# - "User authorization workflows"
```

**Configuration**:

```python
SearchRequest(
    query="your query",
    ranking_strategy="semantic",
    threshold=0.7,  # Minimum similarity score (0-1)
    limit=10
)
```

**Recommended Use Cases**:
- Exploratory searches
- Question answering
- Cross-language search
- Finding conceptually similar discussions
- When users describe problems in their own words

---

### 2. BM25 (Best Match 25)

**Description**: Probabilistic ranking function based on term frequency and inverse document frequency.

**How it Works**:
- Scores documents by relevance to query terms
- Applies term frequency saturation (diminishing returns)
- Normalizes by document length
- Weights terms by rarity (IDF)

**BM25 Formula**:

```
score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1 + 1)) / (f(qi,D) + k1 · (1 - b + b · |D|/avgdl))
```

Where:
- `f(qi,D)` = term frequency of qi in document D
- `|D|` = document length in terms
- `avgdl` = average document length in collection
- `k1` = term frequency saturation parameter (default: 1.5)
- `b` = length normalization parameter (default: 0.75)
- `IDF(qi)` = inverse document frequency of term qi

**Strengths**:
- Excellent for exact keyword matching
- Interpretable scores and explanations
- Handles rare technical terms well
- Fast computation
- Well-studied and proven algorithm

**Weaknesses**:
- Doesn't understand synonyms
- Misses paraphrased queries
- Vocabulary mismatch problems
- Requires text preprocessing (tokenization, stemming)

**Example**:

```python
# Query with specific technical terms
query = "OAuth2 token validation"

results = await search_service.search(
    query=query,
    ranking_strategy="bm25",
    limit=10
)

# Will match:
# - "OAuth2 implementation guide"
# - "Token validation best practices"
# - "OAuth2 security token management"
```

**Configuration**:

```python
from src.ranking.scoring import BM25Scorer

# Custom BM25 parameters
scorer = BM25Scorer(
    k1=1.5,  # Term frequency saturation (1.2-2.0 typical)
    b=0.75   # Length normalization (0.0-1.0)
)

# Using in search
SearchRequest(
    query="OAuth token",
    ranking_strategy="bm25",
    limit=10
)
```

**Parameter Effects**:

**k1 (Term Frequency Saturation)**:
- **Lower (0.5-1.2)**: Faster saturation, multiple mentions matter less
- **Higher (2.0-3.0)**: Slower saturation, more weight to repeated terms
- **Default (1.5)**: Balanced, works well for most collections

**b (Length Normalization)**:
- **0.0**: No length normalization (short docs not favored)
- **1.0**: Full length normalization (short docs strongly favored)
- **Default (0.75)**: Moderate normalization, balanced approach

**Recommended Use Cases**:
- Searching for specific technical terms
- Acronym searches (OAuth, API, JWT)
- Finding exact phrases or code snippets
- When users know precise keywords
- Compliance/audit searches

---

### 3. Hybrid Search

**Description**: Combines semantic and BM25 rankings for best overall quality.

#### 3a. Reciprocal Rank Fusion (RRF)

**Description**: Rank-based fusion that combines results from multiple retrieval methods.

**How it Works**:

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

Where:
- `rank_i(d)` = rank of document d in retrieval method i
- `k` = constant (default: 60)

**Strengths**:
- Robust to score distribution differences
- No parameter tuning required
- Proven to work well in practice
- Handles missing scores gracefully

**Weaknesses**:
- Slightly slower than single method
- Rank-based (ignores score magnitudes)

**Example**:

```python
query = "OAuth implementation best practices"

results = await search_service.search(
    query=query,
    ranking_strategy="hybrid_rrf",
    limit=10
)

# Combines:
# - Semantic: "authentication patterns", "security guidelines"
# - BM25: "OAuth", "implementation", "best practices"
# Result: Best of both worlds
```

**Configuration**:

```python
from src.search.hybrid_search import HybridSearchFusion

fusion = HybridSearchFusion(
    strategy="rrf",
    rrf_k=60  # Typical values: 10-100 (60 is standard)
)
```

**k Parameter Effects**:
- **Lower (10-30)**: More emphasis on top ranks
- **Higher (60-100)**: More balanced across ranks
- **Default (60)**: Standard from research literature

**Recommended Use Cases**:
- **Default choice** for most searches
- When you want robust results without tuning
- Mixed query types (keywords + concepts)
- General-purpose search applications

---

#### 3b. Weighted Score Fusion

**Description**: Score-based fusion with configurable weights for semantic and keyword results.

**How it Works**:

```
score(d) = α · normalize(semantic_score(d)) + β · normalize(bm25_score(d))
```

Where:
- `α` = semantic weight (default: 0.7)
- `β` = keyword weight (default: 0.3)
- `α + β = 1.0`

**Strengths**:
- Fine-grained control over fusion
- Score-based (preserves magnitude information)
- Interpretable weights
- Can optimize for specific use cases

**Weaknesses**:
- Requires weight tuning
- Sensitive to score distributions
- Normalization required

**Example**:

```python
from src.search.hybrid_search import HybridSearchFusion

# Emphasize semantic matching
fusion = HybridSearchFusion(
    strategy="weighted",
    semantic_weight=0.8,  # Higher weight on semantic
    keyword_weight=0.2    # Lower weight on keywords
)

# Emphasize keyword matching
fusion_keyword = HybridSearchFusion(
    strategy="weighted",
    semantic_weight=0.3,  # Lower weight on semantic
    keyword_weight=0.7    # Higher weight on keywords
)
```

**Weight Selection Guidelines**:

| Use Case | Semantic Weight | Keyword Weight | Rationale |
|----------|----------------|----------------|-----------|
| General search | 0.7 | 0.3 | Balanced, slight semantic preference |
| Technical docs | 0.3 | 0.7 | Exact terms matter more |
| Question answering | 0.8 | 0.2 | Concepts matter more |
| Code search | 0.2 | 0.8 | Exact syntax critical |
| Exploratory | 0.9 | 0.1 | Broad conceptual matching |

**Recommended Use Cases**:
- When you have specific requirements
- Domain-specific applications
- After A/B testing to optimize weights
- When you need to explain ranking decisions

---

## Feature Descriptions

### Feature Categories

The ranking system uses 40+ features across multiple categories:

#### 1. Query-Document Similarity (6 features)

| Feature | Description | Range | Importance |
|---------|-------------|-------|------------|
| `semantic_score` | Cosine similarity of embeddings | [0, 1] | High |
| `bm25_score` | BM25 relevance score | [0, ∞) | High |
| `exact_match` | Exact phrase match bonus | {0, 1} | Medium |
| `term_coverage` | % of query terms in document | [0, 1] | High |
| `title_match` | Query match in title/channel | {0, 1} | Medium |
| `entity_overlap` | Shared entities (people, teams) | [0, 1] | Low |

#### 2. Temporal Features (5 features)

| Feature | Description | Range | Importance |
|---------|-------------|-------|------------|
| `recency` | Time since message (exponential decay) | [0, 1] | High |
| `activity_burst` | Recent activity spike detection | [0, 1] | Medium |
| `temporal_relevance` | Time alignment with query context | [0, 1] | Low |
| `edit_freshness` | Time since last edit | [0, 1] | Low |
| `response_velocity` | Reply rate in thread | [0, 1] | Low |

**Recency Formula**:
```
recency = exp(-ln(2) · age_hours / half_life_hours)
```
- Half-life default: 30 days
- Recent messages score higher

#### 3. Engagement Features (6 features)

| Feature | Description | Range | Importance |
|---------|-------------|-------|------------|
| `thread_depth` | Number of replies | [0, ∞) | Medium |
| `participant_count` | Unique participants | [0, ∞) | Medium |
| `reaction_count` | Total reactions | [0, ∞) | Medium |
| `reaction_diversity` | Unique reaction types | [0, ∞) | Low |
| `cross_team_engagement` | Multiple teams involved | {0, 1} | High |
| `view_count` | Message views (if available) | [0, ∞) | Low |

**Interpretation**:
- High engagement suggests important discussions
- Cross-team involvement may indicate coordination gaps
- Diversity of reactions indicates broad interest

#### 4. Source Authority (4 features)

| Feature | Description | Range | Importance |
|---------|-------------|-------|------------|
| `author_seniority` | Author's position/tenure | [0, 1] | Medium |
| `channel_importance` | Channel activity/membership | [0, 1] | Medium |
| `team_influence` | Team's organizational importance | [0, 1] | Low |
| `domain_expertise` | Author expertise in topic area | [0, 1] | Medium |

#### 5. Content Features (3 features)

| Feature | Description | Range | Importance |
|---------|-------------|-------|------------|
| `message_length` | Character count (normalized) | [0, 1] | Low |
| `code_snippet_present` | Contains code blocks | {0, 1} | Medium |
| `link_count` | External references | [0, ∞) | Low |

---

## When to Use Each Strategy

### Decision Matrix

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| User asks a question | Semantic | Understands intent, not just keywords |
| Searching for "OAuth" | BM25 | Exact term match needed |
| Finding similar discussions | Semantic | Concept-based matching |
| Compliance search | BM25 | Exact phrase matching required |
| General search | Hybrid (RRF) | Best overall quality |
| Multi-language content | Semantic | Cross-language understanding |
| Code search | BM25 or Hybrid (0.3/0.7) | Exact syntax matters |
| Exploring topics | Semantic | Broad conceptual matching |
| Finding decisions | Hybrid (RRF) | Needs both keywords and context |

### Performance vs Quality Trade-offs

```
Quality (Relevance)
    ↑
    │                    ╔═══════════╗
    │                    ║  Hybrid   ║
    │               ╔════╣  (RRF)    ║
    │               ║    ╚═══════════╝
    │          ╔════╝
    │     ╔════╣ Semantic
    │     ║    ╚════╗
    │     ║         ║
    │ ════╣         ╚════╗ BM25
    │     ║              ║
    └─────┴──────────────┴─────────────────────►
         Low          Medium          High
                    Latency
```

---

## Parameter Tuning

### BM25 Parameter Tuning

#### Finding Optimal k1

```python
from src.ranking.scoring import BM25Scorer
from src.ranking.metrics import calculate_ndcg

# Test different k1 values
k1_values = [0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
results = {}

for k1 in k1_values:
    scorer = BM25Scorer(k1=k1, b=0.75)
    # Run evaluation with test queries
    ndcg = evaluate_with_test_set(scorer)
    results[k1] = ndcg

# Choose k1 with highest NDCG
best_k1 = max(results, key=results.get)
```

**Guidelines**:
- **Short documents** (tweets, titles): Lower k1 (1.0-1.2)
- **Medium documents** (messages, emails): Default k1 (1.5)
- **Long documents** (articles, docs): Higher k1 (2.0-3.0)

#### Finding Optimal b

```python
# Test different b values
b_values = [0.0, 0.25, 0.5, 0.75, 1.0]

for b in b_values:
    scorer = BM25Scorer(k1=1.5, b=b)
    ndcg = evaluate_with_test_set(scorer)
    # Analyze results
```

**Guidelines**:
- **Similar-length documents**: Lower b (0.0-0.5)
- **Varied-length documents**: Default b (0.75)
- **Strong length bias needed**: Higher b (0.9-1.0)

### Hybrid Fusion Tuning

#### A/B Testing Weights

```python
from src.ranking.evaluation import compare_strategies

# Test different weight combinations
weight_configs = [
    {"semantic": 0.5, "keyword": 0.5},
    {"semantic": 0.6, "keyword": 0.4},
    {"semantic": 0.7, "keyword": 0.3},  # Default
    {"semantic": 0.8, "keyword": 0.2},
]

results = compare_strategies(
    test_queries=test_set,
    configs=weight_configs
)

# Choose config with best MRR/NDCG
```

#### Grid Search for Optimal Weights

```python
import numpy as np

semantic_weights = np.arange(0.1, 1.0, 0.1)
best_score = 0
best_config = None

for sem_weight in semantic_weights:
    kw_weight = 1.0 - sem_weight

    fusion = HybridSearchFusion(
        strategy="weighted",
        semantic_weight=sem_weight,
        keyword_weight=kw_weight
    )

    score = evaluate_fusion(fusion, test_queries)

    if score > best_score:
        best_score = score
        best_config = (sem_weight, kw_weight)

print(f"Best config: {best_config} with score {best_score}")
```

---

## Best Practices

### 1. Start with Defaults

```python
# Use hybrid RRF as default
SearchRequest(
    query="user query",
    ranking_strategy="hybrid_rrf",  # Best default choice
    limit=10
)
```

**Why**: RRF is robust and works well without tuning.

### 2. Optimize for Your Use Case

```python
# Example: Code search application
SearchRequest(
    query="OAuth token validation",
    ranking_strategy="hybrid_weighted",  # Custom weights
    # Configuration would set keyword_weight=0.7
)
```

### 3. Use Offline Evaluation

```bash
# Evaluate before deploying changes
python scripts/evaluate_ranking.py \
  --queries data/test_queries.jsonl \
  --strategies semantic,bm25,hybrid_rrf \
  --metrics mrr,ndcg@10
```

### 4. Monitor Online Metrics

Track in production:
- **Click-through rate (CTR)**: % of searches with clicks
- **Mean Reciprocal Rank (MRR)**: Position of first click
- **Time to success**: Time until user finds answer
- **Abandonment rate**: % of searches with no clicks

### 5. A/B Test Changes

```python
from src.ranking.experiments import ABTest

# Test new strategy against baseline
test = ABTest(
    name="hybrid_weight_tuning",
    control="hybrid_rrf",
    treatment="hybrid_weighted_80_20"
)

# Run for 1-2 weeks, then analyze
results = test.analyze(significance_level=0.05)
```

### 6. Consider Query Types

```python
def choose_strategy(query: str) -> str:
    """Choose strategy based on query characteristics."""

    # Question queries → Semantic
    if query.startswith(("how", "what", "why", "when", "where")):
        return "semantic"

    # Exact phrase queries → BM25
    if '"' in query or query.isupper():
        return "bm25"

    # Default → Hybrid
    return "hybrid_rrf"
```

### 7. Use Query Expansion Carefully

```python
from src.search.query_parser import expand_query

# Expand acronyms
expanded = expand_query("OAuth")  # → "OAuth OR OAuth2 OR OAuth2.0"

# But be careful with semantic search (may over-match)
```

### 8. Leverage Features for Reranking

```python
from src.ranking.features import FeatureExtractor

extractor = FeatureExtractor()

# After initial retrieval, extract features
for result in results[:20]:  # Top 20
    features = extractor.extract(query, result)

    # Rerank using ML model or custom logic
    result['rerank_score'] = ml_model.predict(features)
```

---

## Performance Considerations

### Latency Targets

| Operation | Target (p95) | Typical |
|-----------|--------------|---------|
| Semantic search (100 docs) | 50ms | 20-30ms |
| BM25 scoring (1000 docs) | 100ms | 40-60ms |
| Hybrid fusion (500+500) | 50ms | 20-30ms |
| Feature extraction (1 doc) | 50ms | 10-20ms |
| Complete pipeline | 200ms | 100-150ms |

### Optimization Strategies

#### 1. Use Approximate Nearest Neighbors (ANN)

```python
# For semantic search with large collections
from src.db.vector_store import VectorStore

vector_store = VectorStore(
    index_type="hnsw",  # Hierarchical NSW for fast ANN
    ef_construction=200,
    M=16
)
```

#### 2. Limit Result Sizes

```python
# Retrieve more initially, but limit final results
SearchRequest(
    query="OAuth",
    limit=10,  # Return to user
    # Internally: retrieve_limit=100 for reranking
)
```

#### 3. Cache Collection Statistics

```python
from src.ranking.scoring import calculate_collection_stats

# Calculate once, reuse
stats = calculate_collection_stats(all_documents)
# Cache for 1 hour or until collection changes
```

#### 4. Parallel Retrieval

```python
import asyncio

# Retrieve from semantic and BM25 in parallel
semantic_task = asyncio.create_task(semantic_search(query))
bm25_task = asyncio.create_task(bm25_search(query))

semantic_results = await semantic_task
bm25_results = await bm25_task
```

#### 5. Batch Processing

```python
# Process multiple queries together
from src.ranking.scoring import BM25Scorer

scorer = BM25Scorer(k1=1.5, b=0.75)

# Batch score all documents for all queries
results = scorer.batch_score_multi_query(
    queries=["query1", "query2", "query3"],
    documents=all_documents,
    avg_doc_length=stats["avg_doc_length"],
    term_idfs=term_idfs
)
```

### Memory Optimization

```python
# For large collections, use generators
def score_documents(query, documents):
    """Yield scored documents without loading all into memory."""
    for doc in documents:
        score = calculate_score(query, doc)
        yield (doc, score)

# Use top-k heap to keep only best results
import heapq

top_k = heapq.nlargest(10, score_documents(query, docs), key=lambda x: x[1])
```

---

## Troubleshooting

### Common Issues

#### 1. BM25 Scores All Zero

**Problem**: All BM25 scores return 0.0

**Causes**:
- Query terms not found in any documents
- Case mismatch (query: "OAUTH", docs: "oauth")
- Tokenization issues

**Solution**:
```python
# Ensure case-insensitive matching
query_terms = [term.lower() for term in query.split()]

# Check term presence
print(f"Query terms: {query_terms}")
print(f"Terms in collection: {stats['term_document_frequencies'].keys()}")
```

#### 2. Hybrid Search Not Improving Results

**Problem**: Hybrid search performs worse than individual methods

**Causes**:
- Score normalization issues
- Inappropriate weights
- One method dominating

**Solution**:
```python
# Check score distributions
print(f"Semantic scores: {[r['semantic_score'] for r in semantic_results[:5]]}")
print(f"BM25 scores: {[r['bm25_score'] for r in bm25_results[:5]]}")

# Adjust normalization or weights
fusion = HybridSearchFusion(
    strategy="rrf"  # RRF is more robust to score distributions
)
```

#### 3. Slow Feature Extraction

**Problem**: Feature extraction takes too long

**Causes**:
- Extracting features for too many documents
- Complex features enabled
- Inefficient metadata access

**Solution**:
```python
from src.ranking.feature_config import get_minimal_config

# Use minimal feature set
config = get_minimal_config()
extractor = FeatureExtractor(config=config)

# Only extract for top-k results
for doc in results[:20]:  # Not all results
    features = extractor.extract(query, doc)
```

#### 4. Poor Recall

**Problem**: System doesn't find relevant documents

**Causes**:
- Threshold too high
- Query too specific
- Missing synonyms

**Solution**:
```python
# Lower similarity threshold
SearchRequest(
    query=query,
    threshold=0.5,  # Instead of 0.7
    limit=20  # Retrieve more results
)

# Use hybrid search
ranking_strategy="hybrid_rrf"
```

---

## Further Reading

- **BM25 Paper**: [Robertson & Zaragoza, 2009](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- **Dense Retrieval**: [Karpukhin et al., 2020, DPR](https://arxiv.org/abs/2004.04906)
- **Hybrid Search**: [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- **Evaluation Metrics**: [NDCG, MRR Overview](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

---

**Last Updated**: December 2024
**Version**: 1.0
**Related**: See [EVALUATION.md](./EVALUATION.md) for evaluation methodologies
