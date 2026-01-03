# Ranking Strategies Guide

## Overview

This guide explains the ranking strategies available in the Coordination Gap Detector system. The system supports three main strategies: semantic search, keyword search (BM25), and hybrid search.

**Note:** Advanced features like ML-based ranking, feature extraction, automated parameter tuning, and A/B testing frameworks are planned for future milestones but not yet implemented.

## Table of Contents

1. [Available Strategies](#available-strategies)
2. [Strategy Details](#strategy-details)
3. [When to Use Each Strategy](#when-to-use-each-strategy)
4. [Usage Examples](#usage-examples)
5. [Understanding the Algorithms](#understanding-the-algorithms)
6. [Future Enhancements](#future-enhancements)

---

## Available Strategies

| Strategy | Type | Best For | How It Works |
|----------|------|----------|--------------|
| **semantic** | Dense retrieval | Conceptual matches, paraphrasing | Vector similarity using embeddings |
| **bm25** | Sparse retrieval | Exact keywords, technical terms | Statistical term matching |
| **hybrid_rrf** | Combined | General-purpose (recommended) | Reciprocal Rank Fusion |
| **hybrid_weighted** | Combined | Custom weighting needs | Weighted score combination |

### Quick Decision Guide

```
Need exact keyword matching (e.g., "OAuth2", "JWT")?
    → Use bm25

Need to understand concepts/paraphrasing (e.g., "how to authenticate users")?
    → Use semantic

Want best overall results without tuning?
    → Use hybrid_rrf (recommended default)

Have specific weight preferences?
    → Use hybrid_weighted
```

---

## Strategy Details

### 1. Semantic Search

**What it does**: Finds documents with similar meaning using vector embeddings.

**How it works**:
- Query and documents are converted to 384-dimensional vectors
- Cosine similarity measures how "close" vectors are
- Returns documents with highest similarity scores

**Strengths**:
- Understands synonyms (e.g., "login" matches "authentication")
- Handles paraphrasing well
- Captures conceptual similarity

**Weaknesses**:
- May miss exact keyword matches
- Can return false positives with similar-sounding topics

**Example query**: "how to authenticate users"
- Will match: "OAuth implementation", "user login system", "authentication best practices"

### 2. BM25 Keyword Search

**What it does**: Ranks documents by keyword relevance using statistical term matching.

**How it works**:
- Counts how often query terms appear in documents
- Weights rare terms higher (e.g., "OAuth" more important than "the")
- Normalizes by document length

**Strengths**:
- Excellent for exact keyword matching
- Works well for technical terms and acronyms
- Fast and interpretable

**Weaknesses**:
- Doesn't understand synonyms
- Misses paraphrased content

**Example query**: "OAuth2 PKCE implementation"
- Will match: Documents containing these exact terms

### 3. Hybrid Search (RRF)

**What it does**: Combines semantic and BM25 using Reciprocal Rank Fusion.

**How it works**:
- Runs both semantic and BM25 searches
- Ranks are combined using formula: `score = 1/(k + rank)`
- Results from both methods are merged

**Strengths**:
- Best overall quality without tuning
- Robust to different query types
- No parameter configuration needed

**Weaknesses**:
- Slightly slower than single methods (runs both)

**Recommended as default** for most use cases.

### 4. Hybrid Search (Weighted)

**What it does**: Combines semantic and BM25 using weighted scores.

**How it works**:
- Runs both semantic and BM25 searches
- Scores are normalized and weighted: `score = α·semantic + β·keyword`
- Default weights: semantic=0.7, keyword=0.3

**Strengths**:
- Customizable weights for specific use cases
- Score-based (preserves magnitude information)

**Weaknesses**:
- Requires weight tuning for optimal results

---

## When to Use Each Strategy

| Your Use Case | Recommended Strategy | Why |
|---------------|---------------------|-----|
| General search | `hybrid_rrf` | Best overall quality |
| User asks a question | `semantic` | Understands intent |
| Looking for "OAuth" or "JWT" | `bm25` | Exact term match |
| Finding similar discussions | `semantic` | Concept matching |
| Looking for code snippets | `bm25` | Exact syntax |
| Finding architecture decisions | `hybrid_rrf` | Needs both keywords and context |

---

## Usage Examples

### Semantic Search

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to authenticate users",
    "ranking_strategy": "semantic",
    "limit": 10
  }'
```

### BM25 Keyword Search

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth2 PKCE implementation",
    "ranking_strategy": "bm25",
    "limit": 10
  }'
```

### Hybrid RRF (Recommended)

```bash
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "OAuth implementation decisions",
    "ranking_strategy": "hybrid_rrf",
    "limit": 10
  }'
```

### Comparing All Strategies

```bash
# Create comparison script
cat > /tmp/compare.sh << 'EOF'
#!/bin/bash
QUERY="$1"

echo "=== SEMANTIC ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"semantic\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.content[:60])..."'

echo -e "\n=== BM25 ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"bm25\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.content[:60])..."'

echo -e "\n=== HYBRID RRF ==="
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\", \"ranking_strategy\": \"hybrid_rrf\", \"limit\": 3}" \
  | jq -r '.results[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.content[:60])..."'
EOF

chmod +x /tmp/compare.sh
/tmp/compare.sh "OAuth implementation"
```

---

## Understanding the Algorithms

### Semantic Search: Vector Similarity

**Concept**: Documents and queries are converted to numerical vectors that capture meaning.

```
Query: "user authentication"  →  [0.2, 0.8, 0.1, 0.5, ...]  (384 dimensions)
Doc1:  "OAuth login system"   →  [0.3, 0.7, 0.2, 0.4, ...]
Doc2:  "Database migration"   →  [0.1, 0.1, 0.8, 0.2, ...]

Similarity(Query, Doc1) = 0.92  (high - similar topic)
Similarity(Query, Doc2) = 0.23  (low - different topic)
```

Similarity is measured using **cosine similarity**: higher score (0.0-1.0) means more similar.

### BM25: Statistical Ranking

**Concept**: Scores documents based on how often query terms appear, weighted by rarity.

**Simplified formula**:
```
score(document) = Σ IDF(term) × TF(term, document)
```

Where:
- **TF (Term Frequency)**: How often the term appears in the document
- **IDF (Inverse Document Frequency)**: How rare the term is across all documents

**Example**:
```
Query: "OAuth implementation"

Document: "OAuth implementation guide for OAuth2..."
- "OAuth" appears 2 times → TF = 2
- "OAuth" is rare (appears in 10% of docs) → IDF = high
- "implementation" appears 1 time → TF = 1
- "implementation" is common (50% of docs) → IDF = medium

Score = (2 × high_IDF) + (1 × medium_IDF) = High overall score
```

### Hybrid RRF: Reciprocal Rank Fusion

**Concept**: Combines rankings from multiple search methods using rank positions.

**Formula**:
```
RRF_score(doc) = 1/(k + semantic_rank) + 1/(k + bm25_rank)
```

Where `k = 60` (constant from research literature)

**Example**:
```
Semantic results:         BM25 results:
1. Doc A (rank 1)        1. Doc B (rank 1)
2. Doc C (rank 2)        2. Doc A (rank 2)
3. Doc B (rank 3)        3. Doc D (rank 3)

RRF scores:
Doc A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
Doc B: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
Doc C: 1/(60+2) + 0        = 0.0161
Doc D: 0        + 1/(60+3) = 0.0159

Final ranking: Doc A, Doc B, Doc C, Doc D
```

This balances results from both methods effectively.

---

## Future Enhancements

The following features are planned but not yet implemented:

### Advanced Ranking Features
- ML-based ranking models
- 40+ ranking features (engagement, recency, authority)
- Feature extraction and reranking
- Learning-to-rank algorithms

### Parameter Optimization
- Automated BM25 parameter tuning (k1, b)
- Hybrid weight optimization
- Grid search for optimal configurations
- A/B testing framework

### Query Processing
- Query expansion and reformulation
- Acronym detection and expansion
- Spell correction
- Multi-language support

### Performance Optimization
- Approximate Nearest Neighbor (ANN) indexing
- Result caching
- Parallel retrieval
- Batch processing

---

## Related Documentation

- **[EVALUATION.md](./EVALUATION.md)** - How to evaluate search quality
- **[DEMO.md](./DEMO.md)** - Hands-on examples of all strategies
- **[API_EXAMPLES.md](./API_EXAMPLES.md)** - Comprehensive API usage guide

## Further Reading

### Academic Papers
- **BM25 Algorithm**: [Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- **Dense Retrieval**: [Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering"](https://arxiv.org/abs/2004.04906)
- **Reciprocal Rank Fusion**: [Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

### Related Concepts
- [Cosine Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)
- [TF-IDF and Information Retrieval](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Vector Embeddings Overview](https://en.wikipedia.org/wiki/Word_embedding)

---

**Last Updated**: January 2025
**Status**: Basic ranking strategies implemented; advanced features planned for future milestones
